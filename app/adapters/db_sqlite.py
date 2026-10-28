"""Capa de datos: historial de conversaciones, mensajes y métricas (RF-I-02,
RF-I-04, RF-C-06, UC-10, UC-11). SQLite de la librería estándar: cero
dependencias nuevas, cero riesgo de que alguien lo retire. El esquema evita
SQL específico de SQLite (usa TEXT para UUID/fechas) para poder migrar a
PostgreSQL más adelante solo cambiando el adaptador, no el resto del sistema.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from app.domain import Outcome, RagResult
from app.reports import percentile

SCHEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id_usuario      TEXT PRIMARY KEY,
    telegram_id     TEXT NOT NULL UNIQUE,
    nombre          TEXT,
    correo          TEXT,
    rol             TEXT NOT NULL DEFAULT 'estudiante',
    fecha_registro  TEXT NOT NULL,
    activo          INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS conversaciones (
    id_conversacion TEXT PRIMARY KEY,
    id_usuario      TEXT NOT NULL REFERENCES usuarios(id_usuario) ON DELETE CASCADE,
    fecha_inicio    TEXT NOT NULL,
    fecha_fin       TEXT,
    canal           TEXT NOT NULL DEFAULT 'telegram'
);
CREATE INDEX IF NOT EXISTS idx_conv_usuario ON conversaciones(id_usuario);

CREATE TABLE IF NOT EXISTS mensajes (
    id_mensaje        TEXT PRIMARY KEY,
    id_conversacion   TEXT NOT NULL REFERENCES conversaciones(id_conversacion) ON DELETE CASCADE,
    rol               TEXT NOT NULL CHECK(rol IN ('usuario','asistente')),
    contenido         TEXT NOT NULL,
    timestamp         TEXT NOT NULL,
    tokens_utilizados INTEGER
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON mensajes(id_conversacion);

-- 1 fila por respuesta del sistema (1 conversación : N métricas, no 1:1)
CREATE TABLE IF NOT EXISTS metricas (
    id_metrica              TEXT PRIMARY KEY,
    id_conversacion         TEXT NOT NULL REFERENCES conversaciones(id_conversacion) ON DELETE CASCADE,
    id_mensaje              TEXT REFERENCES mensajes(id_mensaje) ON DELETE SET NULL,
    timestamp                TEXT NOT NULL,
    tiempo_respuesta_seg     REAL,
    fragmentos_recuperados   INTEGER,
    similitud_maxima         REAL,
    resultado                TEXT NOT NULL CHECK(resultado IN ('answered','derived')),
    motivo_derivacion        TEXT,
    intentos                 INTEGER,
    proveedor_llm            TEXT,
    modelo_llm               TEXT,
    calificacion_usuario     INTEGER CHECK(calificacion_usuario BETWEEN 1 AND 5)
);
CREATE INDEX IF NOT EXISTS idx_metric_conv ON metricas(id_conversacion);
CREATE INDEX IF NOT EXISTS idx_metric_ts ON metricas(timestamp);

-- Resuelve DOCUMENTOS (N) <-> METRICAS (N) con tabla intermedia en vez de
-- una relación N:N directa sin soporte.
CREATE TABLE IF NOT EXISTS metricas_fragmentos (
    id_metrica  TEXT NOT NULL REFERENCES metricas(id_metrica) ON DELETE CASCADE,
    posicion    INTEGER NOT NULL,
    doc_id      TEXT NOT NULL,
    doc_name    TEXT NOT NULL,
    chunk_no    INTEGER NOT NULL,
    score       REAL NOT NULL,
    PRIMARY KEY (id_metrica, posicion)
);

CREATE TABLE IF NOT EXISTS documentos (
    id_documento     TEXT PRIMARY KEY,
    nombre_archivo   TEXT NOT NULL,
    version          TEXT,
    n_fragmentos     INTEGER,
    fecha_indexacion TEXT,
    activo           INTEGER NOT NULL DEFAULT 1
);
"""

# Tablas agregadas después de la versión inicial. El DDL es el mismo en SQLite
# y PostgreSQL; en PostgreSQL se crean solo si faltan (ver db_postgres.NEW_TABLES).
# Un aviso por consulta escalada (HU-11 CA-03): si se envió, se guardó en
# vista previa o falló, y a quién.
NOTICES_DDL = """
CREATE TABLE IF NOT EXISTS avisos (
    id_aviso          TEXT PRIMARY KEY,
    id_metrica        TEXT REFERENCES metricas(id_metrica) ON DELETE SET NULL,
    timestamp         TEXT NOT NULL,
    modo              TEXT NOT NULL,
    estado            TEXT NOT NULL CHECK(estado IN ('enviado','guardado','fallido')),
    destinatario      TEXT,
    archivo           TEXT,
    detalle           TEXT,
    motivo_derivacion TEXT,
    tipo_aviso        TEXT NOT NULL DEFAULT 'inmediato',  -- inmediato | resumen
    fecha_resumen     TEXT              -- día (AAAA-MM-DD, hora local) que cubre el resumen
)"""
# Intentos fallidos de entrar al panel (HU-01 CA-03, RNF-SE-01). Nunca guarda
# la contraseña intentada.
FAILED_LOGINS_DDL = """
CREATE TABLE IF NOT EXISTS accesos_fallidos (
    id_acceso   TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    usuario     TEXT,
    ip          TEXT,
    ruta        TEXT
)"""


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class MetricsSummary:
    total: int
    answered: int
    derived: int
    derivation_rate: float          # HU-04, HU-13
    precision_rate: float           # RF-RN-03: 1 - tasa_derivacion
    avg_response_seconds: float | None
    by_reason: dict[str, int]       # HU-08/HU-11: motivos de derivación
    p95_response_seconds: float | None = None  # RNF-EF-01: 95 % de consultas <= 30 s


@dataclass
class MetricRow:
    """Una consulta para exportar (HU-04, HU-13)."""
    fecha: str                       # ISO en UTC, tal como se guardó
    pregunta: str | None
    resultado: str                   # 'answered' | 'derived'
    motivo_derivacion: str | None
    tiempo_respuesta_seg: float | None
    calificacion: int | None
    tiempo_busqueda_seg: float | None = None
    aviso: str | None = None              # estado del aviso: enviado/guardado/fallido
    aviso_destinatario: str | None = None
    aviso_tipo: str | None = None         # inmediato | resumen
    tema: str | None = None               # None = consulta anterior a la clasificación


@dataclass
class DerivationRow:
    """Una consulta derivada, para el resumen diario a la coordinadora."""
    id_metrica: str
    fecha: str                       # ISO en UTC
    pregunta: str | None
    respuesta: str | None
    motivo_derivacion: str | None
    telegram_id: str | None
    nombre: str | None
    aviso_inmediato: str | None      # estado del aviso inmediato, si lo hubo


# Columna agregada después de la versión inicial: enlaza la métrica con el
# mensaje del ESTUDIANTE (id_mensaje apunta a la respuesta del asistente).
# Las bases existentes la reciben al arrancar (ver _migrate / SCHEMA de Postgres).
QUESTION_COLUMN = "id_mensaje_usuario"
# Tiempo de la búsqueda vectorial (RNF-EF-03); NULL si la consulta no llegó a buscar.
SEARCH_TIME_COLUMN = "tiempo_busqueda_seg"
# Tema de la consulta (HU-13, HU-15; ver query_expansion.classify_topic). Las
# métricas anteriores quedan en NULL ("Sin clasificar"): no se reclasifican.
TOPIC_COLUMN = "tema"
# Respuestas originales del modelo que no se usaron (JSON: lista de textos con su
# intento y motivo): NO_INFO y respuestas rechazadas por la validación. Antes
# solo vivían en memoria (RagResult.debug) y no había forma de diagnosticar un
# NO_INFO como el de la baja temporal del 28-oct.
MODEL_REPLIES_COLUMN = "respuestas_modelo"
# Veces que se reintentó porque el modelo dijo NO_INFO pese a pasar el umbral (0 o 1).
NO_INFO_RETRIES_COLUMN = "reintentos_sin_info"
# Afirmaciones sin respaldo en los fragmentos (RF-RN-02, app/rag/grounding.py):
# JSON con cada oración, su motivo, el intento y qué se hizo (regenerada,
# quitada, recortada o derivada), y cuántas se quitaron de la respuesta entregada.
CLAIMS_COLUMN = "afirmaciones_sin_respaldo"
REMOVED_CLAIMS_COLUMN = "afirmaciones_quitadas"
def model_replies_json(result: RagResult) -> str | None:
    """Respuestas del modelo no usadas, como JSON (None si no hubo)."""
    return json.dumps(result.debug, ensure_ascii=False) if result.debug else None


def claims_json(result: RagResult) -> str | None:
    """Afirmaciones sin respaldo y qué se hizo con ellas, como JSON (None si no hubo)."""
    claims = result.unsupported_claims
    return json.dumps(claims, ensure_ascii=False) if claims else None


# Columnas agregadas después de crear su tabla: (tabla, columna, definición para ADD COLUMN)
COLUMN_MIGRATIONS = [
    ("metricas", QUESTION_COLUMN, "TEXT REFERENCES mensajes(id_mensaje) ON DELETE SET NULL"),
    ("metricas", SEARCH_TIME_COLUMN, "REAL"),
    ("metricas", TOPIC_COLUMN, "TEXT"),
    ("metricas", MODEL_REPLIES_COLUMN, "TEXT"),
    ("metricas", NO_INFO_RETRIES_COLUMN, "INTEGER NOT NULL DEFAULT 0"),
    ("metricas", CLAIMS_COLUMN, "TEXT"),
    ("metricas", REMOVED_CLAIMS_COLUMN, "INTEGER NOT NULL DEFAULT 0"),
    ("avisos", "tipo_aviso", "TEXT NOT NULL DEFAULT 'inmediato'"),
    ("avisos", "fecha_resumen", "TEXT"),
]


def _last_notice(column: str) -> str:
    """Dato del aviso de una métrica; si tiene inmediato y resumen, el inmediato."""
    return (f"(SELECT a.{column} FROM avisos a WHERE a.id_metrica = m.id_metrica "
            "ORDER BY CASE WHEN a.tipo_aviso = 'resumen' THEN 1 ELSE 0 END, "
            "a.timestamp DESC LIMIT 1)")


def derivations_query(ph: str) -> str:
    """Consultas derivadas entre dos instantes (ISO UTC, inicio incluido y fin
    excluido), con la pregunta, la respuesta y el estudiante. Común a SQLite
    (ph='?') y PostgreSQL (ph='%s')."""
    return (
        "SELECT m.id_metrica, m.timestamp AS fecha, q.contenido AS pregunta, "
        "r.contenido AS respuesta, m.motivo_derivacion, u.telegram_id, u.nombre, "
        "(SELECT a.estado FROM avisos a WHERE a.id_metrica = m.id_metrica "
        " AND a.tipo_aviso = 'inmediato' ORDER BY a.timestamp DESC LIMIT 1) AS aviso_inmediato "
        "FROM metricas m "
        "JOIN conversaciones c ON c.id_conversacion = m.id_conversacion "
        "JOIN usuarios u ON u.id_usuario = c.id_usuario "
        f"LEFT JOIN mensajes q ON q.id_mensaje = m.{QUESTION_COLUMN} "
        "LEFT JOIN mensajes r ON r.id_mensaje = m.id_mensaje "
        f"WHERE m.resultado = 'derived' AND m.timestamp >= {ph} AND m.timestamp < {ph} "
        "ORDER BY m.timestamp")


def export_query(start: str | None, end: str | None, ph: str) -> tuple[str, list]:
    """Consulta de exportación común a SQLite (ph='?') y PostgreSQL (ph='%s').
    start inclusivo, end exclusivo (ISO en UTC).

    La pregunta sale del enlace directo id_mensaje_usuario. Las métricas
    anteriores a esa columna no lo tienen: para ellas se toma el último mensaje
    del estudiante en la conversación hasta la hora de la respuesta (las horas
    van al segundo, así que con dos mensajes en el mismo segundo puede
    emparejar el equivocado; por eso ya no se depende de esto)."""
    where, params = [], []
    if start:
        where.append(f"m.timestamp >= {ph}"); params.append(start)
    if end:
        where.append(f"m.timestamp < {ph}"); params.append(end)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    sql = (
        "SELECT m.timestamp AS fecha, "
        "COALESCE(q.contenido, ("
        "  SELECT u.contenido FROM mensajes u, mensajes a "
        "  WHERE a.id_mensaje = m.id_mensaje AND u.id_conversacion = m.id_conversacion "
        "    AND u.rol = 'usuario' AND u.timestamp <= a.timestamp "
        "  ORDER BY u.timestamp DESC LIMIT 1)) AS pregunta, "
        "m.resultado, m.motivo_derivacion, m.tiempo_respuesta_seg, "
        f"m.calificacion_usuario AS calificacion, m.{SEARCH_TIME_COLUMN}, "
        f"{_last_notice('estado')} AS aviso, "
        f"{_last_notice('destinatario')} AS aviso_destinatario, "
        f"{_last_notice('tipo_aviso')} AS aviso_tipo, m.{TOPIC_COLUMN} "
        f"FROM metricas m LEFT JOIN mensajes q ON q.id_mensaje = m.{QUESTION_COLUMN} "
        f"{clause} ORDER BY m.timestamp")
    return sql, params


class SqliteRepository:
    """Repositorio de historial y métricas. Una conexión compartida y con
    lock: suficiente para el bot (una petición a la vez por usuario) y para
    el panel de administración leyendo en paralelo."""

    def __init__(self, path: Path, session_timeout_minutes: int = 30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """Agrega a bases ya existentes las columnas y tablas que llegaron después."""
        self._conn.execute(NOTICES_DDL)
        self._conn.execute(FAILED_LOGINS_DDL)
        for table, column, definition in COLUMN_MIGRATIONS:
            cols = {r["name"] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in cols:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._conn:
            yield self._conn

    # ---- usuarios (RNF-SE-01) ----------------------------------------------
    def get_or_create_user(self, telegram_id: str, nombre: str = "",
                           now: datetime | None = None) -> str:
        now = _now(now)
        with self._tx() as c:
            row = c.execute("SELECT id_usuario FROM usuarios WHERE telegram_id=?",
                            (telegram_id,)).fetchone()
            if row:
                return row["id_usuario"]
            uid = _uuid()
            c.execute("INSERT INTO usuarios(id_usuario, telegram_id, nombre, "
                     "fecha_registro) VALUES (?,?,?,?)",
                     (uid, telegram_id, nombre, _iso(now)))
            return uid

    # ---- sesiones de conversación (RF-C-06, UC-11, HU-10) -------------------
    def get_or_create_conversation(self, user_id: str, canal: str = "telegram",
                                   now: datetime | None = None) -> str:
        now = _now(now)
        with self._tx() as c:
            row = c.execute(
                "SELECT id_conversacion, fecha_inicio FROM conversaciones "
                "WHERE id_usuario=? AND fecha_fin IS NULL "
                "ORDER BY fecha_inicio DESC LIMIT 1", (user_id,)).fetchone()
            if row:
                last_msg = c.execute(
                    "SELECT MAX(timestamp) AS t FROM mensajes WHERE id_conversacion=?",
                    (row["id_conversacion"],)).fetchone()["t"]
                # sin mensajes aún: la sesión sigue "activa" desde que se abrió
                last = last_msg or row["fecha_inicio"]
                if now - datetime.fromisoformat(last) <= self.session_timeout:
                    return row["id_conversacion"]
                c.execute("UPDATE conversaciones SET fecha_fin=? WHERE id_conversacion=?",
                         (_iso(now), row["id_conversacion"]))
            cid = _uuid()
            c.execute("INSERT INTO conversaciones(id_conversacion, id_usuario, "
                     "fecha_inicio, canal) VALUES (?,?,?,?)",
                     (cid, user_id, _iso(now), canal))
            return cid

    def close_conversation(self, conversation_id: str, now: datetime | None = None) -> None:
        with self._tx() as c:
            c.execute("UPDATE conversaciones SET fecha_fin=? "
                     "WHERE id_conversacion=? AND fecha_fin IS NULL",
                     (_iso(_now(now)), conversation_id))

    # ---- mensajes e historial (RF-I-01, RF-I-02) -----------------------------
    def add_message(self, conversation_id: str, rol: str, contenido: str,
                    tokens: int | None = None, now: datetime | None = None) -> str:
        if rol not in ("usuario", "asistente"):
            raise ValueError("rol debe ser 'usuario' o 'asistente'")
        mid = _uuid()
        with self._tx() as c:
            c.execute("INSERT INTO mensajes(id_mensaje, id_conversacion, rol, "
                     "contenido, timestamp, tokens_utilizados) VALUES (?,?,?,?,?,?)",
                     (mid, conversation_id, rol, contenido, _iso(_now(now)), tokens))
        return mid

    def get_history(self, conversation_id: str, limit_turns: int = 6) -> list[dict]:
        rows = self._conn.execute(
            "SELECT rol, contenido FROM mensajes WHERE id_conversacion=? "
            "ORDER BY timestamp ASC", (conversation_id,)).fetchall()
        msgs = [{"role": "user" if r["rol"] == "usuario" else "assistant",
                "content": r["contenido"]} for r in rows]
        return msgs[-limit_turns * 2:]

    # ---- métricas (RF-I-04, HU-04, HU-07, HU-08, HU-11, HU-13) ---------------
    def record_metric(self, conversation_id: str, message_id: str | None,
                      result: RagResult, now: datetime | None = None,
                      question_id: str | None = None, tema: str | None = None) -> str:
        """message_id = respuesta del asistente; question_id = mensaje del estudiante;
        tema = clave de query_expansion.TOPIC_LABELS (HU-13, HU-15)."""
        mid = _uuid()
        with self._tx() as c:
            c.execute(
                "INSERT INTO metricas(id_metrica, id_conversacion, id_mensaje, timestamp, "
                "tiempo_respuesta_seg, fragmentos_recuperados, similitud_maxima, resultado, "
                f"motivo_derivacion, intentos, proveedor_llm, modelo_llm, {QUESTION_COLUMN}, "
                f"{SEARCH_TIME_COLUMN}, {TOPIC_COLUMN}, {MODEL_REPLIES_COLUMN}, {NO_INFO_RETRIES_COLUMN}, "
                f"{CLAIMS_COLUMN}, {REMOVED_CLAIMS_COLUMN}) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, conversation_id, message_id, _iso(_now(now)),
                 result.timings.get("total"), len(result.hits), result.max_score,
                 result.outcome.value,
                 result.reason.value if result.outcome is Outcome.DERIVED else None,
                 result.attempts, result.provider or None, result.model or None,
                 question_id, result.timings.get("retrieval"), tema,
                 model_replies_json(result), result.no_info_retries,
                 claims_json(result), result.removed_claims))
            for i, h in enumerate(result.hits):
                c.execute(
                    "INSERT INTO metricas_fragmentos(id_metrica, posicion, doc_id, "
                    "doc_name, chunk_no, score) VALUES (?,?,?,?,?,?)",
                    (mid, i, h.chunk.doc_id, h.chunk.doc_name, h.chunk.chunk_no, h.score))
        return mid

    def rate_response(self, metric_id: str, calificacion: int) -> None:
        if not 1 <= calificacion <= 5:
            raise ValueError("la calificación debe ser de 1 a 5")
        with self._tx() as c:
            c.execute("UPDATE metricas SET calificacion_usuario=? WHERE id_metrica=?",
                     (calificacion, metric_id))

    # ---- avisos a la coordinadora (HU-11 CA-03, UC-12) ----------------------
    def record_notice(self, metric_id: str | None, estado: str, modo: str,
                      destinatario: str = "", archivo: str = "", detalle: str = "",
                      motivo: str | None = None, now: datetime | None = None,
                      tipo_aviso: str = "inmediato", fecha_resumen: str | None = None) -> str:
        nid = _uuid()
        with self._tx() as c:
            c.execute("INSERT INTO avisos(id_aviso, id_metrica, timestamp, modo, estado, "
                      "destinatario, archivo, detalle, motivo_derivacion, tipo_aviso, "
                      "fecha_resumen) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (nid, metric_id, _iso(_now(now)), modo, estado, destinatario or None,
                       archivo or None, detalle or None, motivo, tipo_aviso, fecha_resumen))
        return nid

    def summary_sent(self, fecha_resumen: str) -> bool:
        """¿Ya se procesó el resumen de ese día? (evita mandarlo dos veces)."""
        return self._conn.execute(
            "SELECT 1 FROM avisos WHERE tipo_aviso = 'resumen' AND fecha_resumen = ? LIMIT 1",
            (fecha_resumen,)).fetchone() is not None

    def last_notice_at(self, motivos: list[str]) -> str | None:
        """Hora (ISO UTC) del último aviso inmediato enviado o guardado de esos motivos."""
        marks = ",".join("?" * len(motivos))
        return self._conn.execute(
            "SELECT MAX(timestamp) AS t FROM avisos WHERE tipo_aviso = 'inmediato' AND "
            f"estado IN ('enviado','guardado') AND motivo_derivacion IN ({marks})",
            motivos).fetchone()["t"]

    def count_derived(self, motivos: list[str], after: str, until: str) -> int:
        """Consultas derivadas por esos motivos con after < timestamp <= until (ISO UTC)."""
        marks = ",".join("?" * len(motivos))
        return self._conn.execute(
            "SELECT COUNT(*) AS n FROM metricas WHERE resultado = 'derived' AND "
            f"motivo_derivacion IN ({marks}) AND timestamp > ? AND timestamp <= ?",
            (*motivos, after, until)).fetchone()["n"]

    def derived_between(self, start: str, end: str) -> list[DerivationRow]:
        rows = self._conn.execute(derivations_query("?"), (start, end)).fetchall()
        return [DerivationRow(**dict(r)) for r in rows]

    def notices(self, limit: int = 100) -> list[dict]:
        """Avisos, el más reciente primero."""
        rows = self._conn.execute("SELECT * FROM avisos ORDER BY timestamp DESC LIMIT ?",
                                  (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- accesos no autorizados al panel (HU-01 CA-03, RNF-SE-01) ------------
    def record_failed_login(self, usuario: str, ip: str | None, ruta: str,
                            now: datetime | None = None) -> str:
        aid = _uuid()
        with self._tx() as c:
            c.execute("INSERT INTO accesos_fallidos(id_acceso, timestamp, usuario, ip, ruta) "
                      "VALUES (?,?,?,?,?)", (aid, _iso(_now(now)), usuario, ip, ruta))
        return aid

    def failed_logins(self, limit: int = 100) -> list[dict]:
        """Intentos fallidos, el más reciente primero."""
        rows = self._conn.execute("SELECT * FROM accesos_fallidos ORDER BY timestamp DESC "
                                  "LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- documentos (espejo del manifiesto de la base de conocimiento) ------
    def sync_documents(self, docs: list[dict]) -> None:
        with self._tx() as c:
            for d in docs:
                c.execute(
                    "INSERT INTO documentos(id_documento, nombre_archivo, version, "
                    "n_fragmentos, fecha_indexacion, activo) VALUES (?,?,?,?,?,1) "
                    "ON CONFLICT(id_documento) DO UPDATE SET nombre_archivo=excluded."
                    "nombre_archivo, version=excluded.version, "
                    "n_fragmentos=excluded.n_fragmentos, "
                    "fecha_indexacion=excluded.fecha_indexacion, activo=1",
                    (d["doc_id"], d["name"], d.get("version"), d["n_chunks"],
                     d["indexed_at"]))

    # ---- reportes (HU-04, HU-13, HU-15) --------------------------------------
    def metric_rows(self, start: str | None = None, end: str | None = None) -> list[MetricRow]:
        """Consultas para exportar a CSV. start inclusivo, end exclusivo (ISO UTC)."""
        sql, params = export_query(start, end, "?")
        return [MetricRow(**dict(r)) for r in self._conn.execute(sql, params).fetchall()]

    def metrics_summary(self, date_from: str | None = None,
                        date_to: str | None = None) -> MetricsSummary:
        where, params = [], []
        if date_from:
            where.append("timestamp >= ?"); params.append(date_from)
        if date_to:
            where.append("timestamp <= ?"); params.append(date_to)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._conn.execute(f"SELECT resultado, motivo_derivacion, "
                                  f"tiempo_respuesta_seg FROM metricas {clause}",
                                  params).fetchall()
        total = len(rows)
        answered = sum(1 for r in rows if r["resultado"] == "answered")
        derived = total - answered
        by_reason: dict[str, int] = {}
        for r in rows:
            if r["motivo_derivacion"]:
                by_reason[r["motivo_derivacion"]] = by_reason.get(r["motivo_derivacion"], 0) + 1
        times = [r["tiempo_respuesta_seg"] for r in rows if r["tiempo_respuesta_seg"] is not None]
        return MetricsSummary(
            total=total, answered=answered, derived=derived,
            derivation_rate=(derived / total) if total else 0.0,
            precision_rate=(answered / total) if total else 0.0,
            avg_response_seconds=(sum(times) / len(times)) if times else None,
            by_reason=by_reason,
            p95_response_seconds=percentile(times),
        )
