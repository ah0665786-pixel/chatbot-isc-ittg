"""Capa de datos sobre PostgreSQL: mismo esquema y mismos métodos públicos
que SqliteRepository (app/adapters/db_sqlite.py), para que el resto del
sistema (ConversationService, telegram_bot, web_admin) no necesite saber
cuál motor está en uso. Se activa configurando DATABASE_URL en el entorno;
si no está definida, el sistema sigue usando SQLite por defecto.
"""
from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

import psycopg2
import psycopg2.extras

from app.domain import Outcome, RagResult
from app.adapters.db_sqlite import (  # mismos dataclasses y consulta de exportación
    QUESTION_COLUMN, MetricRow, MetricsSummary, export_query)

# Mismo esquema que SQLite (ver db_sqlite.py): TEXT para UUID/fechas por
# portabilidad. ON CONFLICT y CHECK funcionan igual en PostgreSQL.
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

-- Columnas agregadas después de la versión inicial (las bases existentes
-- las reciben aquí al arrancar; en SQLite lo hace SqliteRepository._migrate).
ALTER TABLE metricas ADD COLUMN IF NOT EXISTS id_mensaje_usuario TEXT
    REFERENCES mensajes(id_mensaje) ON DELETE SET NULL;
"""


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


_SCHEMA_LOCK_ID = 727_001  # id arbitrario del candado para crear el esquema


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _uuid() -> str:
    return str(uuid.uuid4())


class PostgresRepository:
    """Mismo contrato que SqliteRepository, sobre PostgreSQL (para AWS)."""

    def __init__(self, dsn: str, session_timeout_minutes: int = 30):
        self.dsn = dsn
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self._lock = threading.Lock()
        self._conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
        self._conn.autocommit = False
        with self._conn, self._conn.cursor() as cur:
            # CREATE TABLE IF NOT EXISTS no es seguro en paralelo en PostgreSQL:
            # si bot y panel arrancan a la vez contra una base nueva, uno falla
            # con UniqueViolation en pg_type. El candado consultivo (se libera
            # al terminar la transacción) hace que creen el esquema por turnos.
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_ID,))
            cur.execute(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator["psycopg2.extensions.cursor"]:
        with self._lock, self._conn, self._conn.cursor() as cur:
            yield cur

    # ---- usuarios (RNF-SE-01) ----------------------------------------------
    def get_or_create_user(self, telegram_id: str, nombre: str = "",
                           now: datetime | None = None) -> str:
        now = _now(now)
        with self._tx() as c:
            c.execute("SELECT id_usuario FROM usuarios WHERE telegram_id=%s",
                      (telegram_id,))
            row = c.fetchone()
            if row:
                return row["id_usuario"]
            uid = _uuid()
            c.execute("INSERT INTO usuarios(id_usuario, telegram_id, nombre, "
                     "fecha_registro) VALUES (%s,%s,%s,%s)",
                     (uid, telegram_id, nombre, _iso(now)))
            return uid

    # ---- sesiones de conversación (RF-C-06, UC-11, HU-10) -------------------
    def get_or_create_conversation(self, user_id: str, canal: str = "telegram",
                                   now: datetime | None = None) -> str:
        now = _now(now)
        with self._tx() as c:
            c.execute(
                "SELECT id_conversacion, fecha_inicio FROM conversaciones "
                "WHERE id_usuario=%s AND fecha_fin IS NULL "
                "ORDER BY fecha_inicio DESC LIMIT 1", (user_id,))
            row = c.fetchone()
            if row:
                c.execute(
                    "SELECT MAX(timestamp) AS t FROM mensajes WHERE id_conversacion=%s",
                    (row["id_conversacion"],))
                last_msg = c.fetchone()["t"]
                last = last_msg or row["fecha_inicio"]
                if now - datetime.fromisoformat(last) <= self.session_timeout:
                    return row["id_conversacion"]
                c.execute("UPDATE conversaciones SET fecha_fin=%s WHERE id_conversacion=%s",
                         (_iso(now), row["id_conversacion"]))
            cid = _uuid()
            c.execute("INSERT INTO conversaciones(id_conversacion, id_usuario, "
                     "fecha_inicio, canal) VALUES (%s,%s,%s,%s)",
                     (cid, user_id, _iso(now), canal))
            return cid

    def close_conversation(self, conversation_id: str, now: datetime | None = None) -> None:
        with self._tx() as c:
            c.execute("UPDATE conversaciones SET fecha_fin=%s "
                     "WHERE id_conversacion=%s AND fecha_fin IS NULL",
                     (_iso(_now(now)), conversation_id))

    # ---- mensajes e historial (RF-I-01, RF-I-02) -----------------------------
    def add_message(self, conversation_id: str, rol: str, contenido: str,
                    tokens: int | None = None, now: datetime | None = None) -> str:
        if rol not in ("usuario", "asistente"):
            raise ValueError("rol debe ser 'usuario' o 'asistente'")
        mid = _uuid()
        with self._tx() as c:
            c.execute("INSERT INTO mensajes(id_mensaje, id_conversacion, rol, "
                     "contenido, timestamp, tokens_utilizados) VALUES (%s,%s,%s,%s,%s,%s)",
                     (mid, conversation_id, rol, contenido, _iso(_now(now)), tokens))
        return mid

    def get_history(self, conversation_id: str, limit_turns: int = 6) -> list[dict]:
        with self._conn.cursor() as c:
            c.execute(
                "SELECT rol, contenido FROM mensajes WHERE id_conversacion=%s "
                "ORDER BY timestamp ASC", (conversation_id,))
            rows = c.fetchall()
        msgs = [{"role": "user" if r["rol"] == "usuario" else "assistant",
                "content": r["contenido"]} for r in rows]
        return msgs[-limit_turns * 2:]

    # ---- métricas (RF-I-04, HU-04, HU-07, HU-08, HU-11, HU-13) ---------------
    def record_metric(self, conversation_id: str, message_id: str | None,
                      result: RagResult, now: datetime | None = None,
                      question_id: str | None = None) -> str:
        """message_id = respuesta del asistente; question_id = mensaje del estudiante."""
        mid = _uuid()
        with self._tx() as c:
            c.execute(
                "INSERT INTO metricas(id_metrica, id_conversacion, id_mensaje, timestamp, "
                "tiempo_respuesta_seg, fragmentos_recuperados, similitud_maxima, resultado, "
                f"motivo_derivacion, intentos, proveedor_llm, modelo_llm, {QUESTION_COLUMN}) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (mid, conversation_id, message_id, _iso(_now(now)),
                 result.timings.get("total"), len(result.hits), result.max_score,
                 result.outcome.value,
                 result.reason.value if result.outcome is Outcome.DERIVED else None,
                 result.attempts, result.provider or None, result.model or None,
                 question_id))
            for i, h in enumerate(result.hits):
                c.execute(
                    "INSERT INTO metricas_fragmentos(id_metrica, posicion, doc_id, "
                    "doc_name, chunk_no, score) VALUES (%s,%s,%s,%s,%s,%s)",
                    (mid, i, h.chunk.doc_id, h.chunk.doc_name, h.chunk.chunk_no, h.score))
        return mid

    def rate_response(self, metric_id: str, calificacion: int) -> None:
        if not 1 <= calificacion <= 5:
            raise ValueError("la calificación debe ser de 1 a 5")
        with self._tx() as c:
            c.execute("UPDATE metricas SET calificacion_usuario=%s WHERE id_metrica=%s",
                     (calificacion, metric_id))

    # ---- documentos (espejo del manifiesto de la base de conocimiento) ------
    def sync_documents(self, docs: list[dict]) -> None:
        with self._tx() as c:
            for d in docs:
                c.execute(
                    "INSERT INTO documentos(id_documento, nombre_archivo, version, "
                    "n_fragmentos, fecha_indexacion, activo) VALUES (%s,%s,%s,%s,%s,1) "
                    "ON CONFLICT(id_documento) DO UPDATE SET nombre_archivo=excluded."
                    "nombre_archivo, version=excluded.version, "
                    "n_fragmentos=excluded.n_fragmentos, "
                    "fecha_indexacion=excluded.fecha_indexacion, activo=1",
                    (d["doc_id"], d["name"], d.get("version"), d["n_chunks"],
                     d["indexed_at"]))

    # ---- reportes (HU-04, HU-13, HU-15) --------------------------------------
    def metric_rows(self, start: str | None = None, end: str | None = None) -> list[MetricRow]:
        """Consultas para exportar a CSV. start inclusivo, end exclusivo (ISO UTC)."""
        sql, params = export_query(start, end, "%s")
        with self._lock, self._conn, self._conn.cursor() as c:
            c.execute(sql, params)
            return [MetricRow(**dict(r)) for r in c.fetchall()]

    def metrics_summary(self, date_from: str | None = None,
                        date_to: str | None = None) -> MetricsSummary:
        where, params = [], []
        if date_from:
            where.append("timestamp >= %s"); params.append(date_from)
        if date_to:
            where.append("timestamp <= %s"); params.append(date_to)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._conn.cursor() as c:
            c.execute(f"SELECT resultado, motivo_derivacion, "
                     f"tiempo_respuesta_seg FROM metricas {clause}", params)
            rows = c.fetchall()
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
        )
