"""Capa de datos: historial de conversaciones, mensajes y métricas (RF-I-02,
RF-I-04, RF-C-06, UC-10, UC-11). SQLite de la librería estándar: cero
dependencias nuevas, cero riesgo de que alguien lo retire. El esquema evita
SQL específico de SQLite (usa TEXT para UUID/fechas) para poder migrar a
PostgreSQL más adelante solo cambiando el adaptador, no el resto del sistema.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from app.domain import Outcome, RagResult

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
                      result: RagResult, now: datetime | None = None) -> str:
        mid = _uuid()
        with self._tx() as c:
            c.execute(
                "INSERT INTO metricas(id_metrica, id_conversacion, id_mensaje, timestamp, "
                "tiempo_respuesta_seg, fragmentos_recuperados, similitud_maxima, resultado, "
                "motivo_derivacion, intentos, proveedor_llm, modelo_llm) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, conversation_id, message_id, _iso(_now(now)),
                 result.timings.get("total"), len(result.hits), result.max_score,
                 result.outcome.value,
                 result.reason.value if result.outcome is Outcome.DERIVED else None,
                 result.attempts, result.provider or None, result.model or None))
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
        )
