"""Pruebas de PostgresRepository. Se omiten automáticamente si no hay un
PostgreSQL accesible en DATABASE_URL_TEST (no bloquean pytest en máquinas
sin PostgreSQL instalado, como en desarrollo normal con SQLite)."""

from __future__ import annotations

import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from app.domain import Hit, Outcome, RagResult, Reason, Chunk  # noqa: E402

DSN = os.environ.get("DATABASE_URL_TEST")

pytestmark = pytest.mark.skipif(
    not DSN, reason="DATABASE_URL_TEST no está configurada: se omiten pruebas de PostgreSQL"
)


def _fresh_repo():
    from app.adapters.db_postgres import PostgresRepository
    repo = PostgresRepository(DSN, session_timeout_minutes=30)
    # Limpieza para que cada prueba empiece en blanco.
    with repo._conn, repo._conn.cursor() as c:
        for t in ("metricas_fragmentos", "metricas", "mensajes", "conversaciones",
                  "usuarios", "documentos"):
            c.execute(f"DELETE FROM {t}")
    return repo


def test_get_or_create_user_is_idempotent():
    repo = _fresh_repo()
    uid1 = repo.get_or_create_user("12345", "Isagi")
    uid2 = repo.get_or_create_user("12345", "Isagi")
    assert uid1 == uid2


def test_conversation_and_messages_roundtrip():
    repo = _fresh_repo()
    uid = repo.get_or_create_user("12345", "Isagi")
    cid = repo.get_or_create_conversation(uid)
    repo.add_message(cid, "usuario", "hola")
    repo.add_message(cid, "asistente", "hola, ¿en qué te ayudo?")
    history = repo.get_history(cid)
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_record_metric_and_summary():
    repo = _fresh_repo()
    uid = repo.get_or_create_user("12345", "Isagi")
    cid = repo.get_or_create_conversation(uid)
    chunk = Chunk(doc_id="d1", doc_name="Reglamento.pdf", chunk_no=1, text="texto")
    result = RagResult(Outcome.ANSWERED, Reason.NONE, "respuesta", [Hit(chunk, 0.9)],
                       0.9, "groq", "gpt-oss-120b", 1, {"total": 1.2}, "ok", [])
    repo.record_metric(cid, None, result)
    summary = repo.metrics_summary()
    assert summary.total == 1
    assert summary.answered == 1
