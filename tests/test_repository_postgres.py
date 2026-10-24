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


def test_metric_rows_trae_la_pregunta_enlazada():
    repo = _fresh_repo()
    uid = repo.get_or_create_user("12345", "Isagi")
    cid = repo.get_or_create_conversation(uid)
    qid = repo.add_message(cid, "usuario", "¿Cuántas horas dura el servicio social?")
    aid = repo.add_message(cid, "asistente", "Entre 480 y 500 horas.")
    result = RagResult(Outcome.DERIVED, Reason.NOT_IN_CONTEXT, "deriva", [], 0.5, "", "", 1,
                       {"total": 0.4}, "detalle", [])
    mid = repo.record_metric(cid, aid, result, question_id=qid)
    repo.rate_response(mid, 4)
    [row] = repo.metric_rows()
    assert row.pregunta == "¿Cuántas horas dura el servicio social?"
    assert (row.resultado, row.motivo_derivacion, row.calificacion) == \
        ("derived", "not_in_context", 4)


# ---- arranque con otro servicio conectado (RNF-FI-01) ----------------------
# Caso real (2026-10-24): el panel leyó métricas y su sesión quedó "idle in
# transaction"; al reconstruir el bot, su arranque esperó para siempre el
# candado del ALTER TABLE del esquema.

def _start_in_thread(timeout: float = 15.0):
    """Construye un repositorio en otro hilo; regresa (terminó, error)."""
    import threading
    from app.adapters.db_postgres import PostgresRepository
    outcome: dict = {}

    def run():
        try:
            PostgresRepository(DSN).close()
        except Exception as e:  # noqa: BLE001 - se revisa en la prueba
            outcome["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    return not t.is_alive(), outcome.get("error")


def test_lecturas_no_dejan_transaccion_abierta():
    from psycopg2.extensions import TRANSACTION_STATUS_IDLE
    repo = _fresh_repo()
    uid = repo.get_or_create_user("12345", "Isagi")
    cid = repo.get_or_create_conversation(uid)
    try:
        for read in (repo.metrics_summary, repo.metric_rows, lambda: repo.get_history(cid)):
            read()
            assert repo._conn.info.transaction_status == TRANSACTION_STATUS_IDLE,                 "la lectura dejó la sesión 'idle in transaction'"
    finally:
        repo.close()


def test_arranque_no_espera_a_la_lectura_de_otro_servicio():
    panel = _fresh_repo()
    try:
        panel.metric_rows()       # descarga del CSV
        panel.metrics_summary()   # el panel al abrirse; con el código anterior
                                  # esta lectura dejaba la transacción abierta
        finished, error = _start_in_thread()
        assert finished, "el segundo repositorio se quedó esperando el candado (más de 15 s)"
        assert error is None
    finally:
        panel.close()


def test_migracion_agrega_la_columna_si_falta():
    from app.adapters.db_postgres import PostgresRepository
    repo = _fresh_repo()
    with repo._conn, repo._conn.cursor() as c:
        c.execute("ALTER TABLE metricas DROP COLUMN id_mensaje_usuario")
    repo.close()
    repo = PostgresRepository(DSN)
    with repo._conn, repo._conn.cursor() as c:
        c.execute("SELECT 1 FROM information_schema.columns WHERE table_name='metricas' "
                  "AND column_name='id_mensaje_usuario'")
        assert c.fetchone() is not None
    repo.close()


def test_bloqueo_al_arrancar_falla_con_error_claro(monkeypatch):
    """Si otra sesión retiene una tabla, el arranque falla con lock timeout en
    vez de quedarse colgado."""
    import app.adapters.db_postgres as dbp
    monkeypatch.setattr(dbp, "SCHEMA_LOCK_TIMEOUT", "1s")
    _fresh_repo().close()
    holder = psycopg2.connect(DSN)
    try:
        with holder.cursor() as c:
            c.execute("LOCK TABLE metricas IN ACCESS EXCLUSIVE MODE")
        finished, error = _start_in_thread()
        assert finished, "el arranque se quedó colgado en vez de fallar"
        assert isinstance(error, psycopg2.errors.LockNotAvailable)
    finally:
        holder.rollback()
        holder.close()
