"""Pruebas de PostgresRepository. Se omiten automáticamente si no hay un
PostgreSQL accesible en DATABASE_URL_TEST (no bloquean pytest en máquinas
sin PostgreSQL instalado, como en desarrollo normal con SQLite)."""

from __future__ import annotations

import os
import time

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
        for t in ("avisos", "accesos_fallidos", "metricas_fragmentos", "metricas", "mensajes",
                  "conversaciones", "usuarios", "documentos"):
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
    """Si falta algo del esquema y otra sesión retiene la tabla, el arranque
    falla con lock timeout en vez de quedarse colgado. (Con el esquema completo
    ya no pide candados: ver test_arranque_no_espera_a_una_escritura_abierta.)"""
    import app.adapters.db_postgres as dbp
    monkeypatch.setattr(dbp, "SCHEMA_LOCK_TIMEOUT", "1s")
    repo = _fresh_repo()
    with repo._conn, repo._conn.cursor() as c:
        c.execute("DROP INDEX idx_metric_ts")  # crearlo de nuevo pide candado sobre metricas
    repo.close()
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


# ---- tiempo de búsqueda, avisos y accesos fallidos (mismo contrato que SQLite) --
def test_tiempo_de_busqueda_avisos_y_accesos_fallidos():
    from datetime import datetime, timedelta, timezone
    repo = _fresh_repo()
    try:
        cid = repo.get_or_create_conversation(repo.get_or_create_user("12345"))
        result = RagResult(Outcome.DERIVED, Reason.LOW_SIMILARITY, "deriva", [], 0.5, "", "", 0,
                           {"total": 0.4, "retrieval": 0.034}, "detalle", [])
        mid = repo.record_metric(cid, None, result)
        t = datetime(2026, 10, 24, 18, tzinfo=timezone.utc)
        repo.record_notice(mid, "enviado", "prueba", "yo@x.mx", "", "", "low_similarity", now=t)
        [notice] = repo.notices()
        assert (notice["id_metrica"], notice["estado"], notice["timestamp"]) ==             (mid, "enviado", "2026-10-24T18:00:00+00:00")
        [row] = repo.metric_rows()
        assert (row.tiempo_busqueda_seg, row.aviso, row.aviso_destinatario) ==             (0.034, "enviado", "yo@x.mx")
        repo.record_failed_login("intruso", "172.19.0.1", "/panel", now=t)
        repo.record_failed_login("otro", None, "/api/metricas", now=t + timedelta(minutes=1))
        assert [(r["usuario"], r["ip"]) for r in repo.failed_logins()] ==             [("otro", None), ("intruso", "172.19.0.1")]
        with pytest.raises(psycopg2.errors.CheckViolation):
            repo.record_notice(None, "quizas", "prueba")
    finally:
        repo.close()


def test_migracion_agrega_tablas_y_columna_nuevas_sin_perder_datos():
    """Base con el esquema de main (ea441ca): sin avisos, accesos_fallidos ni
    tiempo_busqueda_seg. Al arrancar se agregan y las métricas siguen ahí."""
    from app.adapters.db_postgres import PostgresRepository
    repo = _fresh_repo()
    cid = repo.get_or_create_conversation(repo.get_or_create_user("12345"))
    repo.record_metric(cid, None, RagResult(Outcome.ANSWERED, Reason.NONE, "r", [], 0.9, "g", "m",
                                            1, {"total": 1.0}, "ok", []))
    with repo._conn, repo._conn.cursor() as c:
        c.execute("DROP TABLE avisos")
        c.execute("DROP TABLE accesos_fallidos")
        c.execute("ALTER TABLE metricas DROP COLUMN tiempo_busqueda_seg")
    repo.close()
    repo = PostgresRepository(DSN)
    try:
        with repo._conn, repo._conn.cursor() as c:
            c.execute("SELECT table_name FROM information_schema.tables WHERE table_name IN "
                      "('avisos', 'accesos_fallidos')")
            assert {r["table_name"] for r in c.fetchall()} == {"avisos", "accesos_fallidos"}
        [row] = repo.metric_rows()
        assert row.tiempo_busqueda_seg is None and row.aviso is None
    finally:
        repo.close()


def test_arranque_no_espera_lecturas_abiertas_sobre_las_tablas_nuevas(monkeypatch):
    """Con avisos, accesos_fallidos y tiempo_busqueda_seg ya creados, arrancar no
    ejecuta su DDL: no espera aunque otra sesión tenga una lectura abierta sobre
    esas tablas (un ALTER o un CREATE con llave foránea sí esperaría)."""
    import app.adapters.db_postgres as dbp
    monkeypatch.setattr(dbp, "SCHEMA_LOCK_TIMEOUT", "2s")
    _fresh_repo().close()
    holder = psycopg2.connect(DSN)
    try:
        with holder.cursor() as c:  # transacción abierta, como el panel en N15
            for table in ("metricas", "avisos", "accesos_fallidos"):
                c.execute(f"SELECT * FROM {table} LIMIT 1")
        finished, error = _start_in_thread()
        assert finished and error is None, error
    finally:
        holder.rollback()
        holder.close()


def test_resumen_diario_y_columnas_nuevas_de_avisos():
    """Mismo contrato que SQLite; y una tabla avisos de la versión anterior
    (sin tipo_aviso ni fecha_resumen) las recibe al arrancar sin perder filas."""
    from datetime import datetime, timedelta, timezone
    from app.adapters.db_postgres import PostgresRepository
    repo = _fresh_repo()
    uid = repo.get_or_create_user("12345", "Ana")
    cid = repo.get_or_create_conversation(uid)
    t = datetime(2026, 10, 25, 18, tzinfo=timezone.utc)
    q = repo.add_message(cid, "usuario", "¿pregunta?", now=t)
    a = repo.add_message(cid, "asistente", "orientación", now=t)
    result = RagResult(Outcome.DERIVED, Reason.NOT_IN_CONTEXT, "orientación", [], 0.5, "", "",
                       1, {"total": 0.4}, "", [])
    mid = repo.record_metric(cid, a, result, now=t, question_id=q)
    repo.record_notice(mid, "enviado", "coordinadora", "c@x.mx", now=t)
    [row] = repo.derived_between("2026-10-25T06:00:00+00:00", "2026-10-26T06:00:00+00:00")
    assert (row.id_metrica, row.pregunta, row.respuesta, row.nombre, row.aviso_inmediato) == \
        (mid, "¿pregunta?", "orientación", "Ana", "enviado")
    assert repo.summary_sent("2026-10-25") is False
    repo.record_notice(mid, "guardado", "vista_previa", tipo_aviso="resumen",
                       fecha_resumen="2026-10-25", now=t + timedelta(hours=14))
    assert repo.summary_sent("2026-10-25") is True
    with repo._conn, repo._conn.cursor() as c:
        c.execute("ALTER TABLE avisos DROP COLUMN tipo_aviso")
        c.execute("ALTER TABLE avisos DROP COLUMN fecha_resumen")
    repo.close()
    repo = PostgresRepository(DSN)
    try:
        notices = repo.notices()
        assert len(notices) == 2 and {n["tipo_aviso"] for n in notices} == {"inmediato"}
        assert {n["fecha_resumen"] for n in notices} == {None}
    finally:
        repo.close()


def test_arranque_no_espera_a_una_escritura_abierta(monkeypatch):
    """Con el esquema ya creado, construir otro repositorio no debe esperar a
    que termine una escritura en curso (INSERT sin COMMIT) sobre metricas,
    mensajes o conversaciones. Antes, el CREATE INDEX IF NOT EXISTS pedía un
    candado SHARE aunque el índice existiera y esperaba hasta el lock_timeout."""
    import app.adapters.db_postgres as dbp
    monkeypatch.setattr(dbp, "SCHEMA_LOCK_TIMEOUT", "2s")
    repo = _fresh_repo()
    cid = repo.get_or_create_conversation(repo.get_or_create_user("12345"))
    repo.close()
    holder = psycopg2.connect(DSN)
    try:
        with holder.cursor() as c:  # el bot a media escritura
            c.execute("INSERT INTO conversaciones(id_conversacion, id_usuario, fecha_inicio) "
                      "SELECT 'c-abierta', id_usuario, '2026-10-26' FROM conversaciones LIMIT 1")
            c.execute("INSERT INTO mensajes(id_mensaje, id_conversacion, rol, contenido, "
                      "timestamp) VALUES ('m-abierto', %s, 'usuario', 'hola', '2026-10-26')",
                      (cid,))
            c.execute("INSERT INTO metricas(id_metrica, id_conversacion, timestamp, resultado) "
                      "VALUES ('x-abierta', %s, '2026-10-26', 'answered')", (cid,))
        started = time.monotonic()
        finished, error = _start_in_thread()
        elapsed = time.monotonic() - started
        assert finished and error is None, f"esperó {elapsed:.1f} s y falló: {error!r}"
        assert elapsed < 1.5, f"tardó {elapsed:.1f} s: esperó un candado"
    finally:
        holder.rollback()
        holder.close()
