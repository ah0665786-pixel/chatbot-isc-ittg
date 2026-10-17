from datetime import datetime, timedelta, timezone

import pytest

from app.adapters.db_sqlite import SqliteRepository
from app.domain import Chunk, Hit, Outcome, RagResult, Reason


def rag_answered(score=0.9, hits=2, secs=1.2, provider="groq", model="oss-120b"):
    h = [Hit(Chunk("reglamento", "Reglamento.pdf", i, f"texto {i}"), score) for i in range(hits)]
    return RagResult(Outcome.ANSWERED, Reason.NONE, "respuesta", h, score, provider, model,
                     1, {"total": secs}, "ok")


def rag_derived(reason=Reason.LOW_SIMILARITY, score=0.3, secs=0.5):
    return RagResult(Outcome.DERIVED, reason, "mensaje de derivación", [], score, "", "",
                     0, {"total": secs}, "detalle")


@pytest.fixture
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "app.db", session_timeout_minutes=30)
    yield r
    r.close()


def test_user_creation_is_idempotent_by_telegram_id(repo):
    u1 = repo.get_or_create_user("tg-1", "Isagi")
    u2 = repo.get_or_create_user("tg-1", "Isagi otra vez")
    assert u1 == u2


def test_conversation_reused_within_timeout_new_session_after(repo):
    uid = repo.get_or_create_user("tg-1")
    t0 = datetime.now(timezone.utc)
    c1 = repo.get_or_create_conversation(uid, now=t0)
    repo.add_message(c1, "usuario", "hola", now=t0)
    c2 = repo.get_or_create_conversation(uid, now=t0 + timedelta(minutes=10))
    assert c1 == c2  # dentro del umbral de inactividad: misma sesión

    c3 = repo.get_or_create_conversation(uid, now=t0 + timedelta(minutes=45))
    assert c3 != c1  # fuera del umbral: nueva sesión independiente (HU-10, UC-11)


def test_conversation_without_prior_messages_reused(repo):
    uid = repo.get_or_create_user("tg-1")
    t0 = datetime.now(timezone.utc)
    c1 = repo.get_or_create_conversation(uid, now=t0)
    c2 = repo.get_or_create_conversation(uid, now=t0 + timedelta(minutes=5))
    assert c1 == c2


def test_close_conversation_forces_new_session(repo):
    uid = repo.get_or_create_user("tg-1")
    c1 = repo.get_or_create_conversation(uid)
    repo.close_conversation(c1)
    c2 = repo.get_or_create_conversation(uid)
    assert c1 != c2


def test_add_message_rejects_invalid_role(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    with pytest.raises(ValueError):
        repo.add_message(cid, "robot", "hola")


def test_history_order_and_trim(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    for i in range(10):
        repo.add_message(cid, "usuario" if i % 2 == 0 else "asistente", f"m{i}")
    hist = repo.get_history(cid, limit_turns=3)
    assert len(hist) == 6
    assert [h["content"] for h in hist] == [f"m{i}" for i in range(4, 10)]
    assert hist[0]["role"] == "user" and hist[1]["role"] == "assistant"


def test_record_metric_answered_stores_fragments_and_provider(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    mid = repo.add_message(cid, "asistente", "respuesta")
    res = rag_answered(score=0.87, hits=3)
    metric_id = repo.record_metric(cid, mid, res)
    row = repo._conn.execute("SELECT * FROM metricas WHERE id_metrica=?", (metric_id,)).fetchone()
    assert row["resultado"] == "answered" and row["motivo_derivacion"] is None
    assert row["fragmentos_recuperados"] == 3 and abs(row["similitud_maxima"] - 0.87) < 1e-9
    assert row["proveedor_llm"] == "groq" and row["modelo_llm"] == "oss-120b"
    frags = repo._conn.execute(
        "SELECT COUNT(*) n FROM metricas_fragmentos WHERE id_metrica=?", (metric_id,)).fetchone()
    assert frags["n"] == 3


def test_record_metric_derived_stores_reason_and_no_provider(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    res = rag_derived(Reason.OUT_OF_SCOPE, score=0.1)
    metric_id = repo.record_metric(cid, None, res)
    row = repo._conn.execute("SELECT * FROM metricas WHERE id_metrica=?", (metric_id,)).fetchone()
    assert row["resultado"] == "derived" and row["motivo_derivacion"] == "out_of_scope"
    assert row["proveedor_llm"] is None


def test_rate_response_bounds(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    mid = repo.record_metric(cid, None, rag_answered())
    repo.rate_response(mid, 5)
    with pytest.raises(ValueError):
        repo.rate_response(mid, 6)
    with pytest.raises(ValueError):
        repo.rate_response(mid, 0)


def test_metrics_summary_rates_and_reasons(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    repo.record_metric(cid, None, rag_answered(secs=1.0))
    repo.record_metric(cid, None, rag_answered(secs=3.0))
    repo.record_metric(cid, None, rag_derived(Reason.LOW_SIMILARITY))
    repo.record_metric(cid, None, rag_derived(Reason.OUT_OF_SCOPE))
    repo.record_metric(cid, None, rag_derived(Reason.LOW_SIMILARITY))

    s = repo.metrics_summary()
    assert s.total == 5 and s.answered == 2 and s.derived == 3
    assert s.precision_rate == pytest.approx(0.4)
    assert s.derivation_rate == pytest.approx(0.6)
    assert s.avg_response_seconds == pytest.approx(1.1)  # promedio incluye derivadas (RNF-EF-01)
    assert s.by_reason == {"low_similarity": 2, "out_of_scope": 1}


def test_metrics_summary_empty(repo):
    s = repo.metrics_summary()
    assert s.total == 0 and s.precision_rate == 0.0 and s.avg_response_seconds is None


def test_metrics_summary_date_filter(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    old = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 9, 1, tzinfo=timezone.utc)
    repo.record_metric(cid, None, rag_answered(), now=old)
    repo.record_metric(cid, None, rag_answered(), now=new)
    s = repo.metrics_summary(date_from="2026-06-01")
    assert s.total == 1


def test_sync_documents_upserts(repo):
    docs = [{"doc_id": "reglamento", "name": "Reglamento.pdf", "version": "v1",
            "n_chunks": 34, "indexed_at": "2026-09-01T00:00:00+00:00"}]
    repo.sync_documents(docs)
    docs[0].update(version="v2", n_chunks=40)
    repo.sync_documents(docs)
    row = repo._conn.execute("SELECT * FROM documentos WHERE id_documento='reglamento'").fetchone()
    assert row["version"] == "v2" and row["n_fragmentos"] == 40


def test_foreign_key_cascade_deletes_children(repo):
    uid = repo.get_or_create_user("tg-1")
    cid = repo.get_or_create_conversation(uid)
    repo.add_message(cid, "usuario", "hola")
    with repo._tx() as c:
        c.execute("DELETE FROM usuarios WHERE id_usuario=?", (uid,))
    assert repo._conn.execute("SELECT COUNT(*) n FROM conversaciones").fetchone()["n"] == 0
    assert repo._conn.execute("SELECT COUNT(*) n FROM mensajes").fetchone()["n"] == 0
