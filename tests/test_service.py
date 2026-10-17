from app.domain import Outcome, Reason
from app.rag.pipeline import RagPipeline
from app.service import ConversationService
from tests.conftest import ScriptedLLM

GOOD = ("La baja temporal se solicita ante la Coordinación con el formato de baja temporal "
        "dentro de los primeros 20 días hábiles del semestre. (Reglamento Escolar.txt)")


def svc(kb, cfg, llm):
    return ConversationService(cfg, kb_repo(cfg), RagPipeline(cfg, kb.embedder, kb.store, llm))


def kb_repo(cfg):
    from app.adapters.db_sqlite import SqliteRepository
    return SqliteRepository(cfg.db_path, cfg.session_timeout_minutes)


def test_handle_message_persists_and_returns_metric_id(kb, cfg):
    s = svc(kb, cfg, ScriptedLLM(GOOD))
    result, metric_id = s.handle_message("tg-1", "Isagi", "¿Cómo solicito la baja temporal?")
    assert result.outcome is Outcome.ANSWERED and metric_id
    row = s.repo._conn.execute("SELECT * FROM metricas WHERE id_metrica=?", (metric_id,)).fetchone()
    assert row["resultado"] == "answered"
    hist = s.repo.get_history(s.repo._conn.execute(
        "SELECT id_conversacion FROM conversaciones").fetchone()[0])
    assert [h["content"] for h in hist] == ["¿Cómo solicito la baja temporal?", GOOD]


def test_handle_message_reuses_user_and_session_across_calls(kb, cfg):
    s = svc(kb, cfg, ScriptedLLM(GOOD))
    s.handle_message("tg-1", "Isagi", "primera pregunta")
    s.handle_message("tg-1", "Isagi", "¿Cómo solicito la baja temporal?")
    n_users = s.repo._conn.execute("SELECT COUNT(*) n FROM usuarios").fetchone()["n"]
    n_convs = s.repo._conn.execute("SELECT COUNT(*) n FROM conversaciones").fetchone()["n"]
    assert n_users == 1 and n_convs == 1


def test_too_long_message_is_derived_without_calling_llm(kb, cfg):
    llm = ScriptedLLM(GOOD)
    s = svc(kb, cfg, llm)
    long_text = "hola " * 400  # excede max_question_chars por mucho
    result, metric_id = s.handle_message("tg-1", "Isagi", long_text)
    assert result.reason is Reason.TOO_LONG and not llm.calls
    row = s.repo._conn.execute("SELECT * FROM metricas WHERE id_metrica=?", (metric_id,)).fetchone()
    assert row["resultado"] == "derived" and row["motivo_derivacion"] == "too_long"


def test_rate_updates_metric(kb, cfg):
    s = svc(kb, cfg, ScriptedLLM(GOOD))
    _, metric_id = s.handle_message("tg-1", "Isagi", "¿Cómo solicito la baja temporal?")
    s.rate(metric_id, 5)
    row = s.repo._conn.execute("SELECT calificacion_usuario FROM metricas WHERE id_metrica=?",
                               (metric_id,)).fetchone()
    assert row["calificacion_usuario"] == 5


def test_derived_answer_has_no_rating_flow_upstream(kb, cfg):
    s = svc(kb, cfg, ScriptedLLM("NO_INFO"))
    result, metric_id = s.handle_message("tg-1", "Isagi", "¿Cuántas horas de residencia?")
    assert result.outcome is Outcome.DERIVED
    # el llamador (CLI/Telegram) decide no ofrecer calificación aquí; el
    # servicio sigue permitiendo registrar una si se intentara
    s.rate(metric_id, 3)
    row = s.repo._conn.execute("SELECT calificacion_usuario FROM metricas WHERE id_metrica=?",
                               (metric_id,)).fetchone()
    assert row["calificacion_usuario"] == 3
