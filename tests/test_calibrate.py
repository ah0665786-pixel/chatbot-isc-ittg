import json

from app import calibrate as cal
from app.rag.pipeline import RagPipeline
from tests.conftest import ScriptedLLM


def test_calibration_report(kb, cfg, tmp_path):
    g = tmp_path / "g.jsonl"
    rows = [
        {"q": "¿Cómo solicito la baja temporal?", "answerable": True,
         "expected_doc": "reglamento", "expected_terms": ["baja temporal"]},
        {"q": "¿Cuántas horas dura la residencia profesional?", "answerable": True,
         "expected_doc": "residencia", "expected_terms": ["500 horas"]},
        {"q": "¿Cuál es la capital de Francia?", "answerable": False},
        {"q": "Receta de pozole", "answerable": False},
    ]
    g.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    rep = cal.run(g, RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM("x")), 5)
    assert rep["hit_at_1"] == 1.0 and rep["n_unanswerable"] == 2
    b = rep["threshold_balanced"]
    assert b["recall"] == 1.0 and b["reject_rate"] == 1.0
    cal.print_report(rep, 5)
