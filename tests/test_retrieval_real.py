"""Pruebas de recuperación contra el índice y el modelo de embeddings REALES.

Verifican que, para cada pregunta de data/retrieval_cases.jsonl que sí tiene
respuesta en los documentos, el fragmento correcto llegue al LLM (que quede
dentro de los top_k). Nacieron de un fallo real: "¿Cómo me puedo titular?"
dejaba en el lugar 43 el fragmento con las 10 opciones de titulación.

Se omiten si en esta máquina no hay índice (data/kb) o el modelo no está
descargado, igual que las pruebas de PostgreSQL. Para ver el detalle con
puntajes:  python -m app.cli calibrate data/retrieval_cases.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.calibrate import _relevant
from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "retrieval_cases.jsonl"
KB_DIR = Path(os.environ.get("KB_DIR", ROOT / "data" / "kb"))

pytestmark = pytest.mark.skipif(
    not (KB_DIR / "manifest.json").exists(),
    reason="no hay índice real en data/kb: se omiten las pruebas de recuperación real")


def _cases(answerable: bool) -> list[dict]:
    rows = [json.loads(l) for l in CASES.read_text("utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("answerable", True) is answerable]


@pytest.fixture(scope="module")
def pipeline():
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.pipeline import RagPipeline
    from app.runtime import build_embedder

    cfg = Settings.from_env(load_dotenv_file=False)
    cfg = Settings(**{**cfg.__dict__, "kb_dir": KB_DIR})
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nunca descargar el modelo en una prueba
    try:
        emb = build_embedder(cfg)
        emb.embed_query("prueba")
    except Exception as e:  # sin sentence-transformers o sin el modelo en caché
        pytest.skip(f"modelo de embeddings no disponible: {e}")
    return RagPipeline(cfg, emb, KnowledgeBase.open(cfg, emb).store, llm=None)


@pytest.mark.parametrize("case", _cases(True), ids=lambda c: c["q"])
def test_el_fragmento_correcto_llega_al_llm(pipeline, case):
    hits = pipeline.retrieve(case["q"])  # top_k de la configuración
    found = [i for i, h in enumerate(hits, 1) if _relevant(h, case)]
    listado = "\n".join(f"  {i}. {h.score:.3f} {h.chunk.doc_name} {h.chunk.section}"
                        for i, h in enumerate(hits, 1))
    assert found, (f"el fragmento esperado no está en los top {pipeline.cfg.top_k}:\n{listado}")
    assert hits[found[0] - 1].score >= pipeline.cfg.sim_threshold, "quedó bajo el umbral"
