"""Pruebas de recuperación contra el índice y el modelo de embeddings REALES.

Verifican que, para cada pregunta de data/retrieval_cases.jsonl que sí tiene
respuesta en los documentos, el fragmento correcto llegue al LLM (que quede
dentro de los top_k). Nacieron de un fallo real: "¿Cómo me puedo titular?"
dejaba en el lugar 43 el fragmento con las 10 opciones de titulación.

Se omiten si en esta máquina no hay índice (data/kb) o el modelo no está
descargado, igual que las pruebas de PostgreSQL. Para ver el detalle con
puntajes:  python -m app.cli calibrate data/retrieval_cases.jsonl

Campo "estado" (2026-11-05, base solo con el Manual de Lineamientos y el
Reglamento de Estudiantes): sin él, el caso sigue vigente y debe pasar. Los
marcados se quedan en el archivo, cada uno con su "propuesta":
  corregir_pagina_manual  el Manual repite lo del lineamiento retirado; basta
                          apuntar a esa página (se prueba que la propuesta pasa)
  pierde_recuperacion     el Manual lo dice, pero el fragmento ya no llega al top 5
  respuesta_cambia        el Manual lo dice distinto: hay que cambiar lo esperado
  fuera_de_base           solo estaba en un documento retirado: debe derivarse
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.calibrate import _relevant
from app.rag import guards
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


VIGENTE = "vigente"
ESTADOS = {VIGENTE, "corregir_pagina_manual", "pierde_recuperacion", "respuesta_cambia",
           "fuera_de_base"}


def _param(case: dict):
    """Los casos marcados corren igual, pero como falla esperada con su motivo:
    si un ajuste los arregla, pytest lo muestra (XPASS) sin romper la suite."""
    estado = case.get("estado", VIGENTE)
    if estado == VIGENTE:
        return pytest.param(case, id=case["q"])
    nota = case.get("propuesta", {}).get("nota", "")
    return pytest.param(case, id=case["q"],
                        marks=pytest.mark.xfail(reason=f"{estado}: {nota}", strict=False))


def test_estados_y_propuestas_bien_escritos():
    for c in _cases(True):
        estado = c.get("estado", VIGENTE)
        assert estado in ESTADOS, (c["q"], estado)
        if estado != VIGENTE:
            assert c.get("propuesta", {}).get("nota"), f"{c['q']}: falta la propuesta"


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


@pytest.mark.parametrize("case", [_param(c) for c in _cases(True)])
def test_el_fragmento_correcto_llega_al_llm(pipeline, case):
    hits = pipeline.retrieve(case["q"])  # top_k de la configuración
    found = [i for i, h in enumerate(hits, 1) if _relevant(h, case)]
    listado = "\n".join(f"  {i}. {h.score:.3f} {h.chunk.doc_name} {h.chunk.section}"
                        for i, h in enumerate(hits, 1))
    assert found, (f"el fragmento esperado no está en los top {pipeline.cfg.top_k}:\n{listado}")
    assert hits[found[0] - 1].score >= pipeline.cfg.sim_threshold, "quedó bajo el umbral"


def test_unica_ocasion_de_la_residencia_queda_primero(pipeline):
    # 30-oct: con el fragmento 4.º el modelo dijo NO_INFO (main y rama). Con la
    # expansión debe quedar en el primer o segundo lugar.
    hits = pipeline.retrieve("¿Puedo repetir la residencia si la repruebo?")
    found = [i for i, h in enumerate(hits, 1)
             if "unica ocasion" in " ".join(guards.normalize(h.chunk.text).split())]
    assert found and found[0] <= 2, [f"{h.score:.3f} {h.chunk.doc_name} {h.chunk.section}" for h in hits]


@pytest.mark.parametrize("case", [c for c in _cases(True)
                                  if c.get("estado") == "corregir_pagina_manual"],
                         ids=lambda c: c["q"])
def test_la_propuesta_con_la_pagina_del_manual_recupera_el_fragmento(pipeline, case):
    """Aplicar la propuesta (expected_doc / expected_terms del Manual) basta para
    que el caso vuelva a pasar: así se puede corregir sin volver a medir."""
    fixed = {**case, **{k: v for k, v in case["propuesta"].items() if k != "nota"}}
    hits = pipeline.retrieve(fixed["q"])
    found = [h for h in hits if _relevant(h, fixed) and h.score >= pipeline.cfg.sim_threshold]
    assert found, [f"{h.score:.3f} {h.chunk.doc_name} {h.chunk.section}" for h in hits]
