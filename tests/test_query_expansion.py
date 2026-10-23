import pytest

from app.rag.query_expansion import expand_query


@pytest.mark.parametrize("q", [
    "¿Qué opciones de titulación existen?",
    "¿Cómo me puedo titular?",
    "¿Cómo me titulo?",
    "quiero titularme, ¿qué opciones tengo?",
    "¿Qué tengo que hacer para obtener mi título?",
    "TITULACIÓN",
])
def test_preguntas_de_titulacion_se_expanden(q):
    expanded = expand_query(q)
    assert expanded.startswith(q)  # la pregunta original va intacta al inicio
    assert "tipos de proyectos de titulación integral" in expanded


@pytest.mark.parametrize("q", [
    "¿Cuántas horas dura la residencia profesional?",
    "¿Qué requisitos necesito para una baja temporal?",
    "¿Cuál es la capital de Francia?",
])
def test_otras_preguntas_no_cambian(q):
    assert expand_query(q) == q


def test_pipeline_busca_con_la_consulta_expandida(cfg, kb, monkeypatch):
    from app.rag.pipeline import RagPipeline

    pipe = RagPipeline(cfg, kb.embedder, kb.store, llm=None)
    seen = []
    real = pipe.embedder.embed_query
    monkeypatch.setattr(pipe.embedder, "embed_query", lambda t: seen.append(t) or real(t))
    pipe.retrieve("¿Cómo me puedo titular?")
    assert "titulación integral" in seen[0]


@pytest.mark.parametrize("q", [
    "¿Qué requisitos necesito para titularme?",
    "¿Qué necesito para titularme?",
    "¿Qué me piden para titularme?",
    "¿Necesito el inglés para titularme?",
    "¿Cuántos créditos debo tener para la titulación?",
])
def test_preguntas_de_requisitos_usan_la_expansion_de_requisitos(q):
    # Con el texto de "opciones", el fragmento de requisitos caía del lugar 4 al 47
    expanded = expand_query(q)
    assert "requisitos para la titulación integral" in expanded
    assert "tipos de proyectos" not in expanded  # solo una expansión por tema
