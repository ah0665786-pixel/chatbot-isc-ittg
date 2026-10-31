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


@pytest.mark.parametrize("q", [
    "¿Qué pasa si repruebo una materia?",          # "repruebo" no empieza con "reprob"
    "¿Tengo que volver a meter la materia reprobada el siguiente semestre?",
    "Reprobé una materia en repetición, ¿qué sigue?",
    "¿Qué pasa si repruebo el curso especial?",
    "Si repito una asignatura, ¿cuenta lo que ya había pasado?",
])
def test_reprobar_materias_usa_la_expansion_de_repeticion(q):
    assert "Curso de repetición y curso especial" in expand_query(q)


def test_repetir_la_residencia_no_es_repeticion_de_materias():
    assert "Curso de repetición" not in expand_query("¿Puedo repetir la residencia si la repruebo?")


def test_definicion_de_convalidacion():
    assert "definición y caracterización" in expand_query("¿Qué es la convalidación de estudios?")
    # Las demás preguntas de convalidación ya salían bien sin expansión
    assert expand_query("¿Hasta qué semestre puedo pedir la convalidación?") == \
        "¿Hasta qué semestre puedo pedir la convalidación?"


@pytest.mark.parametrize("q,expande", [
    ("¿Cuántos créditos de actividades complementarias necesito?", True),
    ("¿Hasta qué semestre tengo para cubrir las actividades complementarias?", True),
    ("¿Cuántas horas equivale un crédito de actividades complementarias?", True),
    # Con estas la expansión las empeoraba (medido): se dejan tal cual
    ("¿Qué son las actividades complementarias?", False),
    ("¿Cómo se acredita una actividad complementaria?", False),
    ("¿El inglés cuenta como actividad complementaria?", False),
])
def test_actividades_complementarias_solo_creditos_horas_y_plazos(q, expande):
    assert ("Políticas de operación de las actividades complementarias" in expand_query(q)) is expande


@pytest.mark.parametrize("q", [
    "¿Puedo repetir la residencia si la repruebo?",
    "Si no acredito la residencia profesional, ¿la vuelvo a cursar?",
    "¿Puedo hacer la residencia otra vez?",
])
def test_repetir_la_residencia_usa_el_vocabulario_de_unica_ocasion(q):
    # La Guía responde con "se cursa por una única ocasión", sin "repetir" ni "reprobar"
    expanded = expand_query(q)
    assert "única ocasión" in expanded and "segunda asignación" in expanded
    assert "Curso de repetición" not in expanded


@pytest.mark.parametrize("q", ["¿Qué requisitos necesito para la residencia profesional?",
                               "¿Cuántas horas dura la residencia?"])
def test_otras_preguntas_de_residencia_no_usan_la_de_unica_ocasion(q):
    assert "única ocasión" not in expand_query(q)
