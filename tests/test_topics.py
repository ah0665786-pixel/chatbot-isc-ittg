"""Tema de cada consulta para los reportes de la coordinadora (HU-13 CA-01,
HU-15 CA-03, RF-C-01). Ver app/rag/query_expansion.classify_topic."""
import json
from collections import Counter
from pathlib import Path

import pytest

from app.rag.query_expansion import TOPIC_LABELS, classify_topic, topic_label

CASES = Path(__file__).resolve().parent.parent / "data" / "retrieval_cases.jsonl"
# Vocabulario del campo "tema" de retrieval_cases.jsonl -> tema del reporte
CASE_TOPICS = {
    "titulación": "titulacion",
    "residencia profesional": "residencia",
    "servicio social": "servicio_social",
    "bajas": "bajas",
    "convalidación de estudios": "traslados",
    "repetición de materias": "plan_estudios",
    "actividades complementarias": "actividades",
}


def _cases():
    rows = [json.loads(l) for l in CASES.read_text("utf-8").splitlines() if l.strip()]
    # Las que el sistema no debe responder (capital de Francia, recetas...) son "otros"
    return [(r["q"], CASE_TOPICS[r["tema"]] if r.get("answerable") else "otros") for r in rows]


def test_aciertos_por_tema_en_retrieval_cases(capsys):
    hits, totals, errores = Counter(), Counter(), []
    for q, expected in _cases():
        totals[expected] += 1
        got = classify_topic(q)
        if got == expected:
            hits[expected] += 1
        else:
            errores.append(f"{q!r}: esperado {expected}, salió {got}")
    with capsys.disabled():
        print("\nAciertos por tema (data/retrieval_cases.jsonl):")
        for t in sorted(totals, key=lambda t: -totals[t]):
            print(f"  {topic_label(t):<28} {hits[t]:>2}/{totals[t]}")
    assert not errores, "\n".join(errores)


@pytest.mark.parametrize("q,tema", [
    # Temas que retrieval_cases.jsonl no trae
    ("¿Cómo pido un permiso para faltar a clases?", "permisos"),
    ("¿Dónde tramito un justificante médico?", "permisos"),
    ("¿Cuántas faltas puedo tener en una materia?", "permisos"),
    ("Quiero trasladarme a otro Tecnológico", "traslados"),
    ("¿Cómo funciona la equivalencia de estudios?", "traslados"),
    ("¿Cómo me doy de baja definitiva?", "bajas"),
    ("¿Cuántas materias puedo llevar en el semestre?", "plan_estudios"),
    ("¿Cuándo es la reinscripción?", "plan_estudios"),
    ("¿Qué especialidades tiene la retícula de ISC?", "plan_estudios"),
    ("¿Hay curso de verano?", "plan_estudios"),
    # "falta" como verbo no es un permiso
    ("¿Qué me falta para terminar la carrera?", "otros"),
    # Gana el trámite que se quiere hacer, no el que se menciona de paso
    ("¿Puedo titularme por residencia profesional?", "titulacion"),
    ("¿Necesito el servicio social para la residencia?", "residencia"),
    ("¿Las actividades complementarias cuentan créditos del plan de estudios?", "actividades"),
    ("hola", "otros"),
    ("", "otros"),
])
def test_tema_de_preguntas_sueltas(q, tema):
    assert classify_topic(q) == tema


def h(*preguntas):
    """Historial como lo da repo.get_history: preguntas y respuestas alternadas."""
    out = []
    for p in preguntas:
        out += [{"role": "user", "content": p}, {"role": "assistant", "content": "respuesta"}]
    return out


@pytest.mark.parametrize("previas,q,tema", [
    # El tema sale de la pregunta anterior del estudiante, como en search_query
    (["¿Cuántas horas dura el servicio social?"], "¿Y cuántas horas son?", "servicio_social"),
    (["¿Cuántas horas dura el servicio social?"], "¿En cuánto tiempo lo tengo que terminar?",
     "servicio_social"),
    # "créditos" sola sería plan de estudios; en un seguimiento no cuenta
    (["¿Cuándo empiezo la residencia?"], "¿Y cuántos créditos vale?", "residencia"),
    # Cadena de seguimientos: se remonta hasta la última que nombraba su tema
    (["¿Cómo me titulo?", "¿Y cuánto tarda?"], "¿Y eso cuánto cuesta?", "titulacion"),
    # Si el seguimiento nombra su propio tema, gana aunque empiece con "y"
    (["¿Cuántas horas dura el servicio social?"], "¿Y la residencia?", "residencia"),
    # Una pregunta que no es de seguimiento no hereda nada
    (["¿Cómo me titulo?"], "¿Cuál es la capital de Francia?", "otros"),
    # Seguimiento sin conversación previa (sesión nueva)
    ([], "¿Y cuántas horas son?", "otros"),
])
def test_tema_de_preguntas_de_seguimiento(previas, q, tema):
    assert classify_topic(q, h(*previas)) == tema


def test_solo_cuentan_las_preguntas_del_estudiante():
    # La respuesta del asistente menciona otro tema: no debe contar
    history = [{"role": "user", "content": "¿Cómo pido una baja temporal?"},
               {"role": "assistant", "content": "Para la residencia profesional..."}]
    assert classify_topic("¿Y cuánto tiempo dura?", history) == "bajas"


def test_etiquetas():
    assert set(TOPIC_LABELS) == {"plan_estudios", "residencia", "servicio_social", "titulacion",
                                 "bajas", "traslados", "actividades", "permisos", "otros"}
    assert topic_label(None) == "Sin clasificar"
    assert topic_label("servicio_social") == "Servicio social"


def test_el_bot_guarda_el_tema_de_cada_consulta(cfg, kb):
    from app.adapters.db_sqlite import SqliteRepository
    from app.rag.pipeline import RagPipeline
    from app.service import ConversationService
    from tests.conftest import ScriptedLLM

    good = ("La baja temporal se solicita ante la Coordinación con el formato de baja "
            "temporal dentro de los primeros 20 días hábiles del semestre. (Reglamento Escolar.txt)")
    repo = SqliteRepository(cfg.db_path)
    svc = ConversationService(cfg, repo,
                              RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM(good)))
    svc.handle_message("tg-9", "Ana", "¿Cómo solicito la baja temporal?")
    svc.handle_message("tg-9", "Ana", "¿Y en cuánto tiempo?")  # seguimiento
    svc.handle_message("tg-9", "Ana", "x" * (cfg.max_question_chars + 1))  # derivada sin buscar
    assert [r.tema for r in repo.metric_rows()] == ["bajas", "bajas", "otros"]
    repo.close()
