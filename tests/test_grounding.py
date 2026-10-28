"""Fundamentación por afirmación (RF-RN-02): caso real de la baja temporal
fuera de plazo (Manual de Lineamientos TecNM, pág. 46, numeral 5.4.6.4).

El bot respondió que el Comité Académico "evalúa y notifica a Servicios
Escolares" y que "el resto del trámite sigue los mismos pasos". El documento
solo dice que, pasados los 20 días hábiles, el estudiante manifiesta su
determinación por escrito al Comité Académico; quien notifica a Servicios
Escolares es el Jefe(a) de División, en el trámite dentro de plazo."""
import json

import pytest

from app.adapters.llm_openai_compat import LLMUnavailable
from app.domain import Chunk, Hit, Outcome, Reason
from app.grounding_eval import CASES, check_facts, load_cases
from app.rag import grounding
from app.rag.pipeline import CLAIMS_RETRY_NOTE, SYSTEM_PROMPT, RagPipeline
from tests.conftest import ScriptedLLM

# Texto tal como sale del PDF (con sus palabras partidas: "parti r")
MANUAL_83 = """MANUAL DE LINEAMIENTOS ACADÉMICO-ADMINISTRATIVOS DEL TECNM 27 5.4.6.2 Para \
realizar este trámite, el estudiante manifiesta su determinación por escrito a la División de \
Estudios Profesionales o su equivalente en los Institutos Tecnológicos Descentralizados, quien \
notifica al Departamento de Servicios Escolares o su equivalente en los Institutos Tecnológicos \
Descentralizados, si procede la baja. Lo ant erior a excepción de lo que se indique en los \
lineamientos vigentes referentes al Servicio Social y Residencia Profesional (ver CAPÍTULO 11 y \
CAPÍTULO 12). 5.4.6.3 Una baja parcial autorizada no registra calificación en la asignatura. De \
la baja temporal: 5.4.6.4 Todo estudiante que haya cursado al menos un semestre en el Instituto, \
tiene derecho a solicitar baja temporal en la totalidad de las asignaturas en que esté inscrito, \
dentro de los 20 días hábiles a parti r del inicio oficial de los cursos."""
MANUAL_84 = """Para realizar este trámite, el estudiante manifiesta su determinación por escrito \
al (a la) Jefe(a) de la División de Estudios Profesionales o su equivalente en los Institutos \
Tecnológicos Descentralizados, quien notifica al (a la) Jefe(a) del Departamento de Servicios \
Escolares si procede la baja. Si la solicitud no se realizó en los 20 días hábiles, el \
estudiante debe manifestar su determinación por escrito al Comité Académico. Lo anterior a \
excepción de lo que se indique en los lineamientos vigentes referentes al Servicio Social y \
Residencia Profesional (ver CAPÍTULO 11 y CAPÍTULO 12). 5.4.6.5 Una baja temporal autorizada no \
registra calificación en las asignaturas."""
DOC = "Manual_de_Lineamientos_TecNM.pdf"
HITS = [Hit(Chunk("manual-de-lineamientos-tecnm", DOC, 84, MANUAL_84, "pág. 46"), 0.868),
        Hit(Chunk("manual-de-lineamientos-tecnm", DOC, 83, MANUAL_83, "pág. 46"), 0.863)]
Q = "Que necesito para realizar una baja temporal si ya paso el tiempo de 20 dias naturales"

# Respuesta con los añadidos que dio el bot (y el modelo en 6 de 8 corridas del 28-oct)
WITH_ADDITIONS = """Si ya pasaron los 20 días hábiles (no naturales), debes:
1. Manifestar tu determinación por escrito al Comité Académico, quien evaluará si procede.
2. El Comité Académico evalúa la solicitud y notifica a Servicios Escolares.
3. El resto del trámite sigue los mismos pasos que la baja dentro del plazo.
Aplica si cursaste al menos un semestre en el Instituto.
(Manual de Lineamientos TecNM, pág. 46)"""
CORRECT = """El plazo para la baja temporal es de 20 días hábiles (no naturales) a partir del \
inicio oficial de los cursos. Si ya pasó, debes manifestar tu determinación por escrito al \
Comité Académico.
Aplica a quien haya cursado al menos un semestre en el Instituto, salvo lo que indiquen los \
lineamientos de Servicio Social y Residencia Profesional.
(Manual de Lineamientos TecNM, pág. 46)"""


def pipe(kb, cfg, llm, hits=HITS):
    p = RagPipeline(cfg, kb.embedder, kb.store, llm)
    p.retrieve = lambda q, k=None, history=None: list(hits)  # los fragmentos reales de la pág. 46
    return p


def _facts():
    [case] = [c for c in load_cases(CASES) if c["q"] == Q]
    return case["hechos"]


# ---- hechos del caso en data/retrieval_cases.jsonl -------------------------------
def test_el_caso_real_tiene_hechos_que_debe_y_no_debe_incluir():
    facts = _facts()
    missing, added = check_facts(WITH_ADDITIONS, facts)
    assert added and not missing or "por escrito" in str(missing)
    assert any("servicios escolares" in p for p in added)  # el Comité notifica a SE
    assert any("mismos" in p for p in added)                 # "mismos pasos"
    assert check_facts(CORRECT, facts) == ([], [])


# ---- revisión local por afirmación ------------------------------------------------
def test_detecta_las_tres_afirmaciones_sin_respaldo():
    claims, total = grounding.unsupported(WITH_ADDITIONS, HITS, Q)
    reasons = [c.reason for c in claims]
    assert len(claims) == 3 and total == 5
    assert sum("Comité Académico" in r for r in reasons) == 2
    assert any("procedimientos distintos" in r for r in reasons)


def test_respuesta_correcta_no_tiene_afirmaciones_sin_respaldo():
    assert grounding.unsupported(CORRECT, HITS, Q) == ([], 3)


@pytest.mark.parametrize("sentence", [
    # lo que hace el estudiante, no el órgano: el verbo va antes del actor
    "Entrega tu escrito al Comité Académico.",
    # la acción sí está en el documento, atribuida a ese actor
    "Dentro del plazo, el Jefe(a) de la División de Estudios Profesionales notifica a "
    "Servicios Escolares si procede la baja.",
    "Lo manifiestas a la División de Estudios Profesionales, quien notifica al Departamento de "
    "Servicios Escolares.",
    # aclaración de otra unidad de plazo (regla 2)
    "El documento no menciona días naturales: el plazo es de 20 días hábiles.",
])
def test_oraciones_respaldadas_pasan(sentence):
    assert grounding.unsupported(sentence, HITS, Q)[0] == []


@pytest.mark.parametrize("sentence,motivo", [
    ("El Comité Académico autorizará tu baja.", "Comité Académico"),
    ("Después, el Comité Académico remite tu escrito a Servicios Escolares.", "Comité Académico"),
    ("Normalmente la respuesta tarda una semana en llegar al estudiante.", "lo habitual"),
    ("El documento no especifica qué formato usar para el escrito.", "no dice"),
    ("No se requieren documentos adicionales en el contexto.", "no dice"),
    ("Se sigue el mismo procedimiento que en la baja parcial.", "procedimientos distintos"),
])
def test_oraciones_sin_respaldo_se_detectan(sentence, motivo):
    [claim] = grounding.unsupported(sentence, HITS, Q)[0]
    assert motivo in claim.reason


def test_quitar_recorta_la_clausula_renumera_y_conserva_la_fuente():
    claims, _ = grounding.unsupported(WITH_ADDITIONS, HITS, Q)
    fixed = grounding.remove(WITH_ADDITIONS, claims)
    assert fixed == ("Si ya pasaron los 20 días hábiles (no naturales), debes:\n"
                     "1. Manifestar tu determinación por escrito al Comité Académico.\n"
                     "Aplica si cursaste al menos un semestre en el Instituto.\n"
                     "(Manual de Lineamientos TecNM, pág. 46)")


# ---- ciclo de intentos del pipeline ---------------------------------------------
def test_con_anadido_regenera_una_vez_pidiendo_quitarlo(kb, cfg):
    llm = ScriptedLLM(WITH_ADDITIONS, CORRECT)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and r.text == CORRECT and r.attempts == 2
    note = llm.calls[1][-1]["content"]
    assert note.startswith(CLAIMS_RETRY_NOTE.split("{items}")[0])
    assert "El Comité Académico evalúa la solicitud y notifica a Servicios Escolares." in note
    assert [c["accion"] for c in r.unsupported_claims] == ["regenerada"] * 3
    assert r.removed_claims == 0
    assert r.debug[0].startswith("[intento 1, rechazado: 3 afirmaciones sin respaldo]")
    assert check_facts(r.text, _facts()) == ([], [])


def test_respuesta_correcta_no_regenera(kb, cfg):
    llm = ScriptedLLM(CORRECT)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and r.text == CORRECT and len(llm.calls) == 1
    assert r.unsupported_claims == [] and r.debug == []


def test_si_insiste_se_quitan_las_oraciones_sin_respaldo(kb, cfg):
    llm = ScriptedLLM(WITH_ADDITIONS, WITH_ADDITIONS)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and len(llm.calls) == 2
    assert "evalua" not in r.text and "mismos pasos" not in r.text
    assert "Comité Académico" in r.text and r.text.endswith("(Manual de Lineamientos TecNM, pág. 46)")
    assert check_facts(r.text, _facts())[1] == []  # ya sin añadidos
    actions = [c["accion"] for c in r.unsupported_claims]
    assert actions[:3] == ["regenerada"] * 3
    assert sorted(actions[3:]) == ["quitada", "quitada", "recortada"] and r.removed_claims == 3
    # las dos respuestas originales quedan para diagnóstico
    assert len(r.debug) == 2 and "se quitaron 3" in r.debug[1]


def test_si_casi_todo_carece_de_respaldo_deriva(kb, cfg):
    # Palabras del documento (pasa la validación general), pero 3 de 4 pasos sin respaldo
    invented = ("1. Manifiesta tu determinación por escrito al Comité Académico.\n"
                "2. El Comité Académico evaluará la solicitud de baja temporal.\n"
                "3. El Comité Académico notificará al Departamento de Servicios Escolares si "
                "procede la baja.\n"
                "4. El resto del trámite sigue los mismos pasos.\n"
                "(Manual de Lineamientos TecNM, pág. 46)")
    llm = ScriptedLLM(invented)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.reason is Reason.VALIDATION_FAILED and len(llm.calls) == 2
    assert "afirmaciones sin respaldo" in r.detail
    assert [c["accion"] for c in r.unsupported_claims][-3:] == ["derivada"] * 3


def test_sin_tiempo_para_regenerar_quita_directo(kb, cfg):
    p = pipe(kb, cfg, ScriptedLLM(WITH_ADDITIONS, CORRECT))
    ticks = iter([0.0, cfg.llm_total_budget - 2.0])  # fecha límite; luego quedan 2 s
    p._clock = lambda: next(ticks)
    r = p.answer(Q)
    assert r.outcome is Outcome.ANSWERED and len(p.llm.calls) == 1 and r.removed_claims == 3


def test_proveedor_caido_al_regenerar_entrega_la_primera_sin_anadidos(kb, cfg):
    class AdditionThenDown:
        calls = 0

        def chat(self, messages, temperature=0.3, max_tokens=900, deadline=None):
            AdditionThenDown.calls += 1
            if AdditionThenDown.calls == 1:
                from app.ports import LLMReply
                return LLMReply(WITH_ADDITIONS, "fake", "fake-model")
            raise LLMUnavailable("groq: HTTP 429")

    r = pipe(kb, cfg, AdditionThenDown()).answer(Q)
    assert r.outcome is Outcome.ANSWERED and r.attempts == 1 and r.removed_claims == 3
    assert "mismos pasos" not in r.text


def test_regeneracion_que_falla_la_validacion_usa_la_primera_sin_anadidos(kb, cfg):
    invented_number = CORRECT.replace("20 días", "45 días")
    r = pipe(kb, cfg, ScriptedLLM(WITH_ADDITIONS, invented_number)).answer(Q)
    assert r.outcome is Outcome.ANSWERED and "45" not in r.text and r.removed_claims == 3


def test_tope_de_llamadas_se_respeta_con_no_info_y_afirmaciones(kb, cfg):
    # NO_INFO -> reintento -> añadido -> regeneración con añadido: 3 llamadas, no más
    llm = ScriptedLLM("NO_INFO: no dice días naturales", WITH_ADDITIONS, WITH_ADDITIONS,
                      CORRECT)
    r = pipe(kb, cfg, llm).answer(Q)
    assert len(llm.calls) == 3 and r.outcome is Outcome.ANSWERED and r.removed_claims == 3
    # y la regeneración comparte la misma fecha límite (RNF-FI-02)
    assert len(set(llm.deadlines)) == 1


def test_metrica_guarda_afirmaciones_quitadas(kb, cfg, tmp_path):
    from app.adapters.db_sqlite import SqliteRepository
    from app.service import ConversationService
    repo = SqliteRepository(tmp_path / "app.db")
    svc = ConversationService(cfg, repo, pipe(kb, cfg, ScriptedLLM(WITH_ADDITIONS, WITH_ADDITIONS)))
    svc.handle_message("tg-1", "Ana", Q)
    row = repo._conn.execute("SELECT afirmaciones_sin_respaldo, afirmaciones_quitadas, "
                             "respuestas_modelo, resultado FROM metricas").fetchone()
    claims = json.loads(row["afirmaciones_sin_respaldo"])
    assert len(claims) == 6 and {c["intento"] for c in claims} == {1, 2}
    assert claims[0]["oracion"].startswith("Manifestar tu determinación")
    assert (row["afirmaciones_quitadas"], row["resultado"]) == (3, "answered")
    assert len(json.loads(row["respuestas_modelo"])) == 2


# ---- instrucciones del modelo -----------------------------------------------------
def test_instrucciones_piden_no_agregar_lo_que_el_documento_no_dice():
    for phrase in ("Atribuye cada acción solo a la persona u órgano",
                   "no mezcles sus pasos", "No completes con deducciones",
                   "tampoco lo menciones", "condiciones y excepciones"):
        assert phrase in SYSTEM_PROMPT
    # las reglas que ya existían siguen ahí
    for phrase in ("TODOS los elementos", "NO_INFO seguido de dos puntos",
                   "eso NO es falta de información", "Tampoco transcribas", "fuente"):
        assert phrase in SYSTEM_PROMPT
