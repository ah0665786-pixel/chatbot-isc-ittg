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


@pytest.mark.parametrize("sentence,motivo", [
    # 28-oct: "¿Puedo pedir la baja temporal después de los 20 días hábiles?"
    ("Pasado ese plazo, el estudiante ya no tiene derecho a solicitarla.", "niega"),
    ("No, ya no es posible solicitar la baja temporal.", "niega"),
    # "que" también introduce lo que hace el órgano
    ("Espera la respuesta del Comité Académico, que decidirá si procede la baja.",
     "Comité Académico"),
])
def test_negaciones_y_relativo_que(sentence, motivo):
    [claim] = grounding.unsupported(sentence, HITS, Q)[0]
    assert motivo in claim.reason


def test_negacion_que_si_dice_el_documento_pasa(kb, cfg):
    # El Reglamento de prueba dice "no podrá exceder de dos semestres consecutivos"
    p = RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM("x"))
    hits = p.retrieve("¿Cuánto dura la baja temporal?")
    sentence = "La baja temporal no podrá exceder de dos semestres consecutivos."
    assert grounding.unsupported(sentence, hits, "¿Cuánto dura la baja temporal?")[0] == []


def test_respuesta_que_niega_el_tramite_deriva(kb, cfg):
    denial = ("No, ya no es posible solicitar la baja temporal después de los 20 días hábiles. "
              "Pasado ese plazo, el estudiante ya no tiene derecho a solicitarla. "
              "(Manual de Lineamientos TecNM, pág. 46)")
    r = pipe(kb, cfg, ScriptedLLM(denial)).answer(
        "¿Puedo pedir la baja temporal después de los 20 días hábiles?")
    assert r.reason is Reason.VALIDATION_FAILED and "niega" in r.detail
    assert check_facts(denial, _facts())[1]  # también lo cuenta la evaluación


# ---- ajustes tras el DESPUÉS del 28-oct (3 de 10 derivaron por "【1】") ----------
CITED = ("Necesitas manifestar por escrito tu determinación de baja al Comité Académico cuando "
         "la solicitud no se haya realizado dentro de los 20 días hábiles{marca}. "
         "(Manual de Lineamientos del TecNM, pág. 46)")


@pytest.mark.parametrize("marca", ["【1】", "【1†L1-L4】", " 【3†L10-L12】"])
def test_marcas_de_cita_del_modelo_no_son_cifras_ni_llegan_al_estudiante(kb, cfg, marca):
    llm = ScriptedLLM(CITED.format(marca=marca))
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and len(llm.calls) == 1
    assert "【" not in r.text and r.text == CITED.format(marca="")
    ok, why = pipe(kb, cfg, llm).validate(CITED.format(marca=marca), HITS, Q)
    assert ok, why


def test_indice_suelto_no_llega_al_estudiante(kb, cfg):
    answer = CITED.format(marca=" [1]")
    r = pipe(kb, cfg, ScriptedLLM(answer)).answer(Q)
    assert r.outcome is Outcome.ANSWERED and "[1]" not in r.text
    assert r.text == CITED.format(marca="")


def test_volcado_del_contexto_se_sigue_rechazando_aunque_se_quiten_indices(kb, cfg):
    dump = ("[1] (Manual_de_Lineamientos_TecNM.pdf, pág. 46) Si la solicitud no se realizó en "
            "los 20 días hábiles, el estudiante debe manifestar su determinación por escrito al "
            "Comité Académico.")
    r = pipe(kb, cfg, ScriptedLLM(dump)).answer(Q)
    assert r.reason is Reason.VALIDATION_FAILED and "formato interno" in r.detail


@pytest.mark.parametrize("sentence", [
    "2. No se menciona otro requisito adicional;",       # corrida #3 del 28-oct
    "No se requieren documentos adicionales.",
    "No se especifica ningún formato para el escrito.",
])
def test_no_se_menciona_otro_requisito_se_detecta(sentence):
    [claim] = grounding.unsupported(sentence, HITS, Q)[0]
    assert "no dice" in claim.reason


def test_no_se_menciona_con_aclaracion_de_unidad_pasa():
    sentence = "No se mencionan días naturales: el plazo es de 20 días hábiles."
    assert grounding.unsupported(sentence, HITS, Q)[0] == []


def test_instrucciones_piden_a_quien_aplica_el_tramite():
    assert "a quién aplica el trámite" in SYSTEM_PROMPT
    assert "aunque estén en otra oración" in SYSTEM_PROMPT


def test_al_quitar_el_inicio_de_un_renglon_queda_con_mayuscula():
    answer = ("1. Manifiesta tu determinación por escrito al Comité Académico.\n"
              "2. No se menciona otro requisito adicional; el procedimiento ante la División "
              "de Estudios Profesionales solo aplica dentro de los 20 días hábiles.\n"
              "(Manual de Lineamientos TecNM, pág. 46)")
    claims, _ = grounding.unsupported(answer, HITS, Q)
    assert grounding.remove(answer, claims).splitlines()[1] == (
        "2. El procedimiento ante la División de Estudios Profesionales solo aplica dentro de "
        "los 20 días hábiles.")


# ---- ajustes tras el DESPUÉS con 332fdf5 (28-oct) --------------------------------
# Primer intento de la corrida #4: el paso 2 atribuye al Comité una decisión
DRAFT_4 = """Para solicitar la **baja temporal** cuando ya han transcurrido los 20 días hábiles \
desde el inicio oficial de los cursos, el estudiante debe:

1. **Manifestar por escrito su determinación** de baja al **Comité Académico**.  
2. El Comité Académico es quien recibirá la solicitud y, de acuerdo con lo establecido, \
decidirá si procede la baja.

No se menciona ningún otro documento o requisito adicional para este caso. \
(Manual de Lineamientos del TecNM, pág. 46)"""


@pytest.mark.parametrize("sentence", [
    "El Comité Académico es quien recibirá la solicitud y decidirá si procede la baja.",
    "El Comité Académico es quien recibirá la solicitud y, de acuerdo con lo establecido, "
    "decidirá si procede la baja.",
    "El Comité Académico será el encargado de autorizar la baja.",
])
def test_es_quien_atribuye_la_accion_al_actor(sentence):
    [claim] = grounding.unsupported(sentence, HITS, Q)[0]
    assert "Comité Académico" in claim.reason


def test_es_quien_con_accion_del_documento_pasa():
    sentence = ("El Jefe(a) de la División de Estudios Profesionales es quien notifica al "
                "Departamento de Servicios Escolares si procede la baja.")
    assert grounding.unsupported(sentence, HITS, Q)[0] == []


def test_corrida_4_se_corrige_sin_la_decision_del_comite(kb, cfg):
    r = pipe(kb, cfg, ScriptedLLM(DRAFT_4, DRAFT_4)).answer(Q)
    claims = [c for c in r.unsupported_claims if c["accion"] != "condicion_pedida"]
    assert r.outcome is Outcome.ANSWERED and len(claims) == 4  # 2 + 2
    # además se pidió "al menos un semestre" en la misma regeneración
    assert [c["accion"] for c in r.unsupported_claims].count("condicion_pedida") == 1
    assert "decidirá" not in r.text and "No se menciona" not in r.text
    assert "1. **Manifestar por escrito su determinación** de baja al **Comité Académico**." in r.text


@pytest.mark.parametrize("sentence", ["No se exige ningún documento adicional.",
                                      "No se solicitan otros requisitos."])
def test_no_se_exige_se_detecta(sentence):
    [claim] = grounding.unsupported(sentence, HITS, Q)[0]
    assert "no dice" in claim.reason


def test_instrucciones_piden_no_hablar_de_lo_que_el_documento_no_menciona():
    assert "no menciona, no requiere o no exige" in SYSTEM_PROMPT
    assert "di solo lo que sí dice" in SYSTEM_PROMPT


# Corrida #7: correcta, pero sin nombrar el documento al final
NO_SOURCE = """Para solicitar la **baja temporal** cuando ya han transcurrido los 20 días hábiles \
desde el inicio oficial de los cursos, el estudiante debe:

1. **Manifestar por escrito su determinación** al **Comité Académico**.  
2. Cumplir con el requisito de haber **cursado al menos un semestre** en el Instituto Tecnológico.  

(Se mantiene la excepción de los lineamientos vigentes sobre Servicio Social y Residencia \
Profesional)."""


def test_sin_fuente_se_agrega_la_del_fragmento_usado(kb, cfg):
    llm = ScriptedLLM(NO_SOURCE)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and len(llm.calls) == 1  # sin otra llamada
    assert r.text == NO_SOURCE + "\n\n(Fuente: Manual de Lineamientos TecNM, pág. 46)"


@pytest.mark.parametrize("cita", ["(Manual de Lineamientos TecNM, pág. 46)",
                                  "(Fuente: Manual_de_Lineamientos_TecNM.pdf)",
                                  "(Manual de Lineamientos del TecNM)"])
def test_con_fuente_no_se_agrega_otra(kb, cfg, cita):
    answer = CORRECT.replace("(Manual de Lineamientos TecNM, pág. 46)", cita)
    r = pipe(kb, cfg, ScriptedLLM(answer)).answer(Q)
    assert r.text == answer


def test_aclaracion_de_unidad_con_palabras_propias_pasa():
    # Borrador del 28-oct marcado de más por cobertura (40 %)
    sentence = "**Aclaración:** el documento menciona “días hábiles”; la pregunta habla de días naturales."
    assert grounding.unsupported(sentence, HITS, Q)[0] == []


# ---- fuente agregada cuando hay fragmentos de dos documentos ----------------------
REGLAMENTO_HIT = Hit(Chunk("reglamento-estudiantes", "Reglamento de Estudiantes TecNM.pdf", 7,
                           "XIII. Baja temporal. Es el trámite que realiza el/la estudiante en el "
                           "nivel educativo en el que está inscrito/a, autorizado por el TecNM, "
                           "para suspender durante un periodo lectivo sus estudios.", "pág. 5"),
                     0.866)
TWO_DOCS = [HITS[0], REGLAMENTO_HIT, HITS[1]]  # Manual, Reglamento, Manual


@pytest.mark.parametrize("answer,expected", [
    # usa el Reglamento (definición de baja temporal) aunque el Manual quedó primero
    ("La baja temporal es el trámite que realiza el estudiante en el nivel educativo en el que "
     "está inscrito, autorizado por el TecNM, para suspender durante un periodo lectivo sus "
     "estudios.", "(Fuente: Reglamento de Estudiantes TecNM, pág. 5)"),
    # usa el Manual (fuera de plazo, al Comité Académico)
    ("Si la solicitud no se realizó en los 20 días hábiles, el estudiante debe manifestar su "
     "determinación por escrito al Comité Académico.",
     "(Fuente: Manual de Lineamientos TecNM, pág. 46)"),
])
def test_fuente_agregada_cita_el_documento_usado(kb, cfg, answer, expected):
    r = pipe(kb, cfg, ScriptedLLM(answer), hits=TWO_DOCS).answer(Q)
    assert r.outcome is Outcome.ANSWERED
    assert r.text == f"{answer}\n\n{expected}"


def test_limite_que_el_documento_si_pone_no_es_negacion():
    # 29-oct: "¿Cuántos créditos de actividades complementarias necesito?" derivó de más.
    # El lineamiento dice "no deben de tener más de 2 créditos".
    hit = Hit(Chunk("acom", "Lineamiento-para-Acreditacion-de-Actividades-Complementarias.pdf", 2,
                    "Para cada una de las actividades complementarias autorizadas por el plantel, "
                    "no deben de tener más de 2 créditos asignados. El cumplimiento de los 5 "
                    "créditos debe realizarse dentro de los seis primeros semestres.", "pág. 2"),
              0.85)
    sentence = ("Además, cada actividad complementaria autorizada no puede tener más de 2 "
                "créditos.")
    assert grounding.unsupported(sentence, [hit], "¿Cuántos créditos necesito?")[0] == []


# 30-oct, "¿Cómo solicito una baja temporal extemporánea?": el paréntesis no era la fuente
EXTEMPORANEA = """Para una baja temporal que se solicite fuera del plazo de 20 días hábiles, el \
estudiante debe:

1. Manifestar por escrito su determinación de baja al **Comité Académico**.

(El resto del procedimiento no se menciona para este caso.)  
(Fuente: Manual_de_Lineamientos_TecNM, pág. 46)"""


def test_parentesis_que_no_es_la_fuente_si_se_revisa():
    [claim] = grounding.unsupported(EXTEMPORANEA, HITS, Q)[0]
    assert "resto del procedimiento" in claim.text
    fixed = grounding.remove(EXTEMPORANEA, [claim])
    assert "resto" not in fixed and fixed.endswith("(Fuente: Manual_de_Lineamientos_TecNM, pág. 46)")


@pytest.mark.parametrize("cita", [
    "(Manual de Lineamientos TecNM, pág. 46)", "*(Fuente: Manual de Lineamientos del TecNM, pág. 46)*",
    "(Manual de Lineamientos del TecNM)", "(Lineamiento-para-Acreditacion.pdf)",
    "(Manual de Lineamientos Académico-Administrativos del TecNM, capítulo 14)"])
def test_las_citas_de_fuente_no_se_revisan(cita):
    assert grounding.unsupported(cita, HITS, Q) == ([], 0)


# ---- condiciones y excepciones omitidas (regla 1d), 30-oct ----------------------
from app.rag.pipeline import CONDITIONS_NOTE  # noqa: E402

GUIA_3 = ("Generalidades. • La Residencia Profesional se cursa por una única ocasión. En el caso "
          "de existir causas justificadas fuera del alcance del residente, es necesario proponer "
          "una segunda asignación de proyecto. • El horario diario de actividades del residente "
          "en el desarrollo de su proyecto lo establece la empresa, organismo o dependencia.")
GUIA_CREDITOS = ("La Residencia Profesional tiene un valor de 10 créditos y debe acumular 500 "
                 "horas en un periodo de cuatro a seis meses.")
RES_HITS = [Hit(Chunk("guia-residencia", "Guia_Residencia_Profesional_2022.pdf", 3, GUIA_3,
                      "pág. 4"), 0.944),
            Hit(Chunk("guia-residencia", "Guia_Residencia_Profesional_2022.pdf", 4, GUIA_CREDITOS,
                      "pág. 5"), 0.90)]
RES_Q = "¿Puedo repetir la residencia si la repruebo?"
# Lo que respondió el modelo 3 de 3 veces con 1e42e60
RES_SIN = ("No, la Residencia Profesional solo se cursa una única ocasión; si se reprueba no "
           "puede repetirse. (Guía Residencia Profesional 2022, pág. 4)")
RES_CON = ("No, la Residencia Profesional se cursa por una única ocasión. Solo si existen causas "
           "justificadas fuera de tu alcance se propone una segunda asignación de proyecto. "
           "(Guía Residencia Profesional 2022, pág. 4)")
BAJA_SIN = ("Si ya pasaron los 20 días hábiles, para la baja temporal debes manifestar por "
            "escrito tu determinación al Comité Académico. (Manual de Lineamientos TecNM, pág. 46)")
BAJA_CON = ("Si ya pasaron los 20 días hábiles, para la baja temporal debes manifestar por "
            "escrito tu determinación al Comité Académico. Aplica si cursaste al menos un "
            "semestre en el Instituto. (Manual de Lineamientos TecNM, pág. 46)")


def test_residencia_omite_la_segunda_asignacion():
    [cond] = grounding.omitted_conditions(RES_SIN, RES_HITS, RES_Q)
    assert "segunda asignación de proyecto" in cond and "horario" not in cond
    assert grounding.omitted_conditions(RES_CON, RES_HITS, RES_Q) == []


def test_baja_omite_al_menos_un_semestre():
    [cond] = grounding.omitted_conditions(BAJA_SIN, HITS, Q)
    assert "al menos un semestre" in cond
    assert grounding.omitted_conditions(BAJA_CON, HITS, Q) == []


def test_condicion_de_otro_tramite_o_parrafo_no_se_pide():
    # créditos de la residencia: la excepción de la segunda asignación está en otro párrafo
    creditos = "La Residencia Profesional vale 10 créditos. (Guía Residencia Profesional 2022, pág. 5)"
    assert grounding.omitted_conditions(creditos, RES_HITS,
                                        "¿Cuántos créditos vale la residencia profesional?") == []
    # baja parcial: "Todo estudiante que ... tiene derecho a solicitar baja parcial"
    parcial = Hit(Chunk("manual", DOC, 82, "De la baja parcial: 5.4.6.1 Todo estudiante que haya "
                        "cursado al menos un semestre en el Instituto, tiene derecho a solicitar "
                        "baja parcial en algunas asignaturas.", "pág. 45"), 0.85)
    assert grounding.omitted_conditions(BAJA_CON.replace(" Aplica si cursaste al menos un "
                                                         "semestre en el Instituto.", ""),
                                        [HITS[0], parcial], Q) == []


def test_remision_a_otro_lineamiento_no_se_pide():
    # "Lo anterior a excepción de lo que se indique en los lineamientos ... (ver CAPÍTULO 11)"
    assert all("lo que se indique" not in c
               for c in grounding.omitted_conditions(BAJA_CON, HITS, Q))


def test_residencia_pide_la_excepcion_en_la_misma_regeneracion(kb, cfg):
    llm = ScriptedLLM(RES_SIN, RES_CON)
    r = pipe(kb, cfg, llm, hits=RES_HITS).answer(RES_Q)
    assert r.outcome is Outcome.ANSWERED and r.text == RES_CON and len(llm.calls) == 2
    note = llm.calls[1][-1]["content"]
    assert note.startswith(CONDITIONS_NOTE.split("{items}")[0]) and "segunda asignación" in note
    assert [c["accion"] for c in r.unsupported_claims] == ["condicion_pedida"]
    assert r.removed_claims == 0 and "1 condiciones omitidas" in r.debug[0]


def test_baja_pide_al_menos_un_semestre_en_la_misma_regeneracion(kb, cfg):
    llm = ScriptedLLM(BAJA_SIN, BAJA_CON)
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.text == BAJA_CON and len(llm.calls) == 2
    assert "al menos un semestre" in llm.calls[1][-1]["content"]


def test_si_la_regeneracion_tampoco_la_incluye_se_entrega_sin_otra_llamada(kb, cfg):
    llm = ScriptedLLM(RES_SIN, RES_SIN)
    r = pipe(kb, cfg, llm, hits=RES_HITS).answer(RES_Q)
    assert r.outcome is Outcome.ANSWERED and r.text == RES_SIN and len(llm.calls) == 2


def test_condicion_y_afirmacion_sin_respaldo_van_en_una_sola_nota(kb, cfg):
    con_anadido = RES_SIN.replace(
        " (Guía", " La División de Estudios Profesionales autorizará la segunda asignación de "
                  "proyecto del residente. (Guía")
    llm = ScriptedLLM(con_anadido, RES_CON)
    r = pipe(kb, cfg, llm, hits=RES_HITS).answer(RES_Q)
    assert r.text == RES_CON and len(llm.calls) == 2
    note = llm.calls[1][-1]["content"]
    assert "no están respaldadas" in note and "segunda asignación" in note
    assert sorted(c["accion"] for c in r.unsupported_claims) == ["condicion_pedida", "regenerada"]


def test_sin_tiempo_no_se_pide_la_condicion(kb, cfg):
    p = pipe(kb, cfg, ScriptedLLM(RES_SIN, RES_CON), hits=RES_HITS)
    ticks = iter([0.0, cfg.llm_total_budget - 2.0])  # fecha límite; luego quedan 2 s
    p._clock = lambda: next(ticks)
    r = p.answer(RES_Q)
    assert r.text == RES_SIN and len(p.llm.calls) == 1


def test_despues_de_un_reintento_por_validacion_no_se_pide_la_condicion(kb, cfg):
    # el cupo de regeneración ya se usó: no hay una tercera llamada por la condición
    invented = RES_SIN.replace("2022, pág. 4", "2022, pág. 4. Son 45 días")
    llm = ScriptedLLM(invented, RES_SIN, RES_CON)
    r = pipe(kb, cfg, llm, hits=RES_HITS).answer(RES_Q)
    assert r.text == RES_SIN and len(llm.calls) == 2


def test_condicion_de_baja_se_pide_sin_el_plazo_de_20_dias():
    # 31-oct: con la oración completa el modelo escribía "normalmente deben hacerlo
    # dentro de los 20 días hábiles" en la respuesta de la baja FUERA de plazo
    [cond] = grounding.omitted_conditions(BAJA_SIN, HITS, Q)
    assert cond == ("Todo estudiante que haya cursado al menos un semestre en el Instituto, "
                    "tiene derecho a solicitar baja temporal")


def test_deduccion_despues_de_una_coma_se_recorta_sin_perder_la_condicion():
    answer = ("1. Manifiesta por escrito tu determinación de baja al Comité Académico.\n"
              "*Condición:* Sólo los estudiantes que hayan cursado al menos un semestre en el "
              "Instituto pueden solicitar la baja temporal, y normalmente deben hacerlo dentro de "
              "los 20 días hábiles a partir del inicio oficial de los cursos.\n"
              "(Fuente: Manual de Lineamientos del TecNM, pág. 46)")
    claims, _ = grounding.unsupported(answer, HITS, Q)
    [claim] = claims
    assert "normalmente" in claim.reason and claim.cut_at > 0
    fixed = grounding.remove(answer, claims)
    assert ("*Condición:* Sólo los estudiantes que hayan cursado al menos un semestre en el "
            "Instituto pueden solicitar la baja temporal.") in fixed
    assert "normalmente" not in fixed


# ---- fila 41 de la revisión del 29-oct: fuga interna sin borrar la oración --------
MANUAL_97 = ("La convalidación está caracterizada por los siguientes aspectos:  Permite al "
             "estudiante cambiar de un plan de estudio a otro dentro de las Instituciones "
             "adscritas al TecNM.  Permite cursar una segunda carrera a nivel licenciatura, una "
             "vez que el egresado se ha titulado o ha aprobado su acto profesional de la primera "
             "carrera cursada.  Permite al estudiante, que causó baja definitiva habiendo acr "
             "editado el 50% de créditos o más, reinscribirse en un plan de estudios diferente "
             "que le ofrezca el Instituto, con el propósito de que concluya una carrera "
             "profesional.")
FILA41_HITS = [Hit(Chunk("manual-de-lineamientos-tecnm", DOC, 97, MANUAL_97, "pág. 53"), 0.87)]
FILA41_Q = ("Me dieron de baja definitiva, ¿puedo entrar a otra carrera y que me cuenten lo que "
            "llevaba?")
# Segundo intento real del modelo (con sus espacios finos \u202f)
FILA41 = ("Sí, puedes reinscribirte en un plan de estudios diferente.\u202fSegún el CONTEXTO, la "
          "convalidación permite al estudiante que causó baja definitiva **habiendo acreditado el "
          "50\u202f% de créditos o más** reinscribirse en un plan de estudios distinto que le "
          "ofrezca el Instituto, con el propósito de que concluya una carrera profesional. \n\n"
          "(NO_INFO: el CONTEXTO no especifica si se reconocen exactamente las asignaturas o cómo "
          "se contabilizan los créditos).\n\n(Fuente: Manual de Lineamientos TecNM, pág. 53)")


def test_fuga_interna_se_limpia_sin_borrar_la_condicion_del_50():
    text, leaks = grounding.scrub_internal(FILA41, FILA41_HITS)
    assert "Según el CONTEXTO" in leaks and any("NO_INFO" in x for x in leaks)
    assert "CONTEXTO" not in text and "NO_INFO" not in text
    assert ("La convalidación permite al estudiante que causó baja definitiva **habiendo "
            "acreditado el 50\u202f% de créditos o más**") in text
    assert grounding.unsupported(text, FILA41_HITS, FILA41_Q)[0] == []  # lo demás tiene respaldo


def test_fila_41_se_entrega_con_el_50_por_ciento(kb, cfg):
    llm = ScriptedLLM(FILA41)
    r = pipe(kb, cfg, llm, hits=FILA41_HITS).answer(FILA41_Q)
    assert r.outcome is Outcome.ANSWERED and len(llm.calls) == 1
    assert "50\u202f% de créditos" in r.text and "CONTEXTO" not in r.text
    assert r.text.endswith("(Fuente: Manual de Lineamientos TecNM, pág. 53)")
    assert {c["accion"] for c in r.unsupported_claims} == {"limpiada"}
    assert r.removed_claims == 0 and "fugas internas" in r.debug[0]


@pytest.mark.parametrize("answer,expected", [
    ("Requisitos para la Residencia Profesional según el contexto:\n- Servicio social liberado.",
     "Requisitos para la Residencia Profesional:\n- Servicio social liberado."),
    ("De acuerdo con los fragmentos, la baja se manifiesta por escrito.",
     "La baja se manifiesta por escrito."),
    ("1. Según el CONTEXTO, debes manifestarlo por escrito【1†L1-L4】.",
     "1. Debes manifestarlo por escrito."),
])
def test_limpieza_de_frases_internas(answer, expected):
    assert grounding.scrub_internal(answer, HITS)[0] == expected


def test_si_lo_que_queda_no_tiene_respaldo_se_borra_la_oracion():
    # tras limpiar "el CONTEXTO" queda "el documento no especifica...", que no se entrega
    text, _ = grounding.scrub_internal("El CONTEXTO no especifica el formato del escrito.", HITS)
    [claim] = grounding.unsupported(text, HITS, Q)[0]
    assert "no dice" in claim.reason


def test_contexto_que_usa_el_documento_no_se_limpia():
    hit = Hit(Chunk("x", "Lineamiento.pdf", 1, "El tutor analiza el contexto social del estudiante.",
                    "pág. 2"), 0.85)
    answer = "El tutor analiza el contexto social del estudiante."
    assert grounding.scrub_internal(answer, [hit]) == (answer, [])
