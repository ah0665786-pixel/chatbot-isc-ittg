"""Fuente que cita los fragmentos por su índice (5-nov, Telegram).

Con el Extracto (un .docx sin página) entre los fragmentos, el modelo terminó con
"(Fuente: Manual ..., extractos [1] y [2])": la limpieza de fugas quitó los [n] y al
estudiante le llegó "extractos y". Otra vez escribió "extractos 1 y 2". La fuente se
rehace con el documento y la página de los fragmentos que nombró.
"""
import pytest

from app.domain import Chunk, Hit, Outcome
from app.rag.pipeline import RagPipeline, _source_by_index
from tests.conftest import ScriptedLLM

EXTRACTO = Hit(Chunk("extracto-manual-lineamientos-servicio-social",
                     "Extracto_Manual_Lineamientos_Servicio_Social.docx", 0,
                     "Extracto verificado del Manual de Lineamientos Académico-Administrativos del "
                     "TecNM: servicio social. Créditos para iniciar el servicio social: el "
                     "estudiante debe tener aprobado el 70% de los créditos de su plan de "
                     "estudios. El requisito para iniciar el servicio social se expresa en "
                     "porcentaje de créditos aprobados, no en un semestre determinado.", ""), 0.913)
MANUAL_78 = Hit(Chunk("manual-de-lineamientos-tecnm", "Manual_de_Lineamientos_TecNM.pdf", 142,
                      "11.4.1.7 Para la realización del Servicio Social en un ámbito de influencia "
                      "diferente al del Instituto, los prestadores deben cubrir los siguientes "
                      "requisitos: Presentar la solicitud de autorización para realizar el "
                      "Servicio Social en otras instituciones de Educación Superior (ver Anexo "
                      "XVIII).", "pág. 78"), 0.911)
MANUAL_89 = Hit(Chunk("manual-de-lineamientos-tecnm", "Manual_de_Lineamientos_TecNM.pdf", 162,
                      "12.4.4.4 Es responsable de elaborar el expediente por cada proyecto "
                      "propuesto por la Academia y avalado por el Jefe de Departamento Académico.",
                      "pág. 89"), 0.903)
HITS = [EXTRACTO, MANUAL_78, MANUAL_89]
Q = "que requisitos necesito para hacer el servicio social?"

BODY = ("- Tener aprobado al menos el 70 % de los créditos de su plan de estudios (el requisito "
        "se expresa en porcentaje de créditos aprobados, no en un semestre determinado).\n"
        "- Si desea realizar el Servicio Social en otras instituciones de Educación Superior, "
        "presentar la solicitud de autorización para realizar el Servicio Social (ver Anexo "
        "XVIII).")
MANUAL_NAME = "Manual de Lineamientos Académico‑Administrativos del TecNM"
EXPECTED = ("(Fuente: Extracto Manual Lineamientos Servicio Social; "
            "Manual de Lineamientos TecNM, pág. 78)")


def pipe(kb, cfg, llm, hits=HITS):
    p = RagPipeline(cfg, kb.embedder, kb.store, llm)
    p.retrieve = lambda q, k=None, history=None: list(hits)
    return p


@pytest.mark.parametrize("cita", [
    "extractos [1] y [2]",   # dd72c137 / 0749b8a0: llegó "extractos y"
    "extractos 1 y 2",       # 2a11440d / 511d03f8: llegó tal cual
    "extractos 【1】 y 【2】",
    "fragmentos [1], [2]",
])
def test_fuente_por_indice_se_rehace_con_documento_y_pagina(kb, cfg, cita):
    llm = ScriptedLLM(f"{BODY}\n\n(Fuente: {MANUAL_NAME}, {cita}).")
    r = pipe(kb, cfg, llm).answer(Q)
    assert r.outcome is Outcome.ANSWERED and len(llm.calls) == 1
    assert r.text == f"{BODY}\n\n{EXPECTED}"
    assert "extractos" not in r.text and "fragmentos" not in r.text
    assert not r.unsupported_claims  # los [n] de la fuente ya no cuentan como fugas


def test_fuente_por_indice_con_dos_documentos(kb, cfg):
    reglamento = Hit(Chunk("reglamento-estudiantes", "Reglamento de Estudiantes TecNM.pdf", 7,
                           "Los estudiantes deben presentar la solicitud de autorización ante la "
                           "Coordinación.", "pág. 5"), 0.90)
    hits = [MANUAL_78, reglamento, MANUAL_89]
    body = ("Debes presentar la solicitud de autorización para realizar el Servicio Social en "
            "otras instituciones de Educación Superior (ver Anexo XVIII).")
    llm = ScriptedLLM(f"{body}\n\n(Fuente: Manual de Lineamientos y Reglamento, fragmentos 2 y 1)")
    r = pipe(kb, cfg, llm, hits).answer(Q)
    assert r.outcome is Outcome.ANSWERED
    assert r.text == (f"{body}\n\n(Fuente: Reglamento de Estudiantes TecNM, pág. 5; "
                      "Manual de Lineamientos TecNM, pág. 78)")


def test_extractos_sin_numero_cita_el_fragmento_que_mas_se_parece(kb, cfg):
    body = ("Para realizar el Servicio Social en otras instituciones de Educación Superior debes "
            "presentar la solicitud de autorización (ver Anexo XVIII).")
    llm = ScriptedLLM(f"{body}\n\n(Fuente: {MANUAL_NAME}, extractos)")
    r = pipe(kb, cfg, llm, [MANUAL_89, MANUAL_78]).answer(Q)
    assert r.outcome is Outcome.ANSWERED
    assert r.text == f"{body}\n\n(Fuente: Manual de Lineamientos TecNM, pág. 78)"


def test_varias_paginas_del_mismo_documento():
    text = _source_by_index("Texto.\n\n(Fuente: Manual, extractos [2] y [3])", HITS)
    assert text == "Texto.\n\n(Fuente: Manual de Lineamientos TecNM, págs. 78 y 89)"


@pytest.mark.parametrize("fuente", [
    "(Fuente: Manual de Lineamientos TecNM, pág. 78)",
    "(Fuente: Manual de Lineamientos TecNM, págs. 2 y 3)",
    "(Manual de Lineamientos TecNM, numeral 11.4.1.7)",
    "(Manual de Lineamientos Académico-Administrativos del TecNM, capítulo 1)",
    "(Fuente: Extracto Manual Lineamientos Servicio Social)",
    "(El resto del procedimiento no se menciona para este caso.)",
])
def test_fuente_que_ya_nombra_documento_no_se_toca(fuente):
    text = f"Texto de la respuesta.\n\n{fuente}"
    assert _source_by_index(text, HITS) == text
