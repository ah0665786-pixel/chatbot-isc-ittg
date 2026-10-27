"""Mensajes de orientación al estudiante (RF-C-04, HU-08): cada motivo nombra
la instancia correcta y el correo de la coordinación solo aparece cuando la
consulta le corresponde a la coordinación."""
from dataclasses import replace

import pytest

from app.config import Settings
from app.domain import Reason
from app.notify import notice_phrase
from app.rag.messages import derivation_message
from app.rag.pipeline import RagPipeline
from tests.conftest import ScriptedLLM

CFG = Settings(contact_email="csistemas@tuxtla.tecnm.mx",
               contact_hours="lunes a viernes de 8:00 a 15:00 horas")


@pytest.mark.parametrize("reason,category", [
    (Reason.OUT_OF_SCOPE, "datos_personales"),
    (Reason.OUT_OF_SCOPE, "documento_oficial"),
])
def test_datos_personales_y_tramites_oficiales_van_a_servicios_escolares_sin_correo(
        reason, category):
    text = derivation_message(reason, CFG, category)
    assert "Servicios Escolares (Control Escolar) del ITTG" in text
    assert CFG.contact_email not in text and "@" not in text
    assert CFG.contact_hours not in text      # es el horario de la coordinación
    assert "Coordinación" not in text


@pytest.mark.parametrize("reason", [Reason.LOW_SIMILARITY, Reason.NOT_IN_CONTEXT,
                                    Reason.VALIDATION_FAILED, Reason.LLM_UNAVAILABLE])
def test_dudas_academicas_y_falla_tecnica_van_a_la_coordinacion_con_correo_y_horario(reason):
    text = derivation_message(reason, CFG)
    assert "Coordinación de ISC" in text
    assert CFG.contact_email in text and CFG.contact_hours in text
    assert "Servicios Escolares" not in text


def test_sin_contact_email_no_queda_un_correo_vacio():
    text = derivation_message(Reason.NOT_IN_CONTEXT, replace(CFG, contact_email=""))
    assert "escribe a" not in text and CFG.contact_hours in text


@pytest.mark.parametrize("reason", [Reason.INJECTION, Reason.TOO_LONG])
def test_inyeccion_y_mensaje_largo_no_dan_contacto(reason):
    text = derivation_message(reason, CFG)
    assert "@" not in text and "Coordinación de ISC" not in text
    assert "Servicios Escolares" not in text


@pytest.mark.parametrize("q", ["Quiero ver mi kardex", "Genérame mi constancia de estudios"])
def test_pipeline_no_da_el_correo_de_la_coordinacion_en_datos_personales(kb, cfg, q):
    r = RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM("x")).answer(q)
    assert r.reason is Reason.OUT_OF_SCOPE
    assert "Servicios Escolares" in r.text and cfg.contact_email not in r.text


def test_la_frase_de_aviso_no_contradice_la_instancia():
    # Se agrega debajo del mensaje: remite a "la indicación de arriba", que en
    # datos personales es Servicios Escolares y en lo demás la coordinación
    for immediate in (True, False):
        assert "sigue la indicación de arriba" in notice_phrase(immediate)


def test_datos_personales_con_aviso_a_la_coordinacion(kb, cfg):
    """En modo coordinadora se mantiene la frase del aviso, sin dar el correo."""
    from app.adapters.db_sqlite import SqliteRepository
    from app.service import ConversationService
    cfg = replace(cfg, avisos_modo="coordinadora", coordinadora_email="c@test.local",
                  smtp_host="127.0.0.1", smtp_user="")
    svc = ConversationService(cfg, SqliteRepository(cfg.db_path),
                              RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM("x")))
    result, _ = svc.handle_message("tg-1", "Ana", "Quiero ver mi kardex")
    svc.notifier.close()
    assert "Servicios Escolares" in result.text
    assert "resumen diario que recibe la Coordinación de ISC" in result.text
    assert cfg.contact_email not in result.text
