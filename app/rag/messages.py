"""Mensajes al estudiante cuando se deriva a atención humana (RF-C-04, HU-08).

Cada motivo nombra la instancia que de verdad le corresponde. El correo y el
horario de la coordinación (CONTACT_EMAIL, CONTACT_HOURS) solo se dan cuando
la consulta es de la coordinación: los datos personales y los documentos
oficiales los atiende Servicios Escolares, que tiene su propio horario y canal.
"""
from __future__ import annotations

from app.config import Settings
from app.domain import Reason

SERVICIOS_ESCOLARES = "Servicios Escolares (Control Escolar) del ITTG"


def _coordinacion(cfg: Settings) -> str:
    """"la Coordinación de ISC en horario de atención (...) o escribe a ..." """
    contacto = f" o escribe a {cfg.contact_email}" if cfg.contact_email else ""
    return f"la Coordinación de ISC en horario de atención ({cfg.contact_hours}){contacto}"


def derivation_message(reason: Reason, cfg: Settings, category: str | None = None) -> str:
    if reason is Reason.TOO_LONG:
        return (f"Tu mensaje es demasiado largo (máximo {cfg.max_question_chars} caracteres). "
                "¿Puedes dividirlo en preguntas más cortas?")
    if reason is Reason.OUT_OF_SCOPE and category == "datos_personales":
        return ("Por protección de datos personales no puedo consultar calificaciones, kárdex "
                f"ni adeudos. Para esta información acude a {SERVICIOS_ESCOLARES}.")
    if reason is Reason.OUT_OF_SCOPE:
        return ("Los documentos oficiales (constancias, certificados) los emite directamente "
                f"{SERVICIOS_ESCOLARES}; solicítalos ahí. "
                "Si quieres, te explico los requisitos para tramitarlos.")
    if reason is Reason.INJECTION:
        return ("Solo puedo ayudarte con dudas académicas y administrativas de la carrera de "
                "Ingeniería en Sistemas Computacionales. ¿Sobre qué trámite necesitas información?")
    if reason is Reason.LLM_UNAVAILABLE:
        return ("En este momento el servicio no está disponible. Intenta de nuevo en unos "
                f"minutos o acude a {_coordinacion(cfg)}.")
    # LOW_SIMILARITY, NOT_IN_CONTEXT, VALIDATION_FAILED: duda académica sin
    # respuesta confiable en los documentos -> la coordinación
    return ("No encontré información suficiente y confiable en los documentos oficiales para "
            "responder tu consulta. Para no darte datos incorrectos, te recomiendo acudir a "
            f"{_coordinacion(cfg)}.")
