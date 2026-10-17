"""Mensajes al estudiante cuando se deriva a atención humana (RF-C-04, HU-08)."""
from __future__ import annotations

from app.config import Settings
from app.domain import Reason


def derivation_message(reason: Reason, cfg: Settings, category: str | None = None) -> str:
    contacto = f" o escribe a {cfg.contact_email}" if cfg.contact_email else ""
    horario = f"en horario de atención ({cfg.contact_hours})"
    if reason is Reason.TOO_LONG:
        return (f"Tu mensaje es demasiado largo (máximo {cfg.max_question_chars} caracteres). "
                "¿Puedes dividirlo en preguntas más cortas?")
    if reason is Reason.OUT_OF_SCOPE and category == "datos_personales":
        return ("Por protección de datos personales no puedo consultar calificaciones, kardex "
                "ni adeudos. Para esta información acude a Servicios Escolares "
                f"{horario}{contacto}.")
    if reason is Reason.OUT_OF_SCOPE:
        return ("Los documentos oficiales (constancias, certificados) los emite directamente "
                f"Servicios Escolares. Puedes solicitarlos {horario}{contacto}. "
                "Si quieres, te explico los requisitos para tramitarlos.")
    if reason is Reason.INJECTION:
        return ("Solo puedo ayudarte con dudas académicas y administrativas de la carrera de "
                "Ingeniería en Sistemas Computacionales. ¿Sobre qué trámite necesitas información?")
    if reason is Reason.LLM_UNAVAILABLE:
        return ("En este momento el servicio no está disponible. Intenta de nuevo en unos "
                f"minutos o acude a la Coordinación de ISC {horario}{contacto}.")
    return ("No encontré información suficiente y confiable en los documentos oficiales para "
            "responder tu consulta. Para no darte datos incorrectos, te recomiendo acudir a la "
            f"Coordinación de ISC {horario}{contacto}.")
