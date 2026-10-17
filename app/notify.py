"""Aviso por correo a la coordinadora cuando el sistema escala una consulta
a atención humana (HU-11, UC-12). Usa smtplib de la librería estándar: cero
dependencias nuevas. Si el envío falla (sin internet, credenciales mal
puestas, SMTP caído), NUNCA debe tumbar la respuesta al estudiante: solo se
registra el error en el log y se sigue.
"""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from app.config import Settings
from app.domain import RagResult, Reason

log = logging.getLogger("notify")

# Motivos que sí ameritan avisar a la coordinadora. INJECTION es un intento
# de ataque (no una consulta real que necesite su atención) y TOO_LONG es
# un error del propio mensaje del estudiante: ninguno de los dos se escala.
_ESCALABLE = {
    Reason.OUT_OF_SCOPE,
    Reason.LOW_SIMILARITY,
    Reason.NOT_IN_CONTEXT,
    Reason.VALIDATION_FAILED,
    Reason.LLM_UNAVAILABLE,
}


def should_notify(result: RagResult) -> bool:
    return result.reason in _ESCALABLE


def notify_coordinadora(cfg: Settings, telegram_id: str, display_name: str,
                        question: str, result: RagResult) -> bool:
    """Envía el aviso por correo. Devuelve True si se envió, False si no
    había configuración de correo o si falló (el motivo queda en el log)."""
    if not (cfg.smtp_host and cfg.coordinadora_email):
        return False

    msg = EmailMessage()
    msg["Subject"] = f"[Chatbot ISC] Consulta derivada — {result.reason.value}"
    msg["From"] = cfg.smtp_user or "chatbot-isc@noreply.local"
    msg["To"] = cfg.coordinadora_email
    msg.set_content(
        f"El chatbot no pudo resolver una consulta y la derivó a atención humana.\n\n"
        f"Estudiante (ID Telegram): {telegram_id}\n"
        f"Nombre: {display_name or '(no proporcionado)'}\n"
        f"Motivo de derivación: {result.reason.value}\n"
        f"Detalle: {result.detail}\n\n"
        f"Consulta del estudiante:\n{question}\n"
    )

    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10) as server:
            if cfg.smtp_use_tls:
                server.starttls()
            if cfg.smtp_user:
                server.login(cfg.smtp_user, cfg.smtp_password)
            server.send_message(msg)
        return True
    except Exception as e:  # noqa: BLE001 — nunca debe tumbar la respuesta al estudiante
        log.warning("no se pudo enviar el aviso a la coordinadora: %s", e)
        return False
