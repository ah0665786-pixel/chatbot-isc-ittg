"""Pruebas del aviso por correo a la coordinadora (HU-11, UC-12)."""
from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import Settings
from app.domain import Outcome, Reason, RagResult
from app.notify import notify_coordinadora, should_notify


def _cfg(**overrides) -> Settings:
    base = Settings(
        smtp_host="sandbox.smtp.mailtrap.io", smtp_port=587,
        smtp_user="user", smtp_password="pass", smtp_use_tls=True,
        coordinadora_email="coordinadora@test.local",
    )
    return replace(base, **overrides)


def _derived(reason: Reason) -> RagResult:
    return RagResult(Outcome.DERIVED, reason, "mensaje de orientación", [], 0.0,
                     "", "", 0, {"total": 0.0}, "detalle de prueba", [])


@pytest.mark.parametrize("reason", [
    Reason.OUT_OF_SCOPE, Reason.LOW_SIMILARITY, Reason.NOT_IN_CONTEXT,
    Reason.VALIDATION_FAILED, Reason.LLM_UNAVAILABLE,
])
def test_should_notify_for_escalable_reasons(reason):
    assert should_notify(_derived(reason)) is True


@pytest.mark.parametrize("reason", [Reason.INJECTION, Reason.TOO_LONG, Reason.NONE])
def test_should_not_notify_for_non_escalable_reasons(reason):
    assert should_notify(_derived(reason)) is False


def test_notify_sends_email_via_smtp(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=10):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent["starttls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, msg):
            sent["message"] = msg

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    cfg = _cfg()
    ok = notify_coordinadora(cfg, "12345", "Isagi", "¿cómo tramito mi residencia?",
                             _derived(Reason.LOW_SIMILARITY))

    assert ok is True
    assert sent["host"] == "sandbox.smtp.mailtrap.io"
    assert sent["login"] == ("user", "pass")
    assert sent["starttls"] is True
    assert "LOW_SIMILARITY" in sent["message"]["Subject"] or \
        Reason.LOW_SIMILARITY.value in sent["message"]["Subject"]
    assert sent["message"]["To"] == "coordinadora@test.local"


def test_notify_returns_false_without_smtp_config():
    cfg = _cfg(smtp_host="", coordinadora_email="")
    ok = notify_coordinadora(cfg, "12345", "Isagi", "pregunta",
                             _derived(Reason.LOW_SIMILARITY))
    assert ok is False


def test_notify_does_not_raise_when_smtp_fails(monkeypatch):
    class BoomSMTP:
        def __init__(self, *a, **k):
            raise ConnectionRefusedError("sin conexión")

    monkeypatch.setattr("smtplib.SMTP", BoomSMTP)
    cfg = _cfg()
    ok = notify_coordinadora(cfg, "12345", "Isagi", "pregunta",
                             _derived(Reason.LOW_SIMILARITY))
    assert ok is False
