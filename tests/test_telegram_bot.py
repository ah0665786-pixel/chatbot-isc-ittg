"""Adaptador de Telegram con updates simulados (Update.de_json), sin red.

Nació de un fallo real (RF-I-01, HU-02 CA-01): si el estudiante editaba su
pregunta, Telegram mandaba un edited_message, update.message llegaba en None y
on_message tronaba con AttributeError antes del try: el estudiante se quedaba
sin respuesta.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from telegram import Update

from app import telegram_bot
from app.domain import Outcome, RagResult, Reason


class FakeBot:
    """Bot falso: guarda lo que se le pide enviar en vez de llamar a la API."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send_message(self, chat_id, text, **kw):
        self.sent.append({"chat_id": chat_id, "text": text, **kw})

    async def send_chat_action(self, chat_id, action, **kw):
        pass


class FakeService:
    def __init__(self):
        self.questions: list[str] = []

    def handle_message(self, external_id, display_name, text):
        self.questions.append(text)
        return RagResult(Outcome.ANSWERED, Reason.NONE, "El servicio social dura 480 horas."), "m1"


def _update(bot, kind: str, text: str) -> Update:
    msg = {"message_id": 7, "date": 1761300000, "text": text,
           "chat": {"id": 42, "type": "private"},
           "from": {"id": 42, "is_bot": False, "first_name": "Ana"}}
    if kind == "edited_message":
        msg["edit_date"] = 1761300060
    return Update.de_json({"update_id": 1, kind: msg}, bot)


def _context(bot, svc):
    return SimpleNamespace(bot=bot, application=SimpleNamespace(bot_data={"svc": svc}))


@pytest.mark.parametrize("kind", ["message", "edited_message"])
def test_mensaje_nuevo_o_editado_recibe_respuesta(kind):
    bot, svc = FakeBot(), FakeService()
    update = _update(bot, kind, "¿Cuántas horas dura el servicio social?")
    if kind == "edited_message":
        assert update.message is None  # lo que hacía tronar a on_message
    asyncio.run(telegram_bot.on_message(update, _context(bot, svc)))
    assert svc.questions == ["¿Cuántas horas dura el servicio social?"]
    assert len(bot.sent) == 1 and bot.sent[0]["chat_id"] == 42
    assert "480 horas" in bot.sent[0]["text"]
    assert bot.sent[0]["reply_markup"] is not None  # botones de calificación


def test_start_editado_responde_bienvenida():
    bot = FakeBot()
    asyncio.run(telegram_bot.start(_update(bot, "edited_message", "/start"), _context(bot, None)))
    assert bot.sent and bot.sent[0]["text"] == telegram_bot.WELCOME
