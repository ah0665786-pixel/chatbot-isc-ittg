"""Adaptador de Telegram con updates simulados (Update.de_json), sin red.

Nació de un fallo real (RF-I-01, HU-02 CA-01): si el estudiante editaba su
pregunta, Telegram mandaba un edited_message, update.message llegaba en None y
on_message tronaba con AttributeError antes del try: el estudiante se quedaba
sin respuesta.
"""
from __future__ import annotations

import asyncio
import time
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


# ---- indicador "escribiendo…", botones, errores y respuestas largas --------
from telegram.constants import ChatAction, MessageLimit  # noqa: E402
from telegram.error import BadRequest  # noqa: E402


class RecordingBot(FakeBot):
    """Además de los mensajes, guarda el orden de todo lo que se pide."""

    def __init__(self, reject_html: bool = False):
        super().__init__()
        self.events: list[str] = []
        self.actions: list[str] = []
        self.answers: list[str] = []
        self.edited_markup: list = []
        self.reject_html = reject_html

    async def send_message(self, chat_id, text, **kw):
        if self.reject_html and kw.get("parse_mode") == "HTML":
            raise BadRequest("Can't parse entities")
        self.events.append("message")
        await super().send_message(chat_id, text, **kw)

    async def send_chat_action(self, chat_id, action, **kw):
        self.events.append("typing")
        self.actions.append(action)

    async def answer_callback_query(self, callback_query_id, text=None, **kw):
        self.answers.append(text)

    async def edit_message_reply_markup(self, chat_id=None, message_id=None,
                                        reply_markup=None, **kw):
        self.edited_markup.append(reply_markup)


class ScriptedService:
    def __init__(self, result=None, error=None, delay=0.0, rate_error=None):
        self.result, self.error, self.delay, self.rate_error = result, error, delay, rate_error
        self.ratings: list[tuple[str, int]] = []

    def handle_message(self, external_id, display_name, text):
        time.sleep(self.delay)  # corre en asyncio.to_thread, como el pipeline real
        if self.error:
            raise self.error
        return self.result, "m1"

    def rate(self, metric_id, score):
        if self.rate_error:
            raise self.rate_error
        self.ratings.append((metric_id, score))


def _answered(text):
    return RagResult(Outcome.ANSWERED, Reason.NONE, text)


def _callback(bot, data: str) -> Update:
    return Update.de_json({"update_id": 2, "callback_query": {
        "id": "cb1", "chat_instance": "ci", "data": data,
        "from": {"id": 42, "is_bot": False, "first_name": "Ana"},
        "message": {"message_id": 8, "date": 1761300000, "text": "respuesta",
                    "chat": {"id": 42, "type": "private"},
                    "from": {"id": 1, "is_bot": True, "first_name": "Bot"}}}}, bot)


def _run_and_watch_typing(bot, svc, text="¿servicio social?"):
    """Atiende el mensaje y devuelve cuántos "escribiendo…" hubo al responder."""
    async def scenario():
        await telegram_bot.on_message(_update(bot, "message", text), _context(bot, svc))
        at_reply = len(bot.actions)
        await asyncio.sleep(0.05)  # si el ciclo siguiera vivo, mandaría más
        return at_reply

    return asyncio.run(scenario())


def test_escribiendo_se_envia_mientras_piensa_y_se_detiene_al_responder(monkeypatch):
    """HU-02 CA-03 (Lista B; en el código, HU-01 CA-03): el indicador aparece antes de la respuesta, se renueva
    mientras el pipeline trabaja y deja de enviarse al contestar."""
    monkeypatch.setattr(telegram_bot, "TYPING_INTERVAL_S", 0.01)
    bot = RecordingBot()
    svc = ScriptedService(_answered("El servicio social dura 480 horas."), delay=0.1)
    at_reply = _run_and_watch_typing(bot, svc)
    assert bot.events[0] == "typing" and bot.events[-1] == "message"
    assert at_reply >= 2  # se renovó mientras esperaba
    assert len(bot.actions) == at_reply  # y se detuvo
    assert set(bot.actions) == {ChatAction.TYPING}


def test_escribiendo_se_detiene_si_el_servicio_falla(monkeypatch):
    monkeypatch.setattr(telegram_bot, "TYPING_INTERVAL_S", 0.01)
    bot = RecordingBot()
    svc = ScriptedService(error=RuntimeError("se cayó la BD"), delay=0.05)
    at_reply = _run_and_watch_typing(bot, svc)
    assert len(bot.actions) == at_reply
    assert len(bot.sent) == 1 and "problema inesperado" in bot.sent[0]["text"]
    assert bot.sent[0].get("reply_markup") is None  # no hay nada que calificar


def test_respuesta_derivada_no_lleva_botones():
    bot = RecordingBot()
    derived = RagResult(Outcome.DERIVED, Reason.LOW_SIMILARITY, "Acude a la Coordinación.")
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "x"),
                                        _context(bot, ScriptedService(derived))))
    assert len(bot.sent) == 1 and bot.sent[0]["reply_markup"] is None


def test_botones_de_calificacion_del_1_al_5():
    bot = RecordingBot()
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "x"),
                                        _context(bot, ScriptedService(_answered("Respuesta.")))))
    rows = bot.sent[0]["reply_markup"].inline_keyboard
    assert [b.callback_data for b in rows[0]] == [f"rate:m1:{n}" for n in range(1, 6)]
    assert bot.sent[0]["parse_mode"] == "HTML"


def test_html_rechazado_se_reenvia_en_texto_plano_con_botones():
    bot = RecordingBot(reject_html=True)
    asyncio.run(telegram_bot.on_message(
        _update(bot, "message", "x"),
        _context(bot, ScriptedService(_answered("**Requisitos**: acta de nacimiento.")))))
    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "Requisitos: acta de nacimiento."
    assert not bot.sent[0].get("parse_mode")  # PTB manda DEFAULT_NONE, que es falso
    assert bot.sent[0]["reply_markup"] is not None


def test_respuesta_larga_se_divide_y_los_botones_van_al_final():
    """HU-06 CA-03: antes Telegram la rechazaba (> 4,096), el reintento en
    texto plano también excedía y el estudiante se quedaba sin respuesta."""
    text = "\n\n".join(f"{i}. **Opción {i}**: " + "requisito del reglamento " * 15
                       for i in range(1, 41))
    assert len(text) > MessageLimit.MAX_TEXT_LENGTH
    bot = RecordingBot()
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "x"),
                                        _context(bot, ScriptedService(_answered(text)))))
    assert len(bot.sent) >= 2
    assert all(len(m["text"]) <= MessageLimit.MAX_TEXT_LENGTH for m in bot.sent)
    assert all(m["reply_markup"] is None for m in bot.sent[:-1])
    assert bot.sent[-1]["reply_markup"] is not None
    assert "Opción 1</b>" in bot.sent[0]["text"] and "Opción 40</b>" in bot.sent[-1]["text"]


def test_respuesta_larga_en_texto_plano_tambien_se_divide():
    bot = RecordingBot(reject_html=True)
    text = "\n\n".join("**párrafo** " + "palabra " * 60 for _ in range(30))
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "x"),
                                        _context(bot, ScriptedService(_answered(text)))))
    assert len(bot.sent) >= 2
    assert all(len(m["text"]) <= MessageLimit.MAX_TEXT_LENGTH for m in bot.sent)
    assert all(m["reply_markup"] is None for m in bot.sent[:-1])
    assert bot.sent[-1]["reply_markup"] is not None


def test_calificacion_se_registra_y_se_quitan_los_botones():
    bot, svc = RecordingBot(), ScriptedService()
    asyncio.run(telegram_bot.on_rating(_callback(bot, "rate:m1:4"), _context(bot, svc)))
    assert svc.ratings == [("m1", 4)]
    assert bot.answers == ["¡Gracias por tu calificación!"]
    assert bot.edited_markup == [None]


@pytest.mark.parametrize("data,svc", [
    ("rate:m1:4", ScriptedService(rate_error=RuntimeError("BD caída"))),
    ("rate:m1", ScriptedService()),  # callback_data mal formado
])
def test_calificacion_fallida_avisa_sin_tronar(data, svc):
    bot = RecordingBot()
    asyncio.run(telegram_bot.on_rating(_callback(bot, data), _context(bot, svc)))
    assert svc.ratings == []
    assert bot.answers == ["No se pudo registrar, pero gracias de todas formas."]
    assert bot.edited_markup == []  # los botones siguen para intentarlo de nuevo
