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
from app.adapters.llm_openai_compat import QueueInfo
from app.domain import Outcome, RagResult, Reason
from app.rag.messages import HIGH_DEMAND_MESSAGE, QUEUED_MESSAGE


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

    def handle_message(self, external_id, display_name, text, on_queued=None):
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
    def __init__(self, result=None, error=None, delay=0.0, rate_error=None, queued=False):
        self.result, self.error, self.delay, self.rate_error = result, error, delay, rate_error
        self.queued = queued  # simula que la consulta entra a la fila por límite de uso
        self.ratings: list[tuple[str, int]] = []

    def handle_message(self, external_id, display_name, text, on_queued=None):
        if self.queued and on_queued is not None:
            on_queued(QueueInfo(position=1, estimated_wait=3.0))  # desde el hilo, como la fila real
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


# ---- fila por límite de uso del proveedor ---------------------------------------
def test_aviso_de_fila_llega_antes_de_la_respuesta_y_el_escribiendo_sigue(monkeypatch):
    """Si la consulta entra a la fila, el estudiante recibe primero el aviso de
    espera y luego la respuesta con sus botones; el "escribiendo…" sigue
    mientras espera y se detiene al responder."""
    monkeypatch.setattr(telegram_bot, "TYPING_INTERVAL_S", 0.01)
    bot = RecordingBot()
    svc = ScriptedService(_answered("El servicio social dura 480 horas."), delay=0.1, queued=True)
    at_reply = _run_and_watch_typing(bot, svc)
    texts = [m["text"] for m in bot.sent]
    assert len(texts) == 2
    assert "en fila" in texts[0] and texts[0] == QUEUED_MESSAGE
    assert bot.sent[0].get("reply_markup") is None  # el aviso no se califica
    assert "480 horas" in texts[1] and bot.sent[1]["reply_markup"] is not None
    assert at_reply >= 2 and len(bot.actions) == at_reply


def test_aviso_de_fila_y_respuesta_larga_dividida_con_botones_al_final():
    bot = RecordingBot()
    long = "\n\n".join(f"{i}. " + "requisito " * 60 for i in range(1, 15))
    svc = ScriptedService(_answered(long), queued=True)
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "¿requisitos?"), _context(bot, svc)))
    assert bot.sent[0]["text"] == QUEUED_MESSAGE
    rest = bot.sent[1:]
    assert len(rest) >= 2
    assert all(m.get("reply_markup") is None for m in rest[:-1])
    assert rest[-1]["reply_markup"] is not None


def test_alta_demanda_tras_la_fila_no_lleva_botones():
    bot = RecordingBot()
    res = RagResult(Outcome.DERIVED, Reason.HIGH_DEMAND, HIGH_DEMAND_MESSAGE)
    svc = ScriptedService(res, queued=True)
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "¿horario?"), _context(bot, svc)))
    assert [m["text"] for m in bot.sent] == [QUEUED_MESSAGE, HIGH_DEMAND_MESSAGE]
    assert all(m.get("reply_markup") is None for m in bot.sent)
    assert "Coordinación" not in bot.sent[1]["text"]  # no se deriva


def test_sin_fila_no_hay_aviso_de_espera():
    bot = RecordingBot()
    svc = ScriptedService(_answered("El servicio social dura 480 horas."))
    asyncio.run(telegram_bot.on_message(_update(bot, "message", "¿servicio?"), _context(bot, svc)))
    assert len(bot.sent) == 1 and QUEUED_MESSAGE not in bot.sent[0]["text"]


def test_dos_estudiantes_se_atienden_a_la_vez_y_uno_mismo_en_orden():
    """Con concurrent_updates, dos estudiantes distintos no se esperan entre sí;
    dos mensajes del mismo estudiante sí van en orden (candado por estudiante)."""
    import threading

    class SlowService(ScriptedService):
        def __init__(self):
            super().__init__(_answered("ok, respuesta"), delay=0.2)
            self.active = 0
            self.peak = 0
            self._lock = threading.Lock()

        def handle_message(self, external_id, display_name, text, on_queued=None):
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                return super().handle_message(external_id, display_name, text, on_queued)
            finally:
                with self._lock:
                    self.active -= 1

    def upd(bot, uid, text):
        u = _update(bot, "message", text).to_dict()
        u["message"]["from"]["id"] = u["message"]["chat"]["id"] = uid
        return Update.de_json(u, bot)

    async def scenario(uids):
        bot, svc = RecordingBot(), SlowService()
        ctx = _context(bot, svc)
        t0 = time.perf_counter()
        await asyncio.gather(*(telegram_bot.on_message(upd(bot, uid, f"pregunta {n}"), ctx)
                               for n, uid in enumerate(uids)))
        return svc.peak, time.perf_counter() - t0

    peak, _ = asyncio.run(scenario([1, 2]))
    assert peak == 2  # distintos estudiantes, al mismo tiempo
    peak, _ = asyncio.run(scenario([1, 1]))
    assert peak == 1  # el mismo estudiante, uno tras otro


def test_build_application_atiende_varios_mensajes_a_la_vez(monkeypatch):
    """python-telegram-bot atiende un mensaje a la vez si no se le pide otra cosa."""
    from app.config import Settings

    class FakeRuntime:
        repo = pipeline = None

    monkeypatch.setattr(telegram_bot, "build_runtime", lambda cfg, fake_embeddings=False: FakeRuntime())
    cfg = Settings(bot_concurrency=7)
    app = telegram_bot.build_application(cfg, "123:ABC")
    assert app.concurrent_updates == 7
