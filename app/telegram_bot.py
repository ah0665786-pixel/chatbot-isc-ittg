"""Capa 1 — Interfaz de Usuario: bot de Telegram (python-telegram-bot 22.x).
Adaptador delgado a propósito: toda la lógica de negocio vive en
app.service.ConversationService, que ya está probada por separado. Aquí
solo se traduce entre la API de Telegram y ese servicio.

Uso:
  python -m app.telegram_bot
Requiere TELEGRAM_BOT_TOKEN en el .env (se saca hablando con @BotFather).
"""
from __future__ import annotations

import asyncio
import logging
import os

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, MessageHandler, filters)

from app.config import Settings
from app.domain import Outcome
from app.runtime import build_runtime
from app.service import ConversationService
from app.telegram_format import markdown_to_telegram_html, strip_markdown

log = logging.getLogger("telegram_bot")

WELCOME = (
    "Hola 👋 Soy el asistente virtual de la Coordinación de Ingeniería en Sistemas "
    "Computacionales del ITTG.\n\n"
    "Puedes preguntarme sobre reglamento, trámites y procedimientos académicos, "
    "en cualquier momento del día. Si no puedo resolver tu consulta con certeza, "
    "te diré a dónde acudir en vez de arriesgarme a darte información incorrecta.")

TYPING_INTERVAL_S = 4  # Telegram borra el indicador solo a los ~5 s (HU-01 CA-03)


async def _typing_loop(chat_id: int, bot) -> None:
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(TYPING_INTERVAL_S)
    except asyncio.CancelledError:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc: ConversationService = context.application.bot_data["svc"]
    user = update.effective_user
    text = (update.message.text or "").strip()
    if not text:
        return

    typing_task = asyncio.create_task(_typing_loop(update.effective_chat.id, context.bot))
    try:
        # El pipeline es síncrono (red + CPU); to_thread evita bloquear el
        # bot mientras se atiende a otros estudiantes al mismo tiempo.
        result, metric_id = await asyncio.to_thread(
            svc.handle_message, str(user.id), user.first_name or user.username or "", text)
    except Exception:
        log.exception("fallo procesando mensaje de %s", user.id)
        await update.message.reply_text(
            "Tuve un problema inesperado procesando tu consulta. Intenta de nuevo en un "
            "momento; si persiste, acude a la Coordinación de ISC.")
        return
    finally:
        typing_task.cancel()

    keyboard = None
    if result.outcome is Outcome.ANSWERED:
        row = [InlineKeyboardButton(str(n), callback_data=f"rate:{metric_id}:{n}")
               for n in range(1, 6)]
        keyboard = InlineKeyboardMarkup([row])
    try:
        await update.message.reply_text(markdown_to_telegram_html(result.text),
                                        parse_mode=ParseMode.HTML, reply_markup=keyboard)
    except BadRequest:
        # Telegram rechazó el HTML (etiquetas mal formadas): mejor texto plano
        # que dejar al estudiante sin respuesta.
        log.warning("Telegram rechazó el HTML; se envía como texto plano")
        await update.message.reply_text(strip_markdown(result.text), reply_markup=keyboard)


async def on_rating(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    svc: ConversationService = context.application.bot_data["svc"]
    query = update.callback_query
    try:
        _, metric_id, score = query.data.split(":")
        await asyncio.to_thread(svc.rate, metric_id, int(score))
        await query.answer("¡Gracias por tu calificación!")
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        log.exception("fallo registrando calificación")
        await query.answer("No se pudo registrar, pero gracias de todas formas.")


def build_application(cfg: Settings, token: str, fake_embeddings: bool = False) -> Application:
    rt = build_runtime(cfg, fake_embeddings=fake_embeddings)
    svc = ConversationService(cfg, rt.repo, rt.pipeline)
    app = Application.builder().token(token).build()
    app.bot_data["svc"] = svc
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ayuda", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_handler(CallbackQueryHandler(on_rating, pattern=r"^rate:"))
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # baja el ruido de cada llamada HTTP
    cfg = Settings.from_env()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Falta TELEGRAM_BOT_TOKEN en el .env (lo da @BotFather en Telegram)")
    app = build_application(cfg, token)
    log.info("Bot iniciado, esperando mensajes (Ctrl+C para detener)...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
