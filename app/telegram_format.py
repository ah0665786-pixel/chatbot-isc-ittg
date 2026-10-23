"""Convierte el Markdown que escribe el LLM (**negrita**, `código`, ### títulos)
al HTML que entiende Telegram. Sin esto el estudiante ve los asteriscos
literales, porque reply_text manda el texto sin formato.

Se usa HTML y no el modo Markdown de Telegram porque este último rechaza el
mensaje completo ante cualquier "*" o "_" sin cerrar, algo común en textos
de reglamento (p. ej. "art_12"). Con HTML basta con escapar < > &.
"""
from __future__ import annotations

import html
import re

_CODE_BLOCK = re.compile(r"```[^\n]*\n?(.*?)```", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
_BULLET = re.compile(r"^([ \t]*)[*-][ \t]+", re.M)


def markdown_to_telegram_html(text: str) -> str:
    """Devuelve texto listo para enviar con parse_mode=HTML."""
    # Los fragmentos de código se apartan primero para no tocar su contenido.
    saved: list[str] = []

    def keep(fragment: str) -> str:
        saved.append(fragment)
        return f"\x00{len(saved) - 1}\x00"

    text = _CODE_BLOCK.sub(lambda m: keep(f"<pre>{html.escape(m.group(1).strip())}</pre>"), text)
    text = _INLINE_CODE.sub(lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), text)

    text = html.escape(text, quote=False)
    text = _HEADING.sub(r"<b>\1</b>", text)
    text = _BULLET.sub(r"\1• ", text)
    text = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = text.replace("**", "")  # restos sin pareja: mejor quitarlos que mostrarlos

    return re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], text)


def strip_markdown(text: str) -> str:
    """Texto plano sin marcas, por si Telegram rechaza el HTML."""
    text = _CODE_BLOCK.sub(lambda m: m.group(1).strip(), text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub(r"\1", text)
    text = _BULLET.sub(r"\1• ", text)
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)
    return text.replace("**", "")
