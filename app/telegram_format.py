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


# Telegram rechaza mensajes de más de 4,096 caracteres (HU-06 CA-03). Se deja
# margen porque Telegram cuenta en UTF-16 y un emoji ocupa 2.
TELEGRAM_LIMIT = 4096
SAFE_LIMIT = 4000


def split_markdown(text: str, limit: int = SAFE_LIMIT) -> list[str]:
    """Parte el Markdown del LLM en trozos cuyo HTML quepa en un mensaje.

    Corta por párrafo; si un párrafo no cabe, por línea; luego por palabra y,
    como último recurso, a la mitad de una palabra. Cada trozo se convierte
    por separado, así que ninguna etiqueta HTML queda abierta entre mensajes,
    y un bloque ``` no se parte entre párrafos mientras quepa entero."""
    def fits(s: str) -> bool:
        return len(markdown_to_telegram_html(s)) <= limit

    parts = [p.strip() for p in _pieces(text, fits, ("\n\n", "\n", " ")) if p.strip()]
    return parts or [text]


def _pieces(text: str, fits, seps: tuple[str, ...]) -> list[str]:
    if fits(text):
        return [text]
    if not seps:
        return _hard_cut(text, fits)
    sep, rest = seps[0], seps[1:]
    out: list[str] = []
    cur: str | None = None
    for block in _blocks(text, sep):
        if cur is not None and fits(cur + sep + block):
            cur += sep + block
            continue
        if cur is not None:
            out.append(cur)
        sub = [block] if fits(block) else _pieces(block, fits, rest)
        out.extend(sub[:-1])
        cur = sub[-1]
    if cur is not None:
        out.append(cur)
    return out


def _blocks(text: str, sep: str) -> list[str]:
    """Divide por `sep`, pero sin separar las dos mitades de un bloque ```."""
    blocks: list[str] = []
    for raw in text.split(sep):
        if blocks and blocks[-1].count("```") % 2:
            blocks[-1] += sep + raw
        else:
            blocks.append(raw)
    return blocks


def _hard_cut(text: str, fits) -> list[str]:
    out = []
    while text:
        lo, hi = 1, len(text)  # el prefijo más largo que cabe (búsqueda binaria)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if fits(text[:mid]) else (lo, mid - 1)
        out.append(text[:lo])
        text = text[lo:]
    return out
