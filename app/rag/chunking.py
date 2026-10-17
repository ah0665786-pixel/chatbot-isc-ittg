"""Fragmentación por oraciones con solapamiento, medida en palabras.
(Se mide en palabras y no en tokens para no depender de un tokenizador;
el tamaño se calibra contra el límite del modelo de embeddings.)"""
from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?…;:])\s+|\n{2,}|\n(?=\s*(?:\d+[.)]|[-•*]|[a-z]\)))")


def _sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text.replace("\r", ""))
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]


def _split_long(sentence: str, size: int) -> list[str]:
    w = sentence.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def chunk_text(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("requiere 0 <= overlap < size")
    units: list[str] = []
    for s in _sentences(text):
        units.extend(_split_long(s, size) if len(s.split()) > size else [s])

    chunks: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for u in units:
        uw = len(u.split())
        if cur and cur_w + uw > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tw = 0
            for prev in reversed(cur):  # solapamiento: últimas oraciones hasta ~overlap
                pw = len(prev.split())
                if tw + pw > overlap:
                    break
                tail.insert(0, prev)
                tw += pw
            cur, cur_w = tail, tw
        cur.append(u)
        cur_w += uw
    if cur:
        chunks.append(" ".join(cur))
    return chunks
