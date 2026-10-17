"""Compuertas baratas y deterministas ANTES de gastar una llamada al LLM:
sanitización, detección de inyección de prompts y alcance (RF-RN-01)."""
from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def sanitize(question: str, max_chars: int) -> str:
    q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", question)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:max_chars]


_INJECTION = [
    r"ignora\w*\s+(todas?\s+)?(las\s+)?(instrucciones|reglas|indicaciones)",
    r"olvida\w*\s+(todo|lo\s+anterior|tus\s+instrucciones)",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"(prompt|mensaje)\s+del\s+sistema", r"system\s*prompt",
    r"revela\w*\s+(tus|las)\s+(instrucciones|reglas)",
    r"(actua|comportate|responde)\s+como\s+(si\s+fueras|un|una)\s+(?!estudiante)",
    r"you\s+are\s+now", r"\bjailbreak\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION), re.IGNORECASE)


def looks_like_injection(question: str) -> bool:
    return bool(_INJECTION_RE.search(normalize(question)))


# Categoría -> patrón (sobre texto sin acentos). Reglas iniciales; la
# clasificación por intención con el LLM (HU-05) las complementará después.
_SCOPE_RULES = {
    "datos_personales": re.compile(
        r"\b(kardex|adeudos?|mis\s+calificaciones|mi\s+promedio|mis\s+notas|"
        r"mi\s+historial|estatus\s+de\s+mi\s+tramite|cuanto\s+debo)\b"),
    "documento_oficial": re.compile(
        r"\b(genera|generame|expide|expideme|emite|emiteme|dame|mandame|enviame|"
        r"necesito\s+que\s+me\s+(des|emitas|expidas))\w*\s+(mi\s+|una\s+|la\s+)?"
        r"(constancia|certificado|kardex|boleta|carta)\b"),
}


def out_of_scope_category(question: str) -> str | None:
    q = normalize(question)
    for cat, rx in _SCOPE_RULES.items():
        if rx.search(q):
            return cat
    return None
