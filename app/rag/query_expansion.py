"""Expansión de la consulta: acerca las palabras del estudiante a las que usan
los documentos oficiales, SOLO para buscar fragmentos. El LLM y la validación
siguen viendo la pregunta original.

Por qué hace falta: el modelo de embeddings da puntajes muy apretados (los 60
mejores fragmentos caben en ~0.05), así que una pregunta redactada distinto al
documento pierde el fragmento correcto. Ejemplo medido: "¿Cómo me puedo
titular?" dejaba en el lugar 43 el fragmento con las 10 opciones (Manual de
Lineamientos, pág. 103), porque el documento no dice "opciones" sino "tipos de
proyectos de titulación integral". Con la expansión queda entre los 5 primeros
(ver data/retrieval_cases.jsonl y tests/test_retrieval_real.py).

Para agregar un tema: patrón del tema e intención sobre el texto normalizado
(minúsculas, sin acentos) y el texto con las palabras que usa el documento.
Medir antes y después con:  python -m app.cli calibrate data/retrieval_cases.jsonl
"""
from __future__ import annotations

import re

from app.rag import guards

# (tema, intención, texto). Por tema se usa SOLO la primera regla cuya
# intención coincida: la expansión equivocada empeora la búsqueda. Medido:
# con el texto de "opciones", "¿Qué requisitos necesito para titularme?" bajaba
# el fragmento de requisitos (Manual, pág. 104) del lugar 4 al 47; con el de
# requisitos queda 1.º. intención None = la regla por defecto del tema.
_TITULACION = re.compile(r"\btitul\w*")  # titular, titularme, titulación, título...
_REQUISITOS = re.compile(r"\b(requisit\w*|necesit\w*|pide\w*|piden|ingles|idioma|lengua\w*"
                         r"|credito\w*|constancia\w*)\b")

EXPANSIONS: list[tuple[re.Pattern[str], re.Pattern[str] | None, str]] = [
    (_TITULACION, _REQUISITOS,
     "Titulación integral: requisitos para la titulación integral "
     "(créditos, lengua extranjera, constancia de no inconveniencia)."),
    (_TITULACION, None,
     "Titulación integral: tipos de proyectos de titulación integral "
     "(residencia profesional, tesis, tesina, EGEL)."),
]


def expand_query(question: str) -> str:
    """Devuelve la pregunta con el texto de expansión de cada tema que mencione."""
    norm = guards.normalize(question)
    extra: list[str] = []
    done: set[str] = set()
    for topic, intent, text in EXPANSIONS:
        if topic.pattern in done or not topic.search(norm):
            continue
        if intent is None or intent.search(norm):
            extra.append(text)
            done.add(topic.pattern)
    return "\n".join([question, *extra]) if extra else question
