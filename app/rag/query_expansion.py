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

Para agregar un tema: un patrón sobre el texto normalizado (minúsculas, sin
acentos) y el texto con las palabras que usa el documento. Medir antes y
después con:  python -m app.cli calibrate data/retrieval_cases.jsonl
"""
from __future__ import annotations

import re

from app.rag import guards

EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    # titular, titularme, titulación, titulo/título...
    (re.compile(r"\btitul\w*"),
     "Titulación integral: tipos de proyectos de titulación integral "
     "(residencia profesional, tesis, tesina, EGEL)."),
]


def expand_query(question: str) -> str:
    """Devuelve la pregunta con el texto de expansión de cada tema que mencione."""
    norm = guards.normalize(question)
    extra = [text for pattern, text in EXPANSIONS if pattern.search(norm)]
    return "\n".join([question, *extra]) if extra else question
