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
_CONVALIDACION = re.compile(r"\b(convalid|revalid)\w*")
_DEFINICION = re.compile(r"\bque (es|son|significa)\b|\ben que consiste\b|\bdefinicion\b")
# Reprobar/repetir SOLO junto a materia/asignatura (o el nombre del curso): así
# "¿Puedo repetir la residencia?" no se lleva esta expansión.
# Ojo con la conjugación: "repruebo", "repito" no empiezan con "reprob"/"repet".
_REPROBAR = r"(reprob|reprueb|repet|repit)\w*"
_REPETICION = re.compile(rf"\b(materia|asignatura)s?\b.*\b{_REPROBAR}"
                         rf"|\b{_REPROBAR}\b.*\b(materia|asignatura)s?\b"
                         r"|\bcurso (especial|de repeticion)\b|\ben repeticion\b")
_ACTIVIDADES = re.compile(r"\bactividad(es)? complementaria")
# Solo preguntas de créditos/horas/plazos: con "¿Qué son?", "¿cómo se acredita?"
# o la del inglés, la expansión las empeoraba (de 2.º a 8.º); sin ella ya salen bien.
_CREDITOS_PLAZO = re.compile(r"\b(credito\w*|maximo|valor|vale|horas?|semestres?|plazo"
                             r"|cuando|hasta|tiempo|limite)\b")

# Los textos usan el VOCABULARIO del documento, no la respuesta: si el
# documento cambia, la expansión no debe empujar hacia un dato viejo.
EXPANSIONS: list[tuple[re.Pattern[str], re.Pattern[str] | None, str]] = [
    (_TITULACION, _REQUISITOS,
     "Titulación integral: requisitos para la titulación integral "
     "(créditos, lengua extranjera, constancia de no inconveniencia)."),
    (_TITULACION, None,
     "Titulación integral: tipos de proyectos de titulación integral "
     "(residencia profesional, tesis, tesina, EGEL)."),
    # "¿Qué es la convalidación?": la definición quedaba 8.ª, detrás de otros
    # fragmentos del mismo capítulo.
    (_CONVALIDACION, _DEFINICION,
     "Convalidación de estudios: definición y caracterización, equiparación y "
     "validación de asignaturas de un plan de estudios a otro."),
    # "repruebo una materia": el documento dice "no acreditó la asignatura" y el
    # cuestionario de hábitos de estudio (Manual-snest-alumnos) ganaba por
    # parecerse al lenguaje del estudiante. El correcto quedaba 13.º a 57.º.
    (_REPETICION, None,
     "Curso de repetición y curso especial: el estudiante no acreditó la asignatura "
     "en el curso ordinario y la cursa por segunda vez o por tercera vez."),
    # Créditos, horas y plazos quedaban 4.º-8.º (el capítulo tiene muchos
    # fragmentos parecidos: tareas del profesor, evaluación...). Con esto, 1.º-2.º.
    (_ACTIVIDADES, _CREDITOS_PLAZO,
     "Políticas de operación de las actividades complementarias: valor curricular, "
     "créditos, horas efectivas y semestres."),
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
