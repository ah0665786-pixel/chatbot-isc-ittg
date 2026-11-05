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
# "¿Puedo repetir la residencia si la repruebo?": la Guía responde con "se cursa
# por una única ocasión" (sin "repetir" ni "reprobar") y el modelo dijo NO_INFO
# con ese fragmento 4.º, tanto con main como con la rama (30-oct).
# "resi": así le dicen los estudiantes ("si me va mal en la resi...").
_RESIDENCIA = re.compile(r"\bresi(dencias?|dente)?\b")
_REPETIR_RESIDENCIA = re.compile(rf"\b{_REPROBAR}|\b(otra vez|de nuevo|segunda vez|volver a"
                                 r"|no acredit\w*|segunda oportunidad|va mal)\b")
# 2026-11-05, base solo con el Manual: sin la Guía ni el Lineamiento de 2015,
# los numerales del Manual quedaban 5.º a 12.º (págs. 86, 89 y 93).
_CIERRE_EMPRESA = re.compile(r"\b(cerr\w*|cierre|quebr\w*|bancarrota|huelga\w*|cancel\w*)\b")
_DURACION = re.compile(r"\b(dura\w*|duracion|horas?|meses|tiempo|vale|valor)\b")
_REQUISITOS_RESIDENCIA = re.compile(r"\b(requisit\w*|necesit\w*|ocup\w*|piden?|porcentaje"
                                    r"|creditos?|servicio social|liber\w*|entrar|empezar|inici\w*"
                                    r"|hacer)\b")
_ACTIVIDADES = re.compile(r"\b(actividad(es)? complementaria|creditos? complementario)")
# Solo preguntas de créditos/horas/plazos: con "¿Qué son?", "¿cómo se acredita?"
# o la del inglés, la expansión las empeoraba (de 2.º a 8.º); sin ella ya salen bien.
_CREDITOS_PLAZO = re.compile(r"\b(credito\w*|maximo|valor|vale|horas?|semestres?|plazo"
                             r"|cuando|hasta|tiempo|limite|ocupo)\b")
_UNA_ACTIVIDAD = re.compile(r"\b(una sola|cada una|por actividad)\b")
_ACREDITAR = re.compile(r"\b(acredit\w*|cumpl\w*|liber\w*|constancia\w*|evalu\w*)\b")
_TRASLADO = re.compile(r"\b(traslad\w*|cambi\w* (a|de) (otro )?(tec|tecnologico|instituto"
                       r"|escuela|plantel|campus))\b")
_VECES = re.compile(r"\b(veces|cuantas|ocasion\w*)\b")
_SITUACION = re.compile(r"\b(especial|repeticion|adeud\w*|baja|situacion)\b|"
                        + _REPROBAR)
_CAMBIO_CARRERA = re.compile(r"\b((convalid|revalid)\w*|cambi\w* de (carrera|plan)"
                             r"|otra carrera)")
_LIMITE_CONVALIDACION = re.compile(r"\b(veces|cuantas|ocasion\w*|semestres?|limite|hasta)\b")
# Servicio social: "el servicio" a secas también cuenta ("¿desde qué semestre puedo
# hacer el servicio?"); "servicios escolares" no, porque va en plural.
_SERVICIO_SOCIAL = re.compile(r"\bservicio( social)?\b")
# Cuándo o con cuántos créditos se empieza. Se excluyen las de horas, plan de
# trabajo, entregas, terminar o liberar: ya recibían su fragmento y esta
# expansión las alejaría de él ("ya empecé el servicio, ¿para cuándo entrego
# el plan de trabajo?" dice "empecé" y "cuándo").
_EMPEZAR_SERVICIO = re.compile(
    r"^(?!.*\b(plan de trabajo|entreg\w*|reporte\w*|horas?|terminar|termino|acabar|acabo"
    r"|liber\w*|meses|anos)\b)"
    r".*\b(empe[zc]\w*|inici\w*|comenz\w*|desde|cuando|porcentaje|semestres?|llevo"
    r"|necesit\w*|requisit\w*|meter)\b")

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
    (_RESIDENCIA, _REPETIR_RESIDENCIA,
     "Generalidades de la Residencia Profesional: se cursa por una única ocasión; por "
     "causas justificadas fuera del alcance del residente, segunda asignación de proyecto."),
    (_RESIDENCIA, _CIERRE_EMPRESA,
     "Residencia profesional, por circunstancias especiales no imputables al residente "
     "(huelgas, bancarrota, cierre de empresa, organismo o dependencia): cancelación y "
     "reasignación de otro proyecto de residencia profesional."),
    (_RESIDENCIA, _DURACION,
     "Generalidades de la Residencia Profesional: valor curricular en créditos y "
     "duración, periodo de meses como tiempo mínimo y tiempo máximo, horas a acumular."),
    (_RESIDENCIA, _REQUISITOS_RESIDENCIA,
     "Requisitos del candidato para realizar la residencia profesional: acreditación del "
     "servicio social, de todas las actividades complementarias, créditos aprobados de su "
     "plan de estudios, ninguna asignatura en curso especial."),
    (_ACTIVIDADES, _UNA_ACTIVIDAD,
     "Políticas de operación de las actividades complementarias: créditos asignados para "
     "cada una de las actividades complementarias autorizadas y valor curricular del conjunto."),
    (_ACTIVIDADES, _CREDITOS_PLAZO,
     "Políticas de operación de las actividades complementarias: valor curricular, "
     "créditos, horas efectivas y semestres."),
    (_ACTIVIDADES, _ACREDITAR,
     "Cumplimiento de las actividades complementarias: evidencias, constancia de "
     "competencia de actividad complementaria emitida por el departamento responsable."),
    (_TRASLADO, _VECES,
     "Generalidades del traslado de estudiantes: número de traslados entre las "
     "instituciones adscritas al TecNM y periodo reglamentario de semestres."),
    (_TRASLADO, _SITUACION,
     "Características del traslado de estudiantes: el traslado procede independientemente "
     "de la situación académica, asignaturas en curso de repetición y/o especial."),
    (_TRASLADO, None,
     "Del traslado de estudiantes: solicitud de traslado y kárdex o constancia de "
     "calificaciones ante la División de Estudios Profesionales del tecnológico de origen, "
     "previo al periodo de reinscripción del instituto receptor."),
    (_CAMBIO_CARRERA, _LIMITE_CONVALIDACION,
     "Políticas de operación de la convalidación: cambio de carrera, número de control, "
     "semestre de ubicación del estudiante y semestres reglamentarios."),
    # 2026-11-05: el 70% de créditos para empezar el servicio social solo está en
    # el glosario del Manual (pág. 183, "Expediente del programa semestral de
    # Servicio Social"); el 11.4.1.3 (pág. 77) remite a "la normativa vigente".
    # Ese fragmento del glosario sigue con otras definiciones ("Formación de
    # tutores", "Formación integral"), así que quedaba 17.º a 21.º o fuera de los
    # 60 primeros. Con el texto de la definición, más "formación integral" (el
    # resto del mismo fragmento), queda entre los 5 primeros sin desplazar a la
    # pág. 77. Medido con y sin la última frase: sin ella quedaba 6.º a 17.º.
    (_SERVICIO_SOCIAL, _EMPEZAR_SERVICIO,
     "Expediente del programa semestral de Servicio Social. Archivo con documentos que "
     "hacen constar la planeación y ejecución del programa de Servicio Social, que se "
     "integra con la convocatoria publicada, la relación de estudiantes con el porcentaje "
     "de créditos aprobados y el acta de calificaciones semestral de los estudiantes que "
     "concluyen el Servicio Social. Formación integral del estudiante."),
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


# ---- preguntas de seguimiento (HU-04 CA-02) ---------------------------------
# "¿Y cuántas horas son?" después de "¿Cuántas horas dura el servicio social?"
# no dice de qué habla: buscada sola recuperaba 0 de 5 fragmentos útiles y aun
# así pasaba el umbral (0.858), porque el modelo da puntajes muy juntos. Para
# esas preguntas la búsqueda antepone la última pregunta del estudiante (sin
# llamar al LLM). El LLM y la validación siguen viendo la pregunta original.
#
# Es seguimiento si (1) empieza con un conector ("y", "entonces", "también"...),
# (2) señala algo dicho antes ("eso", "ahí", "dicho trámite") o (3) no nombra
# ningún tema: todas sus palabras son genéricas ("¿En cuánto tiempo lo tengo
# que terminar?"). "este/esta" no cuentan: "este semestre" o "está" son comunes
# en preguntas independientes. Tampoco "mismo/misma" ("la misma materia").
_CONECTOR = re.compile(r"^(y|e|o|pero|entonces|tambien|ademas|igual|o sea)\b")
_ANAFORA = re.compile(r"\b(eso|esto|ese|esa|esos|esas|ello|ahi|alli|dicho|dicha|dichos|dichas"
                      r"|anterior|mencionado|mencionada)\b")
# Palabras que no dicen de qué trámite se habla. Solo se cuentan las de 4+ letras.
_GENERICAS = set("""cual cuales cuando cuanto cuanta cuantos cuantas donde como quien quienes
porque para pero sino desde hasta entre sobre tambien ademas entonces igual
tengo tiene tienen tienes puedo puede pueden puedes debo debe deben necesito necesita
necesitan hacer hago hace hacen hacerlo pedir piden pide solicitar solicito sacar entregar
empezar empiezo iniciar inicio terminar termino acabar dura duran durar cuesta cuestan vale
valen cuenta cuentan sirve pasa pasaria sigue seria sera son esta estan hay haber
horas hora tiempo dias meses semanas creditos credito requisitos requisito documentos
documento pasos paso plazo plazos fecha fechas costo precio porcentaje minimo maximo veces
semestre semestres lugar lugares opciones opcion tipos tipo forma formas manera maneras
caso casos repruebo reprobar repito repetir acredito acreditar mismo misma algo nada mas
menos tanto tantas tantos todo todos toda todas cada otro otra otros otras""".split())


_MAX_PREVIAS = 3  # tope por si toda la conversación son seguimientos


def is_follow_up(question: str) -> bool:
    """¿La pregunta depende de lo que se habló antes para saber de qué trata?"""
    norm = " ".join(re.findall(r"[a-z0-9]+", guards.normalize(question)))
    if not norm:
        return False
    if _CONECTOR.search(norm) or _ANAFORA.search(norm):
        return True
    return not any(len(w) >= 4 and w not in _GENERICAS for w in norm.split())


def search_query(question: str, history=()) -> str:
    """Consulta para el buscador: la pregunta más la expansión por tema. Si es
    de seguimiento, antepone las preguntas previas del estudiante hasta la
    última que sí nombraba su tema (en "¿Y cuántos créditos vale?" -> "¿Y
    cuándo lo empiezo?", el tema sigue siendo el de la pregunta de antes)."""
    if is_follow_up(question):
        question = "\n".join([*reversed(_previous_questions(history)), question])
    return expand_query(question)


def _previous_questions(history) -> list[str]:
    """Preguntas previas del estudiante, de la más reciente hacia atrás, hasta
    la última que sí nombraba su tema (o _MAX_PREVIAS)."""
    previous: list[str] = []
    for m in reversed(history or ()):
        if m.get("role") != "user" or not m.get("content"):
            continue
        previous.append(m["content"])
        if not is_follow_up(m["content"]) or len(previous) == _MAX_PREVIAS:
            break
    return previous


# ---- tema de la consulta (HU-13 CA-01, HU-15 CA-03, RF-C-01) -----------------
# Cada consulta se guarda con UNO de estos temas para los reportes de la
# coordinadora (tipos de consulta más frecuentes y temas con más derivaciones).
# Las consultas registradas antes de esta columna quedan sin tema: el panel y
# el CSV las muestran como "Sin clasificar".
TOPIC_LABELS = {
    "plan_estudios": "Plan de estudios",
    "residencia": "Residencia profesional",
    "servicio_social": "Servicio social",
    "titulacion": "Titulación",
    "bajas": "Bajas",
    "traslados": "Traslados",
    "actividades": "Actividades complementarias",
    "permisos": "Permisos",
    "otros": "Otros",
}
UNCLASSIFIED_LABEL = "Sin clasificar"

# Orden = prioridad cuando la pregunta nombra varios temas: gana el trámite que
# el estudiante quiere hacer, no el que menciona de paso. "¿Me puedo titular con
# la residencia?" es titulación; "¿Puedo hacer residencias si no he liberado el
# servicio social?" es residencia; "Me dieron de baja definitiva, ¿puedo entrar
# a otra carrera?" es traslado; "Si me cambio de carrera, ¿me reconocen las
# materias?" es traslado, no plan de estudios (por eso plan va al final:
# "materia", "créditos" o "semestre" aparecen en preguntas de todos los temas).
# Convalidación, equivalencia y cambio de carrera van con traslados: son el
# mismo trámite de pasar de un plan, carrera o plantel a otro.
# Inscripción, reinscripción, carga y repetición de materias van con plan de
# estudios: son el avance por el plan.
_RESIDENCIA = re.compile(r"\bresiden(cia|cias|te|tes)\b")
_SERVICIO = re.compile(r"\bservicio social\b")
_TRASLADOS = re.compile(r"\b(traslad\w*|equivalencia\w*|movilidad|segunda carrera|otra carrera"
                        r"|cambi\w* de (carrera|plan|tec|tecnologico|instituto|escuela|plantel"
                        r"|campus))\b")
_BAJAS = re.compile(r"\bbajas?\b")
# "falta" sola no: "¿qué me falta para terminar?" no es un permiso.
_PERMISOS = re.compile(r"\b(permiso\w*|justificante\w*|justific\w*|faltas|faltar a"
                       r"|inasistencia\w*|ausencia\w*)\b")
_PLAN = re.compile(r"\b(plan(es)? de estudios?|reticula|materias?|asignaturas?|especialidad\w*"
                   r"|creditos?|semestres?|carga academica|reinscri\w*|inscri\w*|optativas?"
                   r"|curso (especial|de verano|de repeticion)|verano|repeticion"
                   rf"|{_REPROBAR})\b")
TOPIC_RULES: list[tuple[str, tuple[re.Pattern[str], ...]]] = [
    ("titulacion", (_TITULACION, re.compile(r"\b(egel|ceneval|tesis|tesina)\b"))),
    ("residencia", (_RESIDENCIA,)),
    ("servicio_social", (_SERVICIO,)),
    ("actividades", (_ACTIVIDADES, re.compile(r"\bcreditos? complementarios?\b"))),
    ("traslados", (_CONVALIDACION, _TRASLADOS)),
    ("bajas", (_BAJAS,)),
    ("permisos", (_PERMISOS,)),
    ("plan_estudios", (_REPETICION, _PLAN)),
]


def _topic_of(text: str) -> str | None:
    norm = guards.normalize(text)
    for topic, patterns in TOPIC_RULES:
        if any(p.search(norm) for p in patterns):
            return topic
    return None


def classify_topic(question: str, history=()) -> str:
    """Tema de la consulta (clave de TOPIC_LABELS). Si la pregunta no nombra
    tema y es de seguimiento ("¿Y cuántas horas son?"), el tema es el de las
    preguntas previas del estudiante, igual que hace search_query para buscar.
    Si nombra uno, gana aunque sea de seguimiento: "¿Y la residencia?" después
    de una pregunta de servicio social es de residencia. En una de seguimiento
    no cuentan las palabras genéricas: "¿Y cuántos créditos vale?" sigue siendo
    de servicio social, aunque "créditos" sola sería de plan de estudios."""
    if not is_follow_up(question):
        return _topic_of(question) or "otros"
    own = " ".join(w for w in re.findall(r"[a-z0-9]+", guards.normalize(question))
                   if w not in _GENERICAS)
    topic = (_topic_of(own)
             or next(filter(None, map(_topic_of, _previous_questions(history))), None)
             or _topic_of(question))
    return topic or "otros"


def topic_label(topic: str | None) -> str:
    return TOPIC_LABELS.get(topic, topic) if topic else UNCLASSIFIED_LABEL
