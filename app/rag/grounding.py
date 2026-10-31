"""Revisión por afirmación (RF-RN-02): cada oración de la respuesta debe
estar respaldada por los fragmentos, sin llamar otra vez al modelo.

La validación de la respuesta completa (RagPipeline.validate) mide cobertura
de palabras en conjunto, y una respuesta mayormente correcta con un añadido
la pasaba. Caso real (baja temporal fuera de plazo, Manual pág. 46): el
documento dice que el estudiante "debe manifestar su determinación por escrito
al Comité Académico", y el modelo agregó que el Comité "evalúa y notifica a
Servicios Escolares" (eso lo hace el Jefe(a) de División en el trámite dentro
de plazo) y que "el resto del trámite sigue los mismos pasos".

Revisiones, por oración:
  1. cobertura de términos clave contra el CONTEXTO (y la pregunta);
  2. actor + acción: si la oración dice que una persona u órgano hace algo
     ("el Comité Académico evaluará", "..., quien notifica"), en alguna
     oración del documento deben aparecer ese actor y una acción del mismo
     tipo;
  3. frases que unen procedimientos ("mismos pasos", "el resto del trámite")
     o completan con lo habitual ("normalmente", "por lo general") y que el
     CONTEXTO no dice.

El texto de los PDF trae palabras partidas ("Es tudios", "parti r"), así que
del lado del documento se busca sobre el texto sin espacios.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.domain import Hit
from app.rag import guards

# Oraciones con menos términos clave que esto no se miden por cobertura
# (encabezados como "Para la baja temporal debes:").
CLAIM_MIN_WORDS = 4
# Fracción mínima de términos clave de UNA oración que deben estar en el CONTEXTO.
CLAIM_COVERAGE_MIN = 0.5
# Si se quitan más afirmaciones que esta fracción, la respuesta ya no es
# confiable en conjunto y se deriva en vez de entregar lo que queda.
MAX_REMOVED_FRACTION = 0.5

# Personas u órganos que aparecen en los trámites. Del lado de la respuesta se
# buscan con la expresión; del lado del documento, sin espacios (clave).
ACTORS = [
    ("Comité Académico", r"comite\s+academico", "comiteacademico"),
    ("Comité", r"\bcomite\b", "comite"),  # "del Comité, que decidirá..."
    ("Servicios Escolares", r"servicios\s+escolares", "serviciosescolares"),
    ("División de Estudios Profesionales", r"division(\s+de\s+estudios\s+profesionales)?\b",
     "division"),
    ("Coordinación", r"coordinador(a|es)?\b|coordinacion\b", "coordina"),
    ("Departamento Académico", r"departamento\s+academico", "departamentoacademico"),
    ("Subdirección Académica", r"subdirec(cion|tor|tora)\s+academic[oa]", "subdirec"),
    ("Dirección del plantel", r"\bdirector(a)?\b|\bdireccion\s+del\s+(instituto|plantel)",
     "director"),
    ("Academia", r"\bacademias?\b", "academia"),
    ("Asesor", r"\basesor(a|es)?\b", "asesor"),
    ("Docente", r"\b(docente|profesor|profesora)s?\b", "docente|profesor"),
    ("Tutor", r"\btutor(a|es)?\b", "tutor"),
]
# Acciones que hace una persona u órgano (no el estudiante), por tipo: si el
# documento le atribuye al actor una del mismo tipo, cuenta como respaldo
# ("notifica" e "informa" son lo mismo para esto; "notifica" y "evalúa" no).
ACTIONS = {
    "evaluar": ("evalu", "revis", "analiz", "dictamin", "valid", "verific"),
    "decidir": ("autoriz", "aprob", "aprueb", "decid", "resuelv", "resolv", "acept",
                "rechaz", "otorg", "conced"),
    "notificar": ("notific", "inform", "comunic", "avis", "remit", "turn", "envi",
                  "canaliz"),
    "registrar": ("registr", "asign", "emit", "expid", "expide"),
}
# Forma de la acción en la respuesta: el verbo conjugado (no el sustantivo:
# "la evaluación" o "el aviso" no son una acción atribuida).
_ACTION_RE = {
    kind: re.compile(r"\b(" + "|".join(stems) + r")(a|an|ara|aran|e|en|era|eran|ira|iran)\b")
    for kind, stems in ACTIONS.items()
}
# En este tipo, "-e" sería sustantivo ("el informe", "el aviso"): solo formas
# en -a, salvo "remitir", que es en -ir.
_ACTION_RE["notificar"] = re.compile(
    r"\b(notific|inform|comunic|avis|turn|envi|canaliz)(a|an|ara|aran)\b"
    r"|\bremit(e|en|ira|iran)\b")
# Infinitivo ("evaluar", "decidir", "remitir"): solo cuenta tras "es quien" / "el encargado de"
_ACTION_INF = {kind: re.compile(r"\b(" + "|".join(stems) + r")(ar|er|ir)\b")
               for kind, stems in ACTIONS.items()}
_ACTOR_RES = [(name, re.compile(rx), key) for name, rx, key in ACTORS]
# El actor es el sujeto de la acción si la sigue de cerca ("el Comité
# Académico evaluará") o con un relativo ("..., quien notifica").
_RELATIVE = re.compile(r"(quien|quienes|el cual|la cual|los cuales|las cuales|que)\s+"
                       r"((lo|la|le|les|te|se)\s+)?$")
_MAX_GAP_WORDS = 3
# "El Comité Académico es quien recibirá la solicitud y decidirá si procede"
# (28-oct): con "es quien" el actor es el sujeto de todo lo que sigue en la oración.
_IS_WHO = re.compile(r"^(es|son|sera|seran)\s+(quien|quienes|el que|la que|los que|las que|"
                     r"el encargado|la encargada|el responsable|la responsable)\b")
# Frases que unen procedimientos distintos o completan con lo habitual.
_LINKING = re.compile(r"\bmism[oa]s?\s+(pasos|procedimientos?|tramites?|procesos?|requisitos)\b"
                      r"|\bresto\s+del\s+(tramite|procedimiento|proceso)\b"
                      r"|\bde\s+(la\s+)?misma\s+(forma|manera)\b"
                      r"|\bigual\s+que\s+(en|para)\b")
_SPECULATION = re.compile(r"\b(normalmente|generalmente|por\s+lo\s+general|usualmente|"
                          r"habitualmente|suele[n]?|seguramente|probablemente|posiblemente|"
                          r"es\s+probable|lo\s+mas\s+comun|en\s+la\s+mayoria\s+de\s+los\s+casos)\b")
# "El documento no especifica X": menciona un paso que el documento no dice.
# Se permite cuando aclara otra unidad de plazo (regla 2 del prompt).
_NOT_STATED = re.compile(r"\b(el|los|la|las)\s+(documentos?|contexto|manual|reglamento|"
                         r"lineamientos?|fragmentos?|informacion)\s+(\w+\s+)?no\s+"
                         r"(especifica|indica|menciona|detalla|establece|dice|precisa|aclara)")
_NOT_STATED_2 = re.compile(r"\bno se (menciona|especifica|indica|detalla|establece|requiere|"
                           r"requieren|piden?)\b[^.;]*\b(contexto|documentos?|manual|reglamento|"
                           r"lineamientos?|fragmentos?)\b")
# ...y sin nombrar el documento: "No se menciona otro requisito adicional" (28-oct).
_NOT_STATED_3 = re.compile(r"\bno se (menciona|mencionan|especifica|especifican|indica|indican|"
                           r"detalla|establece|establecen|requiere|requieren|pide|piden|"
                           r"exige|exigen|solicita|solicitan)\s+"
                           r"(ningun|ninguna|otro|otra|otros|otras|algun|alguna|mas|"
                           r"documentos?|requisitos?|pasos?)\b")
# "CONTEXTO" es el nombre interno de los fragmentos en el prompt: el
# estudiante no debe verlo.
_INTERNAL = re.compile(r"\bcontexto\b")
# Negar lo que el documento no niega: el 28-oct el modelo respondió "pasado ese
# plazo ya no tiene derecho a solicitarla", cuando el Manual dice que fuera de
# plazo se manifiesta al Comité Académico.
# Un "no puede" a secas suele ser un límite que el documento sí pone ("no puede
# tener más de 2 créditos", 29-oct, derivó de más): solo cuenta con "ya no".
_DENIAL = re.compile(r"\bya\s+no\s+(es\s+posible|se\s+puede|puede[sn]?|podra[sn]?|"
                     r"tiene[sn]?\s+derecho|tendra[sn]?\s+derecho|procede|aplica)\b"
                     r"|\bno\s+(es\s+posible|tiene[sn]?\s+derecho|tendra[sn]?\s+derecho|procede)\b")
_UNIT_NOTE = re.compile(r"\b(habil|habiles|natural|naturales|calendario)\b")

_STOP = set("""para como este esta esto estos estas donde cuando tiene tienen debe deben puede
pueden sobre entre desde hasta cada todo todos toda todas otro otra otros otras solo sino pero
tambien ademas segun mediante durante antes despues favor puedes deberas seran sera hacer
haber hecho ellos ellas usted ustedes debes tienes necesitas realizar caso dicho dicha""".split())
_LIST_MARK = re.compile(r"^(\s*(?:\d+[.)]|[-*•])\s+)")
# Cita de fuente: "(Manual de Lineamientos, pág. 46)" o una línea "Fuente: ...".
_SOURCE_LINE = re.compile(r"^\s*[(*_]*\s*fuente", re.IGNORECASE)
# Paréntesis que son cita ("(Manual de Lineamientos, pág. 46)"), no contenido
# como "(no naturales)": no se miden por cobertura.
_CITATION = re.compile(r"\([^)]*(\.pdf|p[aá]g|manual|reglamento|lineamiento|fuente)[^)]*\)",
                       re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")
_CLAUSE_CUT = re.compile(r",\s*(quien|quienes|el cual|la cual|los cuales|las cuales|que)\s")


@dataclass
class Claim:
    text: str            # la oración tal como está en la respuesta
    reason: str          # por qué no tiene respaldo
    cut_at: int = -1     # si >= 0, basta con quitar la oración desde aquí (cláusula "..., quien ...")
    # Cambia la conclusión de la respuesta (niega lo que el documento permite): quitar
    # la oración dejaría el resto engañoso ("No, ..."), así que no se quita, se deriva.
    severe: bool = False

    def as_dict(self) -> dict:
        return {"oracion": self.text, "motivo": self.reason}


def _nospace(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", guards.normalize(text))


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", guards.normalize(text))
            if len(w) >= 4 and w not in _STOP and not w.isdigit()]


class Context:
    """Fragmentos citados, preparados para buscar en ellos."""

    def __init__(self, hits: Sequence[Hit], question: str):
        texts = [h.chunk.text for h in hits]
        self.flat = _nospace(" ".join(texts) + " " + question)
        self.norm = " ".join(guards.normalize(" ".join(texts)).split())
        # oraciones del documento (sin espacios) para la revisión actor + acción
        self.sentences = [_nospace(s) for t in texts
                          for s in re.split(r"(?<=\.)\s+", " ".join(t.split())) if s.strip()]

    def covers(self, word: str) -> bool:
        if word in self.flat:
            return True
        return len(word) >= 7 and word[:-2] in self.flat  # notificará ~ notifica

    def attributes(self, actor_key: str, kind: str) -> bool:
        """¿Alguna oración del documento nombra al actor junto a una acción de este tipo?"""
        keys = actor_key.split("|")
        stems = [s.strip() for s in ACTIONS[kind]]
        return any(any(k in s for k in keys) and any(st in s for st in stems)
                   for s in self.sentences)


def _actor_actions(sentence: str) -> list[tuple[str, str, str, int]]:
    """(actor, clave, tipo de acción, posición del verbo) para cada acción que
    la oración atribuye a una persona u órgano."""
    s = " ".join(guards.normalize(sentence).split())
    actors = [(m.end(), name, key) for name, rx, key in _ACTOR_RES for m in rx.finditer(s)]
    found = []
    for kind in ACTIONS:
        for rx, infinitive in ((_ACTION_RE[kind], False), (_ACTION_INF[kind], True)):
            for m in rx.finditer(s):
                before = [a for a in actors if a[0] <= m.start()]
                if not before:
                    continue  # sin actor antes: suele ser lo que hace el estudiante
                end, name, key = max(before)
                gap = s[end:m.start()]
                is_who = _IS_WHO.match(gap.strip(" ,")) is not None
                if infinitive:  # "será el encargado de autorizar"; si no, lo suele hacer el estudiante
                    subject = is_who
                else:
                    # un inciso entre comas no corta la relación: "el Comité, a su vez, remitirá"
                    words = re.sub(r",[^,]*,", " ", gap).split()
                    subject = (is_who or _RELATIVE.search(gap.strip(" ,") + " ") is not None
                               or (len(words) <= _MAX_GAP_WORDS and "," not in " ".join(words)
                                   and words[:1] != ["y"]
                                   and not set(words) & {"a", "al", "ante"}))
                if subject:
                    found.append((name, key, kind, m.start()))
    return found


def _check(sentence: str, ctx: Context) -> Claim | None:
    norm = " ".join(guards.normalize(sentence).split())
    for rx, why in ((_LINKING, "une pasos de procedimientos distintos"),
                    (_SPECULATION, "completa con lo habitual o una deducción")):
        m = rx.search(norm)
        if m and m.group(0) not in ctx.norm:
            # "... pueden solicitar la baja temporal, y normalmente deben ..." (31-oct):
            # si la frase va después de una coma, basta con quitar desde ahí.
            cuts = [c.start() for c in re.finditer(r",\s", sentence)
                    if _normpos(sentence, c.start()) < m.start()]
            return Claim(sentence, f"{why}: \"{m.group(0)}\"", cuts[-1] if cuts else -1)
    if (any(rx.search(norm) for rx in (_NOT_STATED, _NOT_STATED_2, _NOT_STATED_3))
            and not _UNIT_NOTE.search(norm)):
        return Claim(sentence, "menciona algo que el documento no dice")
    m = _DENIAL.search(norm)
    if m and _phrase(m.group(0)) not in ctx.norm:
        return Claim(sentence, f"niega algo que el documento no niega: \"{m.group(0)}\"",
                     severe=True)
    if _INTERNAL.search(norm) and not _INTERNAL.search(ctx.norm):
        return Claim(sentence, "menciona el CONTEXTO interno")
    wrong = [a for a in _actor_actions(sentence) if not ctx.attributes(a[1], a[2])]
    if wrong:
        name, _key, kind, pos = min(wrong, key=lambda a: a[3])
        # Si todo lo añadido está en una cláusula "..., quien ...", basta con quitarla
        cuts = [m.start() for m in _CLAUSE_CUT.finditer(sentence)
                if _normpos(sentence, m.start()) <= pos]
        return Claim(sentence, f"atribuye a {name} una acción ({kind}) que el documento "
                               f"no le atribuye", cuts[-1] if cuts else -1)
    terms = _terms(_CITATION.sub(" ", sentence))
    # La aclaración de otra unidad de plazo (regla 2) se redacta con palabras propias
    # ("Aclaración: el documento menciona días hábiles"): no se mide por cobertura.
    if len(terms) >= CLAIM_MIN_WORDS and not _UNIT_NOTE.search(norm):
        cover = sum(ctx.covers(w) for w in terms) / len(terms)
        if cover < CLAIM_COVERAGE_MIN:
            return Claim(sentence, f"solo {cover:.0%} de sus términos aparece en el documento")
    return None


def _phrase(text: str) -> str:
    """La negación sin el "ya": basta con que el documento diga "no puede"."""
    return " ".join(re.sub(r"^ya\s+", "", text).split())


def _normpos(sentence: str, i: int) -> int:
    """Posición en el texto normalizado (sin acentos, espacios juntados) que
    corresponde a la posición i del texto original."""
    return len(" ".join(guards.normalize(sentence[:i]).split()))


def _lines(answer: str):
    """(prefijo de lista, oraciones) por renglón; None para renglones que no se
    revisan (vacíos o la cita de la fuente)."""
    for line in answer.splitlines():
        if not line.strip() or _is_source(line):
            yield line, None
            continue
        m = _LIST_MARK.match(line)
        prefix = m.group(1) if m else ""
        yield prefix, [s for s in _SENTENCE_END.split(line[len(prefix):]) if s.strip()]


def unsupported(answer: str, hits: Sequence[Hit], question: str) -> tuple[list[Claim], int]:
    """(afirmaciones sin respaldo, número de afirmaciones revisadas)."""
    ctx = Context(hits, question)
    claims, total = [], 0
    for _prefix, sentences in _lines(answer):
        for s in sentences or []:
            if _is_source(s):
                continue
            total += 1
            c = _check(s, ctx)
            if c:
                claims.append(c)
    return claims, total


def _is_source(sentence: str) -> bool:
    """¿Es la cita de la fuente? Un paréntesis solo lo es si trae página, "Fuente" o
    ".pdf", o si es corto y nombra un documento. El 30-oct pasó sin revisar
    "(El resto del procedimiento no se menciona para este caso.)" porque todo
    paréntesis contaba como fuente."""
    s = sentence.strip()
    if _SOURCE_LINE.match(s):
        return True
    m = re.fullmatch(r"[*_]*\((.*)\)[*_]*\s*\.?[*_]*", s)
    if not m:
        return False
    inner = guards.normalize(m.group(1))
    return bool(re.search(r"p[aá]g\b|p[aá]g\.|fuente|\.pdf|\bcap[ií]tulo\b", inner)) or (
        len(inner.split()) <= 12 and re.search(r"manual|reglamento|lineamiento|gu[ií]a", inner)
        is not None)


def remove(answer: str, claims: Sequence[Claim]) -> str:
    """Quita de la respuesta las afirmaciones sin respaldo (o solo la cláusula
    "..., quien ..." cuando ahí está el añadido) y renumera los pasos."""
    bad = {c.text: c for c in claims}
    out, step = [], 0
    for prefix, sentences in _lines(answer):
        if sentences is None:
            out.append(prefix)
            continue
        kept = []
        for s in sentences:
            c = bad.get(s)
            if c is None:
                kept.append(s)
            elif c.cut_at > 0:
                kept.append(s[:c.cut_at].rstrip(" ,;") + ".")
        if not re.match(r"\s*\d+[.)]", prefix):
            step = 0  # otro bloque: una lista que siga empieza de nuevo en 1
        if not kept:
            continue
        if len(kept) < len(sentences) and kept[0][:1].islower():
            kept[0] = kept[0][0].upper() + kept[0][1:]  # "2. el procedimiento..." -> "2. El..."
        if re.match(r"\s*\d+[.)]", prefix):
            step += 1
            prefix = re.sub(r"\d+", str(step), prefix, count=1)
        out.append(prefix + " ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---- condiciones y excepciones omitidas (regla 1d) -------------------------------
# Medido con el modelo real: "¿Puedo repetir la residencia si la repruebo?" respondió
# "se cursa por una única ocasión" 3 de 3 veces sin la excepción que sigue en la Guía
# ("En el caso de existir causas justificadas ... segunda asignación de proyecto"), y
# la baja fuera de plazo omitió "todo estudiante que haya cursado al menos un
# semestre" en 6 de 6. Si la respuesta omite una condición así, se pide incluirla en
# la MISMA regeneración que ya existe (no hay una llamada más).
# Dos tipos, con reglas distintas para no pedirlas donde no importan:
#  - quién puede hacer el trámite ("Todo estudiante que haya cursado al menos un
#    semestre ... tiene derecho a solicitar baja temporal"): cuenta si ese trámite
#    está en la pregunta y en la respuesta (así no se pide la de la baja parcial);
#  - excepciones ("En el caso de existir causas justificadas ..."): cuentan si están
#    junto a una oración que la respuesta usó (así "¿Cuántos créditos vale la
#    residencia?" no pide la de la segunda asignación, que está en otro párrafo).
_EXCEPTION = re.compile(r"\ben (el|los) casos? de\b|\ba excepcion de\b|\bsalvo\b|\bexcepto\b")
_ELIGIBILITY = re.compile(r"\btodos? (los )?estudiantes? que\b")
_PROCEDURE = re.compile(r"\b(?:solicitar|realizar|cursar|tramitar|presentar)\s+(?:la |el |su )?"
                        r"(\w+\s+\w+)")
# Solo remite a otro reglamento ("Lo anterior a excepción de lo que se indique en los
# lineamientos ... (ver CAPÍTULO 11)"): no dice nada que el estudiante pueda usar aquí
# y pedirlo costaba regeneraciones en casi todas las respuestas de bajas.
_CROSS_REF = re.compile(r"lo que se indique|lo previsto en|lo establecido en|\bver cap")
# La respuesta incluye la condición si tiene al menos esta fracción de sus términos
# propios (los que no vienen de la pregunta).
CONDITION_MIN_COVER = 0.34
# Una oración del documento "se usó" si la respuesta tiene esta fracción de sus términos.
USED_SENTENCE_COVER = 0.6
MAX_CONDITIONS = 2


def _stems(text: str) -> set[str]:
    return {w[:6] for w in _terms(text)}


def _doc_sentences(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [s for s in re.split(r"(?<=\.)\s+(?=[A-ZÁÉÍÓÚÑ0-9])", flat) if s.strip()]


def _clean_condition(sentence: str) -> str:
    s = re.sub(r"^.*?(\d+(\.\d+){2,}\s+)", "", sentence)  # quita "5.4.6.4 " y lo anterior
    s = s.split("•")[0]  # la Guía sigue con otra viñeta en la misma "oración"
    return re.sub(r"\s+", " ", s).strip()


def _condition_text(sentence: str, norm: str) -> str:
    """Lo que se le pide incluir al modelo. De quién puede hacer el trámite, solo
    hasta el trámite: la oración del Manual sigue con "dentro de los 20 días
    hábiles", y en la baja fuera de plazo el modelo la copiaba entera o escribía
    "normalmente deben hacerlo dentro de los 20 días hábiles" (31-oct)."""
    text = _clean_condition(sentence)
    if _ELIGIBILITY.search(norm):
        m = re.search(r"^.*?\b(?:solicitar|realizar|cursar|tramitar|presentar)\s+"
                      r"(?:la |el |su )?\S+\s+\S+", text, re.IGNORECASE)
        if m:
            return m.group(0).rstrip(" ,;")
    return text


def _relevant_condition(norm: str, i: int, used: list[int], answer: str, question: str) -> bool:
    if _ELIGIBILITY.search(norm):
        m = _PROCEDURE.search(norm)
        if m:  # el trámite al que da derecho, contra pregunta y respuesta (sin espacios:
            key = _nospace(m.group(1))  # el PDF parte palabras, "baj a")
            return key in _nospace(question) and key in _nospace(answer)
    elif not _EXCEPTION.search(norm):
        return False
    return any(abs(i - j) <= 1 for j in used)


def omitted_conditions(answer: str, hits: Sequence[Hit], question: str) -> list[str]:
    """Condiciones o excepciones de los fragmentos que la respuesta omite y que
    importan para la pregunta (ver _EXCEPTION y _ELIGIBILITY)."""
    a_stems, q_stems = _stems(answer), _stems(question)
    found: list[tuple[str, set[str]]] = []
    for h in hits:
        sentences = _doc_sentences(h.chunk.text)
        used = []
        for i, s in enumerate(sentences):
            st = _stems(s)
            if len(st) >= 3 and len(st & a_stems) / len(st) >= USED_SENTENCE_COVER:
                used.append(i)
        for i, s in enumerate(sentences):
            norm = " ".join(guards.normalize(s).split())
            if _CROSS_REF.search(norm) or not _relevant_condition(norm, i, used, answer,
                                                                  question):
                continue
            # Qué debe aparecer en la respuesta: en "Todo estudiante que haya cursado al
            # menos un semestre en el Instituto, tiene derecho a ..." basta la condición
            # (hasta la coma); en una excepción, la oración completa.
            core = s
            if _ELIGIBILITY.search(norm):
                m = re.search(r"\bque\b(.*?),", s, re.IGNORECASE)
                core = m.group(1) if m else s
            own = _stems(core) - q_stems - {"haya", "hayan"}
            if not own or len(own & a_stems) / len(own) >= CONDITION_MIN_COVER:
                continue  # sin términos propios, o la respuesta ya la incluye
            # el mismo lineamiento suele venir en dos documentos (Manual y Lineamiento)
            if any(len(own & k) / len(own | k) >= 0.5 for _, k in found):
                continue
            found.append((_condition_text(s, norm), own))
    return [s for s, _ in found[:MAX_CONDITIONS]]
