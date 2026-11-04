"""Evaluación de exactitud con preguntas de estudiantes (data/eval_exactitud.jsonl).

Cada caso trae una pregunta escrita como la haría un estudiante, la respuesta
esperada sacada de los PDF de data/docs (cita textual, documento y página del
PDF), los hechos que la respuesta debe incluir y los que no debe incluir, y si
el bot debe responder o derivar. Todos están "pendiente de validación por la
coordinadora" hasta que ella los revise.

Comandos (solo `--correr` llama al modelo real y gasta cuota de Groq):

    python -m app.eval_exactitud --verificar        # citas contra los PDF y hechos contra la respuesta esperada
    python -m app.eval_exactitud --estimar          # tokens que gastaría con Groq, sin llamar al modelo
    python -m app.eval_exactitud --revision docs/eval_exactitud_revision.docx   # archivo para la coordinadora
    python -m app.eval_exactitud --correr --si-gastar --env C:/ruta/.env --kb C:/ruta/data/kb \
        --out data/eval_exactitud_FECHA.jsonl
    python -m app.eval_exactitud --resumen data/eval_exactitud_FECHA.jsonl

Clasificación de cada respuesta (ver `classify`):
  acierto              debía responder, respondió, trae todos los hechos y ningún añadido
  incompleta           debía responder y respondió sin añadidos, pero le falta algún hecho (dudoso)
  error                respondió algo que el documento no dice (un "no incluye"), o respondió
                       una consulta que debía derivar
  derivacion_correcta  debía derivar y derivó (si el motivo no es el esperado, queda como dudoso)
  derivo_de_mas        debía responder y derivó (no es un error: no dio datos falsos)
  inconcluso           el proveedor no respondió (falla o alta demanda): repetir
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "eval_exactitud.jsonl"
DOCS = ROOT / "data" / "docs"
HISTORY_GLOB = "fundamentacion_*.jsonl"   # consultas reales ya medidas con Groq (tokens)
PENDING = "pendiente de validación por la coordinadora"
ANSWER, DERIVE = "responder", "derivar"
UNAVAILABLE = {"llm_unavailable", "high_demand"}
# Plan gratuito de Groq (docs/capacidad-y-costos.md): 200,000 tokens por día y
# ~3 consultas por minuto (8,000 tokens/minuto).
GROQ_TPD = 200_000
SECONDS_PER_CALL = 22.0
MAX_CALLS = 3  # 1 generación + a lo más 2 regeneraciones (validación / NO_INFO)


# ---- casos ------------------------------------------------------------------------------
def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _norm(text: str) -> str:
    from app.grounding_eval import _norm as norm
    return norm(text)


def check_facts(text: str, facts: dict | None) -> tuple[list[str], list[str]]:
    from app.grounding_eval import check_facts as check
    return check(text, facts)


def _squash(text: str) -> str:
    """Para comparar citas con el texto del PDF: pypdf mete espacios dentro de
    las palabras ("Soc ial"), así que se comparan sin espacios ni signos."""
    return re.sub(r"[^a-z0-9]", "", _norm(text))


def _page_text(doc: str, cache: dict) -> list[str]:
    if doc not in cache:
        from app.rag.loaders import load_sections
        cache[doc] = [t for _, t in load_sections(DOCS / doc)]
    return cache[doc]


def verify(cases: list[dict], docs_dir: Path | None = None) -> list[str]:
    """Problemas del conjunto, sin llamar al modelo:
    - cada fragmento de la cita (separados por "[...]") está en la página citada
      del PDF (o en la siguiente, si la cita cruza de página);
    - las expresiones regulares compilan;
    - la respuesta esperada cumple sus propios hechos (si no, el hecho está mal escrito);
    - todos siguen pendientes de validación o ya validados, con su campo."""
    global DOCS
    if docs_dir is not None:
        DOCS = docs_dir
    problems, cache = [], {}
    for c in cases:
        cid = c["id"]
        for key in ("id", "tema", "q", "esperado", "respuesta_esperada", "hechos", "validacion"):
            if key not in c:
                problems.append(f"{cid}: falta el campo {key}")
        valid = {}
        for kind in ("incluye", "no_incluye"):
            valid[kind] = []
            for p in c["hechos"].get(kind, []):
                try:
                    re.compile(p)
                    valid[kind].append(p)
                except re.error as e:
                    problems.append(f"{cid}: regex inválida en {kind}: {p} ({e})")
        if c["esperado"] == DERIVE:
            if not c.get("motivo_esperado"):
                problems.append(f"{cid}: un caso a derivar necesita motivo_esperado")
            continue
        missing, added = check_facts(c["respuesta_esperada"], valid)  # solo las que compilan
        if missing:
            problems.append(f"{cid}: la respuesta esperada no cumple sus hechos: {missing}")
        if added:
            problems.append(f"{cid}: la respuesta esperada choca con 'no incluye': {added}")
        if not (DOCS / c["documento"]).exists():
            problems.append(f"{cid}: no existe el documento {c['documento']}")
            continue
        pages = _page_text(c["documento"], cache)
        n = c["pagina"]
        if not 1 <= n <= len(pages):
            problems.append(f"{cid}: la página {n} no existe en {c['documento']}")
            continue
        where = _squash(" ".join(pages[n - 1:n + 1]))
        for frag in re.split(r"\[\.\.\.\]|\[…\]", c["cita"]):
            if len(_squash(frag)) >= 12 and _squash(frag) not in where:
                problems.append(f"{cid}: la cita no aparece en {c['documento']} pág. {n}: "
                                f"\"{frag.strip()[:70]}...\"")
    return problems


# ---- clasificación ------------------------------------------------------------------------
@dataclass
class Verdict:
    category: str
    doubtful: bool
    why: str


def _cites_document(text: str, doc: str) -> bool:
    from app.rag.pipeline import _cites_source
    from app.domain import Chunk, Hit
    return _cites_source(text, [Hit(Chunk("x", doc, 0, ""), 1.0)])


def classify(case: dict, outcome: str, reason: str, text: str) -> Verdict:
    if reason in UNAVAILABLE:
        return Verdict("inconcluso", True, f"el proveedor no respondió ({reason}): repetir")
    if case["esperado"] == DERIVE:
        if outcome == "derived":
            ok = reason in case.get("motivo_esperado", [])
            return Verdict("derivacion_correcta", not ok,
                           "" if ok else f"derivó por {reason}, se esperaba "
                                         f"{'/'.join(case['motivo_esperado'])}")
        _, added = check_facts(text, case["hechos"])
        return Verdict("error", True, "respondió una consulta que debía derivar"
                       + (f"; además dice {added}" if added else ""))
    if outcome == "derived":
        return Verdict("derivo_de_mas", True, f"derivó ({reason}) una consulta con respuesta "
                                              "en los documentos")
    missing, added = check_facts(text, case["hechos"])
    if added:
        return Verdict("error", True, f"dice algo que el documento no dice: {added}")
    if missing:
        return Verdict("incompleta", True, f"le faltan hechos: {missing}")
    if case.get("documento") and not _cites_document(text, case["documento"]):
        return Verdict("acierto", True, f"no cita {case['documento']} (revisar la fuente)")
    return Verdict("acierto", False, "")


def summarize(rows: list[dict]) -> dict:
    """Porcentajes del reporte. Acierto sobre los casos a responder, derivación
    correcta sobre los casos a derivar, errores sobre todos los evaluados; los
    inconclusos no cuentan en ninguno (hay que repetirlos)."""
    done = [r for r in rows if r["categoria"] != "inconcluso"]
    ans = [r for r in done if r["esperado"] == ANSWER]
    der = [r for r in done if r["esperado"] == DERIVE]

    def pct(n, d):
        return round(100 * n / d, 1) if d else 0.0

    count = {k: sum(r["categoria"] == k for r in rows) for k in
             ("acierto", "incompleta", "error", "derivacion_correcta", "derivo_de_mas", "inconcluso")}
    return {
        "evaluados": len(done), "inconclusos": count["inconcluso"],
        "acierto_pct": pct(count["acierto"], len(ans)),
        "derivacion_correcta_pct": pct(count["derivacion_correcta"], len(der)),
        "error_pct": pct(count["error"], len(done)),
        "incompletas_pct": pct(count["incompleta"], len(ans)),
        "derivo_de_mas_pct": pct(count["derivo_de_mas"], len(ans)),
        "conteo": count,
        "dudosos": [r["id"] for r in rows if r["dudoso"]],
        "tokens": sum(r.get("tokens", 0) for r in rows),
    }


# ---- estimación de tokens (sin llamar al modelo) ---------------------------------------------
def _prompt_chars(pipe, question: str) -> int:
    """Caracteres del prompt que el pipeline mandaría en la 1a llamada; 0 si no
    llegaría al modelo (fuera de alcance, inyección o sin fragmentos sobre el umbral)."""
    from app.rag import guards
    q = guards.sanitize(question, pipe.cfg.max_question_chars)
    if guards.looks_like_injection(q) or guards.out_of_scope_category(q):
        return 0
    hits = [h for h in pipe.retrieve(q) if h.score >= pipe.cfg.sim_threshold]
    if not hits:
        return 0
    return sum(len(m["content"]) for m in pipe._build_messages(q, hits, []))


def fit_tokens(samples: list[tuple[int, int]]) -> tuple[float, float]:
    """Mínimos cuadrados: tokens de una llamada ≈ a * caracteres del prompt + b
    (b absorbe la respuesta y el razonamiento del modelo)."""
    xs, ys = [s[0] for s in samples], [s[1] for s in samples]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in samples) / var if var else 0.0
    return a, my - a * mx


def estimate(cases: list[dict], pipe, history: list[dict]) -> dict:
    """Tokens estimados para correr el conjunto una vez. Se calibra con las
    consultas que ya se midieron con Groq (data/fundamentacion_*.jsonl):
    - las de UNA llamada dan tokens por carácter de prompt (regresión lineal);
    - todas juntas dan cuántos tokens de más ponen las regeneraciones (factor)."""
    one = [(_prompt_chars(pipe, r["q"]), r["tokens"]) for r in history
           if r.get("llamadas_llm") == 1 and r.get("tokens")]
    one = [s for s in one if s[0] > 0]
    a, b = fit_tokens(one)
    with_calls = [(r, _prompt_chars(pipe, r["q"])) for r in history
                  if r.get("llamadas_llm") and r.get("tokens")]
    with_calls = [(r, ch) for r, ch in with_calls if ch > 0]
    factor = (sum(r["tokens"] for r, _ in with_calls)
              / sum(a * ch + b for _, ch in with_calls)) if with_calls else 1.0
    calls_avg = statistics.mean(r["llamadas_llm"] for r, _ in with_calls) if with_calls else 1.0
    per_case = []
    for c in cases:
        chars = _prompt_chars(pipe, c["q"])
        first = round(a * chars + b) if chars else 0
        per_case.append({"id": c["id"], "esperado": c["esperado"], "chars_prompt": chars,
                         "tokens_1_llamada": first, "tokens_esperados": round(first * factor),
                         "llega_al_modelo": bool(chars)})
    reaching = [p for p in per_case if p["llega_al_modelo"]]
    minimum = sum(p["tokens_1_llamada"] for p in per_case)
    expected = sum(p["tokens_esperados"] for p in per_case)
    # Peor caso: cada consulta hace 3 llamadas y cada regeneración repite el
    # prompt más la respuesta anterior (~1.15 veces la primera)
    worst = round(sum(p["tokens_1_llamada"] * (1 + 2 * 1.15) for p in per_case))
    calls = len(reaching) * calls_avg
    return {
        "calibracion": {"consultas_una_llamada": len(one), "tokens_por_caracter": round(a, 4),
                        "tokens_fijos": round(b), "factor_regeneraciones": round(factor, 3),
                        "llamadas_promedio": round(calls_avg, 2)},
        "casos": len(cases), "llegan_al_modelo": len(reaching),
        "tokens_minimos": minimum, "tokens_esperados": expected, "tokens_peor_caso": worst,
        "pct_tope_diario_esperado": round(100 * expected / GROQ_TPD, 1),
        "pct_tope_diario_peor": round(100 * worst / GROQ_TPD, 1),
        "llamadas_esperadas": round(calls), "minutos_esperados": round(calls * SECONDS_PER_CALL / 60),
        "por_caso": per_case,
    }


def load_history(data_dir: Path = ROOT / "data") -> list[dict]:
    rows = []
    for f in sorted(data_dir.glob(HISTORY_GLOB)):
        rows += [json.loads(l) for l in f.read_text("utf-8").splitlines() if l.strip()]
    return rows


def local_pipeline(kb: Path):
    """Pipeline con búsqueda real y SIN modelo: solo arma prompts para contar."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.config import Settings
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.pipeline import RagPipeline
    from app.runtime import build_embedder

    class NoLLM:
        def chat(self, *a, **k):
            raise RuntimeError("la estimación no debe llamar al modelo")

    cfg = Settings.from_env(load_dotenv_file=False)
    cfg = Settings(**{**cfg.__dict__, "kb_dir": kb, "database_url": ""})
    emb = build_embedder(cfg)
    return RagPipeline(cfg, emb, KnowledgeBase.open(cfg, emb).store, NoLLM())


# ---- correr contra el bot (gasta cuota) ------------------------------------------------------
def run(cases: list[dict], env: Path, kb: Path, out: Path, token_cap: int = 0) -> int:
    """Corre los casos con el pipeline completo (búsqueda real + Groq, sin base de
    datos) y guarda cada resultado en cuanto llega; si se corta, al volver a
    correrlo sigue con los que faltan. Se detiene si Groq avisa del tope diario o
    si lo gastado en esta corrida llega a token_cap (0 = sin tope): la cuota diaria
    de Groq es la misma que usa el bot con los estudiantes."""
    from app.grounding_eval import RETRY_WAITS, build_pipeline

    pipe, meter = build_pipeline(env, kb)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.read_text("utf-8").splitlines() if l.strip()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for c in cases:
            if c["id"] in done:
                continue
            if token_cap and meter["tokens"] >= token_cap:
                print(f"Se llegó al tope de {token_cap:,} tokens; se detiene. Para seguir, "
                      "vuelve a correr el mismo comando otro día.")
                return 4
            retries = 0
            while True:
                c0, t0 = meter["calls"], meter["tokens"]
                start = time.perf_counter()
                r = pipe.answer(c["q"])
                seconds = time.perf_counter() - start
                calls, tokens = meter["calls"] - c0, meter["tokens"] - t0
                if calls:
                    time.sleep(SECONDS_PER_CALL * calls)  # ~3 llamadas/minuto
                if meter["daily_limit"]:
                    print(f"Groq avisó del tope diario; se detiene. {meter['daily_limit']}")
                    return 3
                if r.reason.value in UNAVAILABLE and retries < len(RETRY_WAITS):
                    time.sleep(RETRY_WAITS[retries])
                    retries += 1
                    continue
                break
            v = classify(c, r.outcome.value, r.reason.value, r.text)
            missing, added = (check_facts(r.text, c["hechos"]) if r.outcome.value == "answered"
                              else ([], []))
            rec = {"id": c["id"], "tema": c["tema"], "q": c["q"], "esperado": c["esperado"],
                   "resultado": r.outcome.value, "motivo": r.reason.value, "texto": r.text,
                   "categoria": v.category, "dudoso": v.doubtful, "por_que": v.why,
                   "faltan": missing, "anadidos": added, "llamadas_llm": calls, "tokens": tokens,
                   "segundos": round(seconds, 2), "proveedor": r.provider, "modelo": r.model,
                   "fragmentos": [f"{h.chunk.doc_name} {h.chunk.section}" for h in r.hits],
                   "fecha": time.strftime("%Y-%m-%d %H:%M:%S")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{c['id']:8} {v.category:20} {'DUDOSO ' if v.doubtful else ''}{v.why}", flush=True)
    return 0


def report(rows: list[dict], cases: list[dict], review_csv: Path | None = None) -> str:
    s = summarize(rows)
    by_id = {c["id"]: c for c in cases}
    lines = [f"Casos evaluados: {s['evaluados']} (inconclusos, a repetir: {s['inconclusos']})",
             f"Acierto (de los que debía responder):            {s['acierto_pct']}%",
             f"Derivación correcta (de los que debía derivar):  {s['derivacion_correcta_pct']}%",
             f"Errores (respuesta falsa o no derivó; de todos): {s['error_pct']}%",
             f"Incompletas (sin añadidos, falta algún hecho):    {s['incompletas_pct']}%",
             f"Derivó de más (tenía respuesta):                 {s['derivo_de_mas_pct']}%",
             f"Tokens gastados: {s['tokens']:,}",
             f"Para revisar a mano ({len(s['dudosos'])}): {', '.join(s['dudosos']) or 'ninguno'}"]
    if review_csv:
        with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "Categoría", "Dudoso", "Por qué", "Pregunta", "Respuesta del bot",
                        "Respuesta esperada", "Cita", "Documento", "Página", "Revisión manual"])
            for r in rows:
                c = by_id.get(r["id"], {})
                w.writerow([r["id"], r["categoria"], "sí" if r["dudoso"] else "", r["por_que"],
                            r["q"], r["texto"], c.get("respuesta_esperada", ""), c.get("cita", ""),
                            c.get("documento", ""), c.get("pagina") or "", ""])
        lines.append(f"Hoja para la revisión manual: {review_csv}")
    return "\n".join(lines)


# ---- archivo para la coordinadora -------------------------------------------------------------
TOPIC_NAMES = {"plan_estudios": "Plan de estudios", "residencia": "Residencia profesional",
               "servicio_social": "Servicio social", "titulacion": "Titulación", "bajas": "Bajas",
               "traslados": "Traslados y convalidación", "actividades": "Actividades complementarias",
               "permisos": "Permisos y evaluaciones", "otros": "Otros"}


def _doc_name(doc: str) -> str:
    return " ".join(re.sub(r"[_\-]+", " ", re.sub(r"\.pdf$", "", doc)).split())


def review_rows(cases: list[dict]) -> list[list[str]]:
    rows = []
    for n, c in enumerate(cases, 1):
        if c["esperado"] == DERIVE:
            expected = f"DEBE DERIVAR. {c['respuesta_esperada']} ({c['nota']})"
            source = "—"
        else:
            expected = c["respuesta_esperada"]
            if c.get("nota"):
                expected += f"\nNota: {c['nota']}"
            source = f"{_doc_name(c['documento'])}, pág. {c['pagina']} (numeral {c['numeral']})"
        rows.append([str(n), TOPIC_NAMES.get(c["tema"], c["tema"]), c["q"], expected,
                     c.get("cita", ""), source])
    return rows


REVIEW_HEADER = ["#", "Tema", "Pregunta (como la escribiría un estudiante)",
                 "Lo que el bot debe responder", "Cita textual del documento",
                 "Documento y página del PDF", "¿Correcta o corregir?", "Comentarios"]


def write_review(cases: list[dict], out: Path) -> list[Path]:
    """Hoja para la coordinadora: un .docx para leer o imprimir y un .csv para
    Excel, ambos con la columna "¿Correcta o corregir?" vacía para que la llene."""
    rows = review_rows(cases)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(REVIEW_HEADER)
        for r in rows:
            w.writerow(r + ["", ""])
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Cm, Pt

    doc = Document()
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(1.5))
    style = doc.styles["Normal"]
    style.font.name, style.font.size = "Calibri", Pt(9)
    doc.add_heading("Conjunto de evaluación de exactitud del chatbot", level=1)
    doc.add_paragraph(
        f"{len(cases)} preguntas escritas como las haría un estudiante. Para cada una, la respuesta "
        "que el chatbot debería dar según los documentos oficiales cargados, con la cita textual y "
        "la página del PDF. Todas están pendientes de su validación.")
    doc.add_paragraph(
        "Por favor marque en la columna \"¿Correcta o corregir?\" una de las dos opciones. Si es "
        "\"corregir\", anote en \"Comentarios\" qué debe decir. Las filas que dicen \"DEBE "
        "DERIVAR\" son consultas que el chatbot no debe contestar (datos personales o temas que "
        "no están en los documentos) y debe mandar a la instancia correspondiente.")
    table = doc.add_table(rows=1, cols=len(REVIEW_HEADER))
    table.style = "Table Grid"
    # Carta horizontal con márgenes de 1.5 cm: 24.9 cm útiles
    widths = [Cm(0.7), Cm(2.0), Cm(3.8), Cm(5.0), Cm(5.8), Cm(2.8), Cm(2.3), Cm(2.5)]
    for cell, text, w in zip(table.rows[0].cells, REVIEW_HEADER, widths):
        cell.text, cell.width = text, w
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
    for r in rows:
        cells = table.add_row().cells
        for cell, text, w in zip(cells, r + ["☐ Correcta\n☐ Corregir", ""], widths):
            cell.text, cell.width = text, w
    doc.save(out)
    return [out, csv_path]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--casos", type=Path, default=CASES)
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--estimar", action="store_true")
    ap.add_argument("--revision", type=Path)
    ap.add_argument("--correr", action="store_true")
    ap.add_argument("--si-gastar", action="store_true",
                    help="confirma que se va a gastar cuota de Groq (requerido con --correr)")
    ap.add_argument("--env", type=Path, default=ROOT / ".env")
    ap.add_argument("--kb", type=Path, default=ROOT / "data" / "kb")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tope-tokens", type=int, default=90_000,
                    help="--correr se detiene al gastar estos tokens (0 = sin tope). Por defecto "
                         "90,000: deja más de la mitad de la cuota diaria para el bot")
    ap.add_argument("--resumen", type=Path)
    a = ap.parse_args(argv)
    cases = load_cases(a.casos)
    if a.verificar:
        problems = verify(cases)
        print("\n".join(problems) if problems else f"{len(cases)} casos sin problemas: las citas "
              "están en sus páginas y las respuestas esperadas cumplen sus hechos.")
        return 1 if problems else 0
    if a.estimar:
        est = estimate(cases, local_pipeline(a.kb), load_history())
        print(json.dumps({k: v for k, v in est.items() if k != "por_caso"}, ensure_ascii=False,
                         indent=2))
        return 0
    if a.revision:
        for p in write_review(cases, a.revision):
            print(f"Escrito: {p}")
        return 0
    if a.correr:
        if not a.si_gastar or not a.out:
            print("--correr gasta cuota de Groq: agrega --si-gastar y --out ARCHIVO.jsonl")
            return 2
        return run(cases, a.env, a.kb, a.out, a.tope_tokens)
    if a.resumen:
        rows = [json.loads(l) for l in a.resumen.read_text("utf-8").splitlines() if l.strip()]
        print(report(rows, cases, a.resumen.with_name(a.resumen.stem + "_revision.csv")))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    logging_off = __import__("logging")
    logging_off.basicConfig(level=logging_off.WARNING)
    sys.exit(main())
