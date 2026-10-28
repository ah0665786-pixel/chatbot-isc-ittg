"""Evaluación de fundamentación con el modelo REAL (RF-RN-02).

Corre consultas de data/retrieval_cases.jsonl contra el pipeline completo
(búsqueda real + LLM real) y revisa cada respuesta contra los "hechos" del
caso: lo que debe incluir y lo que no debe incluir (añadidos que el documento
no dice, como "el Comité Académico notifica a Servicios Escolares").

No escribe en ninguna base de datos: el pipeline se arma sin repositorio.
Usa solo el primer proveedor del .env (Groq), sin respaldo, para que ANTES y
DESPUÉS se comparen con el mismo modelo. El plan gratuito de Groq permite
~8,000 tokens por minuto y tiene tope diario, así que espera entre llamadas,
guarda cada resultado en cuanto llega (se puede reanudar) y se detiene si
Groq avisa del tope diario.

Solo importa módulos que ya existían antes de la validación por afirmación,
para poder correrlo también con el código de main (medición ANTES):

    python -m app.grounding_eval --env C:/ruta/.env --kb C:/ruta/data/kb \
        --out data/fundamentacion_antes_FECHA.jsonl --bajas 10 --respondibles
    python -m app.grounding_eval --resumen data/fundamentacion_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "data" / "retrieval_cases.jsonl"
SECONDS_PER_CALL = 22.0   # ~3 llamadas por minuto: cabe en 8,000 tokens/minuto
RETRY_WAITS = (60.0, 90.0, 120.0)  # proveedor en su límite por minuto: esperar y repetir
# Corridas de las demás consultas de bajas que van antes de las preguntas con
# respuesta; las que faltan para llegar a --bajas van al final del plan.
FIRST_RUNS = 3

log = logging.getLogger("grounding_eval")


def _norm(text: str) -> str:
    from app.rag import guards
    return " ".join(guards.normalize(text).split())


def check_facts(text: str, facts: dict | None) -> tuple[list[str], list[str]]:
    """(hechos que faltan, añadidos encontrados). Cada hecho es una expresión
    regular sobre el texto en minúsculas, sin acentos y con espacios juntados."""
    if not facts:
        return [], []
    t = _norm(text)
    missing = [p for p in facts.get("incluye", []) if not re.search(p, t)]
    added = [p for p in facts.get("no_incluye", []) if re.search(p, t)]
    return missing, added


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def _plan(cases: list[dict], bajas: int, respondibles: bool, only: str | None) -> list[tuple[str, int, dict]]:
    """Lista de (grupo, corrida, caso), por prioridad: todas las corridas de la
    consulta del caso real, 3 de cada una de las demás de bajas, las preguntas
    con respuesta y al final el resto de corridas de bajas. Si el tope diario
    corta la evaluación, lo más importante ya quedó medido."""
    plan, rest = [], []
    if bajas:
        tema = [c for c in cases if c.get("tema") == "bajas" and c.get("answerable", True)]
        main = [c for c in tema if "naturales" in c["q"]]  # la consulta del caso real
        others = [c for c in tema if c not in main]
        plan += [("bajas", run, c) for c in main for run in range(1, bajas + 1)]
        plan += [("bajas", run, c) for run in range(1, min(bajas, FIRST_RUNS) + 1) for c in others]
        rest = [("bajas", run, c) for run in range(FIRST_RUNS + 1, bajas + 1) for c in others]
    if respondibles:
        plan += [("respondibles", 1, c) for c in cases if c.get("answerable", True)]
    plan += rest
    if only:
        plan = [p for p in plan if only in p[2]["q"]]
    return plan


def build_pipeline(env: Path, kb: Path):
    from dotenv import load_dotenv
    load_dotenv(env, override=True)  # las claves quedan en el entorno; nunca se imprimen
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.adapters.llm_openai_compat import OpenAICompatLLM, build_llm
    from app.config import Settings
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.pipeline import RagPipeline
    from app.runtime import build_embedder

    cfg = Settings.from_env(load_dotenv_file=False)
    cfg = Settings(**{**cfg.__dict__, "kb_dir": kb, "database_url": ""})
    first = cfg.llm_endpoints[0]
    llm = build_llm([first], cfg.llm_timeout, cfg.llm_max_retries, cfg.llm_total_budget)
    meter = {"calls": 0, "tokens": 0, "daily_limit": "", "last_429": ""}
    provider: OpenAICompatLLM = llm.providers[0]
    extract = provider._extract

    def counted_extract(r):
        meter["calls"] += 1
        meter["tokens"] += int((r.json().get("usage") or {}).get("total_tokens") or 0)
        return extract(r)

    def on_response(r):
        if r.status_code == 429:
            r.read()
            meter["calls"] += 1
            meter["last_429"] = r.text[:300]
            if "per day" in r.text.lower() or "tpd" in r.text.lower() or "rpd" in r.text.lower():
                meter["daily_limit"] = r.text[:300]

    provider._extract = counted_extract
    provider._client.event_hooks["response"].append(on_response)
    emb = build_embedder(cfg)
    pipe = RagPipeline(cfg, emb, KnowledgeBase.open(cfg, emb).store, llm)
    return pipe, meter


def _record(group: str, run: int, case: dict, r, calls: int, tokens: int, retries: int) -> dict:
    missing, added = check_facts(r.text, case.get("hechos")) if r.outcome.value == "answered" else ([], [])
    return {
        "grupo": group, "corrida": run, "q": case["q"], "tema": case.get("tema"),
        "resultado": r.outcome.value, "motivo": r.reason.value, "detalle": r.detail,
        "texto": r.text, "intentos": r.attempts, "reintentos_sin_info": getattr(r, "no_info_retries", 0),
        "no_usadas": list(r.debug),
        "afirmaciones_sin_respaldo": list(getattr(r, "unsupported_claims", []) or []),
        "proveedor": r.provider, "modelo": r.model, "similitud_max": round(r.max_score, 3),
        "fragmentos": [f"{h.chunk.doc_name} #{h.chunk.chunk_no}" for h in r.hits],
        "llamadas_llm": calls, "tokens": tokens, "repeticiones": retries,
        "faltan": missing, "anadidos": added, "con_hechos": bool(case.get("hechos")),
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def run(args) -> int:
    cases = load_cases(Path(args.cases))
    plan = _plan(cases, args.bajas, args.respondibles, args.solo)
    out = Path(args.out)
    done = set()
    if out.exists():
        for l in out.read_text("utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                done.add((d["grupo"], d["corrida"], d["q"]))
    pending = [p for p in plan if (p[0], p[1], p[2]["q"]) not in done]
    print(f"{len(plan)} consultas en el plan, {len(pending)} pendientes -> {out}", flush=True)
    pipe, meter = build_pipeline(Path(args.env), Path(args.kb))
    from app.domain import Reason
    spent = 0
    with out.open("a", encoding="utf-8") as f:
        for n, (group, run_no, case) in enumerate(pending, 1):
            if args.max_tokens and spent >= args.max_tokens:
                print(f"tope de tokens de la evaluación alcanzado ({spent}); se reanuda después",
                      flush=True)
                return 2
            retries = 0
            while True:
                c0, t0 = meter["calls"], meter["tokens"]
                r = pipe.answer(case["q"])
                calls, tokens = meter["calls"] - c0, meter["tokens"] - t0
                spent += tokens
                if calls:
                    time.sleep(SECONDS_PER_CALL * calls)
                if meter["daily_limit"]:
                    print(f"Groq avisó del tope diario; se detiene. {meter['daily_limit']}", flush=True)
                    return 3
                if r.reason is Reason.LLM_UNAVAILABLE and retries < len(RETRY_WAITS):
                    time.sleep(RETRY_WAITS[retries])
                    retries += 1
                    continue
                break
            rec = _record(group, run_no, case, r, calls, tokens, retries)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            flag = " AÑADIDO" if rec["anadidos"] else ""
            print(f"[{n}/{len(pending)}] {group} #{run_no} {rec['resultado']}/{rec['motivo']} "
                  f"llamadas={calls} tokens={tokens}{flag} | {case['q'][:60]}", flush=True)
    print(f"terminado; tokens gastados en esta corrida: {spent}", flush=True)
    return 0


def summarize(paths: list[str], cases_path: Path = CASES) -> str:
    """Los hechos se vuelven a revisar con el archivo de casos actual, para que
    ANTES y DESPUÉS se midan con los mismos patrones aunque se hayan afinado
    después de correr."""
    facts = {c["q"]: c.get("hechos") for c in load_cases(cases_path)}
    lines = []
    for p in paths:
        rows = [json.loads(l) for l in Path(p).read_text("utf-8").splitlines() if l.strip()]
        for r in rows:
            if r["resultado"] == "answered":
                r["faltan"], r["anadidos"] = check_facts(r["texto"], facts.get(r["q"]))
        lines.append(f"== {p}")
        for group in ("bajas", "respondibles"):
            g = [r for r in rows if r["grupo"] == group]
            if not g:
                continue
            ans = [r for r in g if r["resultado"] == "answered"]
            der = [r for r in g if r["resultado"] == "derived"]
            unavailable = [r for r in der if r["motivo"] == "llm_unavailable"]
            lines.append(f"  {group}: {len(g)} consultas, {len(ans)} respondidas, "
                         f"{len(der)} derivadas ({len(unavailable)} por proveedor no disponible), "
                         f"tokens {sum(r['tokens'] for r in g)}")
            reasons: dict[str, int] = {}
            for r in der:
                reasons[r["motivo"]] = reasons.get(r["motivo"], 0) + 1
            if reasons:
                lines.append(f"    motivos de derivación: {reasons}")
            removed = [r for r in g if r.get("afirmaciones_sin_respaldo")]
            if removed:
                lines.append(f"    con afirmaciones sin respaldo detectadas: {len(removed)}")
            if group == "bajas":
                for q in dict.fromkeys(r["q"] for r in g):
                    qr = [r for r in g if r["q"] == q]
                    qa = [r for r in qr if r["resultado"] == "answered"]
                    added = [r for r in qa if r["anadidos"]]
                    missing = [r for r in qa if r["faltan"]]
                    lines.append(f"    {q[:70]!r}: {len(qa)}/{len(qr)} respondidas, "
                                 f"{len(added)} con añadido, {len(missing)} con hechos faltantes")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="app.grounding_eval")
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--kb", default=str(ROOT / "data" / "kb"))
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--out")
    ap.add_argument("--bajas", type=int, default=0, help="corridas por cada consulta de bajas")
    ap.add_argument("--respondibles", action="store_true",
                    help="una corrida por cada pregunta con respuesta")
    ap.add_argument("--solo", help="solo las consultas que contengan este texto")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="detenerse al gastar estos tokens (0 = sin tope)")
    ap.add_argument("--resumen", nargs="+", help="resumir archivos de resultados")
    args = ap.parse_args(argv)
    if args.resumen:
        print(summarize(args.resumen, Path(args.cases)))
        return 0
    logging.basicConfig(level=logging.WARNING)
    if not args.out:
        ap.error("falta --out")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
