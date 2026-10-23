"""Calibración del umbral de similitud con un conjunto de preguntas reales.

Formato de data/golden.jsonl (una línea por pregunta):
  {"q": "¿Qué necesito para una baja temporal?", "answerable": true,
   "expected_doc": "reglamento", "expected_terms": ["baja temporal"]}
  {"q": "¿Cuál es la capital de Francia?", "answerable": false}

'answerable' = la respuesta SÍ está en los documentos. Las preguntas que NO
están (answerable=false) enseñan al sistema a derivar en vez de inventar."""
from __future__ import annotations

import json
from pathlib import Path

from app.rag import guards


def _relevant(hit, item: dict) -> bool:
    doc = guards.normalize(item.get("expected_doc", ""))
    if doc and doc not in guards.normalize(hit.chunk.doc_id + " " + hit.chunk.doc_name):
        return False
    terms = [guards.normalize(t) for t in item.get("expected_terms", [])]
    return not terms or any(t in guards.normalize(hit.chunk.text) for t in terms)


def run(golden: Path, pipeline, k: int) -> dict:
    items = [json.loads(l) for l in golden.read_text("utf-8").splitlines() if l.strip()]
    ans_best, unans_best, rows = [], [], []
    hit1 = hitk = 0
    for it in items:
        hits = pipeline.retrieve(it["q"], k)  # misma búsqueda que el bot (con expansión)
        top = hits[0].score if hits else 0.0
        if it.get("answerable", True):
            rel = [i for i, h in enumerate(hits) if _relevant(h, it)]
            best_rel = hits[rel[0]].score if rel else 0.0
            ans_best.append(best_rel)
            hit1 += bool(rel and rel[0] == 0)
            hitk += bool(rel)
            rows.append((it["q"], "sí", top, best_rel, bool(rel)))
        else:
            unans_best.append(top)
            rows.append((it["q"], "no", top, None, None))

    sweep = []
    for i in range(0, 101):
        t = i / 100
        rec = sum(s >= t for s in ans_best) / len(ans_best) if ans_best else 0.0
        spec = sum(s < t for s in unans_best) / len(unans_best) if unans_best else 0.0
        sweep.append((t, rec, spec))
    best_j = max(sweep, key=lambda x: (x[1] + x[2], x[0]))
    r95 = [x for x in sweep if x[1] >= 0.95]
    return {
        "n_answerable": len(ans_best), "n_unanswerable": len(unans_best),
        "hit_at_1": hit1 / len(ans_best) if ans_best else None,
        f"hit_at_{k}": hitk / len(ans_best) if ans_best else None,
        "threshold_balanced": {"t": best_j[0], "recall": best_j[1], "reject_rate": best_j[2]},
        "threshold_recall95": ({"t": max(r95, key=lambda x: x[0])[0],
                                "reject_rate": max(r95, key=lambda x: x[0])[2]} if r95 else None),
        "rows": rows,
    }


def print_report(rep: dict, k: int) -> None:
    print(f"\nPreguntas con respuesta en los documentos: {rep['n_answerable']}"
          f" | sin respuesta: {rep['n_unanswerable']}")
    if rep["hit_at_1"] is not None:
        print(f"Hit@1 = {rep['hit_at_1']:.0%}   Hit@{k} = {rep[f'hit_at_{k}']:.0%}"
              "   (¿el fragmento correcto está entre los recuperados?)")
    b = rep["threshold_balanced"]
    print(f"Umbral equilibrado : {b['t']:.2f}  (recupera {b['recall']:.0%} de las respondibles, "
          f"rechaza {b['reject_rate']:.0%} de las NO respondibles)")
    r = rep["threshold_recall95"]
    if r:
        print(f"Umbral con recall>=95%: {r['t']:.2f}  (rechaza {r['reject_rate']:.0%} de las NO respondibles)")
    print("\nDetalle (¿resp? | puntaje top | puntaje del fragmento correcto | acertó)")
    for q, a, top, rel, ok in rep["rows"]:
        rel_s = "-" if rel is None else f"{rel:.3f}"
        ok_s = "-" if ok is None else ("sí" if ok else "NO")
        print(f"  {a:>2} | {top:.3f} | {rel_s:>6} | {ok_s:>2} | {q[:70]}")
