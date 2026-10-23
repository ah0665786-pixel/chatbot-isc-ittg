"""Uso:
  python -m app.cli ingest [carpeta] [--version 2026-1] [--rebuild] [--fake]
  python -m app.cli ask "¿requisitos de residencia?" [--fake]        (sin guardar)
  python -m app.cli chat [--user <id>] [--fake]                      (guarda en BD)
  python -m app.cli report [--from 2026-09-01] [--to 2026-09-30]
  python -m app.cli calibrate data/golden.jsonl [--fake]
  python -m app.cli redteam [data/injection_cases.jsonl] [--fake]   (RNF-SE-04)
  python -m app.cli loadtest [--n-users 20] [--messages 1] [--real-llm]  (RNF-EF-02)
  python -m app.cli check-llm
  python -m app.cli kb            # lista documentos indexados
--fake usa embeddings de prueba (sin descargar modelos)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app import calibrate as cal
from app import loadtest as load_mod
from app import redteam as rt_mod
from app.adapters.llm_openai_compat import OpenAICompatLLM
from app.adapters.vectorstore_faiss import IncompatibleIndex
from app.config import Settings
from app.domain import Outcome
from app.runtime import build_runtime
from app.service import ConversationService, sync_kb_to_repo


def _show(res) -> None:
    print(f"\n[{res.outcome.value}/{res.reason.value}] score máx={res.max_score:.3f} "
          f"modelo={res.provider}/{res.model} intentos={res.attempts} "
          f"total={res.timings.get('total', 0):.2f}s")
    if res.detail and res.detail != "ok":
        print(f"  detalle: {res.detail}")
    print(f"\n{res.text}\n")
    for h in res.hits:
        print(f"  · {h.score:.3f}  {h.chunk.doc_name} {h.chunk.section} #{h.chunk.chunk_no}")
    for d in res.debug:
        print(f"\n  [diagnóstico] {d}")


def _report(rt, date_from: str | None, date_to: str | None) -> None:
    s = rt.repo.metrics_summary(date_from, date_to)
    print(f"\nConsultas totales : {s.total}")
    if not s.total:
        print("(sin datos todavía — usa 'chat' para generar interacciones)")
        return
    print(f"Respondidas        : {s.answered}")
    print(f"Derivadas          : {s.derived}")
    print(f"Tasa de precisión  : {s.precision_rate:.1%}  (RF-RN-03: meta >= 90%)")
    print(f"Tasa de derivación : {s.derivation_rate:.1%}")
    if s.avg_response_seconds is not None:
        print(f"Tiempo prom. resp. : {s.avg_response_seconds:.2f} s  (RNF-EF-01: meta <= 30 s)")
    if s.by_reason:
        print("Motivos de derivación:")
        for reason, n in sorted(s.by_reason.items(), key=lambda x: -x[1]):
            print(f"  {reason:<20} {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="app.cli")
    ap.add_argument("cmd", choices=["ingest", "ask", "chat", "calibrate", "check-llm",
                                    "kb", "report", "redteam", "loadtest"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--version", default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--user", default="cli-user", help="telegram_id simulado para 'chat'")
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--n-users", type=int, default=20)
    ap.add_argument("--messages", type=int, default=1)
    ap.add_argument("--real-llm", action="store_true",
                    help="usa el LLM real de .env en vez del simulado (cuidado con la cuota)")
    ap.add_argument("--kb-real", action="store_true",
                    help="loadtest con la base de conocimiento completa y BD de carga aparte")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Settings.from_env()

    if a.cmd == "check-llm":
        if not cfg.llm_endpoints:
            print("No hay proveedores LLM configurados (revisa .env)")
            return 1
        rc = 0
        for e in cfg.llm_endpoints:
            p = OpenAICompatLLM(e.name, e.base_url, e.api_key, e.model, cfg.llm_timeout, 1, e.extra)
            try:
                models = p.list_models()
                listed = "sí" if e.model in models else "NO (¿retirado o mal escrito?)"
                extra_s = f" extra={e.extra}" if e.extra else ""
                print(f"[{e.name}] modelo '{e.model}' listado por el proveedor: {listed}{extra_s}")
                r = p.chat([{"role": "user", "content": "Responde solo: OK"}], max_tokens=300)
                if not r.text.strip():
                    print(f"[{e.name}] AVISO: respuesta vacía con max_tokens=300. Es un modelo "
                          f'razonador: agrega LLM_i_EXTRA={{"reasoning_effort":"low"}} en .env')
                else:
                    print(f"[{e.name}] prueba de chat: {r.text[:60]!r}")
            except Exception as ex:
                rc = 1
                print(f"[{e.name}] FALLA: {ex}")
        return rc

    # 'loadtest' sin --real-llm arma su propio entorno (sandbox, o con --kb-real
    # la base de conocimiento completa) y su propia BD: no toca la base real.
    if a.cmd == "loadtest" and not a.real_llm:
        results, wall = load_mod.run(cfg, n_users=a.n_users, messages_per_user=a.messages,
                                     real_kb=a.kb_real)
        load_mod.print_report(results, wall, a.n_users, False, a.kb_real)
        out = load_mod.save_report(results, wall, a.n_users,
                                   load_mod.mode_label(False, a.kb_real))
        print(f"\nResultado guardado en {out}")
        return 0

    try:
        rt = build_runtime(cfg, fake_embeddings=a.fake, rebuild=(a.cmd == "ingest" and a.rebuild))
    except IncompatibleIndex as ex:
        print(ex)
        return 2

    if a.cmd == "ingest":
        folder = Path(a.arg) if a.arg else cfg.docs_dir
        for r in rt.kb.ingest_dir(folder, a.version):
            print(f"  {r.status:<9} {r.name}  ({r.n_chunks} fragmentos) {r.message}")
        sync_kb_to_repo(rt.kb, rt.repo)
        print(f"\nBase de conocimiento: {len(rt.kb.docs)} documentos, {len(rt.kb.store)} fragmentos")

    elif a.cmd == "kb":
        for d in rt.kb.docs.values():
            print(f"  {d.name:<40} v={d.version or '-':<8} {d.n_chunks:>4} frag.  {d.indexed_at}")
        print(f"Total: {len(rt.kb.docs)} documentos, {len(rt.kb.store)} fragmentos")

    elif a.cmd == "ask":  # prueba rápida, no queda guardada en la base de datos
        _show(rt.pipeline.answer(a.arg or input("Pregunta: ")))

    elif a.cmd == "chat":  # sesión real: se guarda en la base de datos, como hará el bot
        svc = ConversationService(cfg, rt.repo, rt.pipeline)
        print(f"Sesión: usuario={a.user}")
        print("Escribe tu pregunta (vacío para salir).")
        while (q := input("\n> ").strip()):
            res, metric_id = svc.handle_message(a.user, a.user, q, canal="cli")
            _show(res)
            if res.outcome is Outcome.ANSWERED:
                cal_in = input("  Calificación 1-5 (Enter para omitir): ").strip()
                if cal_in.isdigit() and 1 <= int(cal_in) <= 5:
                    svc.rate(metric_id, int(cal_in))

    elif a.cmd == "report":
        _report(rt, a.date_from, a.date_to)

    elif a.cmd == "calibrate":
        rep = cal.run(Path(a.arg or "data/golden.jsonl"), rt.pipeline, cfg.top_k)
        cal.print_report(rep, cfg.top_k)

    elif a.cmd == "redteam":
        cases = rt_mod.load_cases(a.arg or "data/injection_cases.jsonl")
        results = rt_mod.run(cases, rt.pipeline)
        rt_mod.print_report(results)
        models = sorted({f"{e.name}/{e.model}" for e in cfg.llm_endpoints})
        out = rt_mod.save_report(results, Path("data"), model=", ".join(models))
        print(f"\nResultados guardados en {out}")

    elif a.cmd == "loadtest":  # solo llega aquí con --real-llm
        results, wall = load_mod.run(cfg, n_users=a.n_users, messages_per_user=a.messages,
                                     real_llm=True)
        load_mod.print_report(results, wall, a.n_users, True)
        out = load_mod.save_report(results, wall, a.n_users, load_mod.mode_label(True))
        print(f"\nResultado guardado en {out}")

    rt.repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
