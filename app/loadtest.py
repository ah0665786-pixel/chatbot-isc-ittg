"""Prueba de carga (RNF-EF-02: 20 consultas concurrentes, sin degradar el
tiempo de respuesta, sin errores). Usa una carpeta de datos AISLADA
(data/loadtest_*) para no tocar la base de conocimiento ni la base de datos
reales, y un LLM simulado con latencia realista por defecto, para medir la
concurrencia de nuestro propio código (SQLite, validación, pipeline) sin
depender de los límites de Groq (ver docs/capacidad-y-costos.md).

Con --kb-real usa la base de conocimiento COMPLETA (embeddings e índice
reales, preguntas reales de data/retrieval_cases.jsonl) con el LLM simulado y
una base de datos aparte (<base>_carga en PostgreSQL, o SQLite aislado). Mide
el sistema tal como corre en producción salvo el proveedor externo, sin gastar
cuota ni ensuciar las métricas reales.

Con --real-llm usa el proveedor configurado en .env, sobre tus datos reales,
para una muestra pequeña de latencia auténtica — ten en cuenta el límite del
plan gratuito antes de pedir muchos usuarios con esta opción."""
from __future__ import annotations

import json
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.adapters.db_sqlite import SqliteRepository
from app.adapters.embedders import HashingEmbedder
from app.config import Settings
from app.domain import Chunk
from app.ports import LLMReply
from app.rag.knowledge_base import KnowledgeBase
from app.rag.pipeline import RagPipeline
from app.runtime import build_runtime
from app.service import ConversationService

SAMPLE_QUESTIONS = [
    "¿Qué pasa si no cumplo el reglamento escolar?",
    "¿Cuáles son mis obligaciones como estudiante?",
    "¿Qué es una baja temporal?",
    "¿Cómo se aplican las sanciones?",
    "¿Qué derechos tengo como estudiante del ITTG?",
]

_SYNTH_DOC = (
    "Artículo 9. Las sanciones por incumplimiento del reglamento incluyen amonestación "
    "verbal o por escrito, y se registran en el expediente escolar de forma proporcional "
    "a la gravedad de la falta. Artículo 12. Los estudiantes tienen derecho a recibir "
    "educación de calidad, a la libre expresión académica y al debido proceso ante "
    "cualquier procedimiento disciplinario. Artículo 13. Son obligaciones del estudiante "
    "cumplir el reglamento, respetar a la comunidad institucional y cuidar las "
    "instalaciones. La baja temporal suspende la condición de estudiante activo por un "
    "periodo determinado, previa autorización de la coordinación de la carrera."
)


class FakeFastLLM:
    """Simula la latencia de red/generación SIN llamadas reales, para medir
    la concurrencia propia del sistema y no la del proveedor externo."""

    def __init__(self, min_s: float = 0.3, max_s: float = 1.2):
        self.min_s, self.max_s = min_s, max_s

    def chat(self, messages, temperature: float = 0.3, max_tokens: int = 900,
             deadline: float | None = None) -> LLMReply:
        time.sleep(random.uniform(self.min_s, self.max_s))
        return LLMReply(
            "Conforme al Artículo 9, las sanciones incluyen amonestación verbal o por "
            "escrito y se registran en el expediente escolar. (Reglamento, Artículo 9)",
            "fake", "fake-fast")


class FakeGroundedLLM(FakeFastLLM):
    """Como FakeFastLLM, pero contesta con texto del primer fragmento del
    CONTEXTO: así la respuesta pasa la validación de sustento igual que una
    real, y la prueba recorre el camino completo (sin reintentos artificiales)."""

    _FIRST = re.compile(r"\[1\] \((.*?)\) (.*?)(?:\n\n\[2\]|\n\nPREGUNTA:)", re.S)

    def chat(self, messages, temperature: float = 0.3, max_tokens: int = 900,
             deadline: float | None = None) -> LLMReply:
        time.sleep(random.uniform(self.min_s, self.max_s))
        m = self._FIRST.search(messages[-1]["content"])
        if not m:
            return LLMReply("NO_INFO", "fake", "fake-grounded")
        words = m.group(2).split()[:40]
        return LLMReply(f"{' '.join(words)} ({m.group(1)})", "fake", "fake-grounded")


@dataclass
class RequestResult:
    user: str
    ok: bool
    seconds: float
    outcome: str = ""
    error: str = ""


def _sandbox_service(cfg: Settings, llm_min_s: float, llm_max_s: float) -> ConversationService:
    """Entorno aislado: ni la BD ni la base de conocimiento son las reales.
    Se limpia antes de cada corrida para que los resultados sean
    reproducibles (dos corridas seguidas no deben mezclarse)."""
    import shutil

    # Umbral casi en cero a propósito: esta prueba mide CONCURRENCIA (hilos,
    # candados de la base de datos, tiempos bajo carga), no calidad de
    # recuperación semántica (para eso está 'calibrate' con el embedder
    # real). Si se usara un umbral realista, HashingEmbedder (bolsa de
    # palabras) derivaría casi todo sin llegar al LLM, y la prueba de carga
    # terminaría midiendo "qué tan rápido deriva" en vez de la concurrencia
    # real del sistema completo.
    test_cfg = replace(cfg, kb_dir=Path("data/loadtest_kb"), db_path=Path("data/loadtest.db"),
                       sim_threshold=0.01)
    shutil.rmtree(test_cfg.kb_dir, ignore_errors=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(test_cfg.db_path) + suffix).unlink(missing_ok=True)

    emb = HashingEmbedder()
    kb = KnowledgeBase.open(test_cfg, emb, rebuild=True)
    chunk = Chunk("reglamento-sintetico", "Reglamento (prueba de carga, no es real).txt", 0, _SYNTH_DOC)
    kb.store.add([chunk], emb.embed_documents([chunk.text]))
    kb.save()
    repo = SqliteRepository(test_cfg.db_path, test_cfg.session_timeout_minutes)
    pipeline = RagPipeline(test_cfg, emb, kb.store, FakeFastLLM(llm_min_s, llm_max_s))
    return ConversationService(test_cfg, repo, pipeline)


def _isolated_repo(cfg: Settings):
    """Base de datos aparte, recreada en cada corrida. Con DATABASE_URL se usa
    PostgreSQL (el mismo motor que el bot, en una base <nombre>_carga del mismo
    servidor); si no, un SQLite aislado. Nunca la base real."""
    if not cfg.database_url:
        db = Path("data/loadtest.db")
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        return SqliteRepository(db, cfg.session_timeout_minutes)

    import psycopg2
    from psycopg2.extensions import make_dsn, parse_dsn

    from app.adapters.db_postgres import PostgresRepository

    params = parse_dsn(cfg.database_url)
    test_db = f"{params['dbname']}_carga"
    admin = psycopg2.connect(cfg.database_url)
    admin.autocommit = True  # CREATE/DROP DATABASE no pueden ir en transacción
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{test_db}"')
        cur.execute(f'CREATE DATABASE "{test_db}"')
    admin.close()
    return PostgresRepository(make_dsn(cfg.database_url, dbname=test_db),
                              cfg.session_timeout_minutes)


def _load_questions(path: Path = Path("data/retrieval_cases.jsonl")) -> list[str]:
    """Preguntas reales con respuesta en los documentos; si no hay archivo, las de ejemplo."""
    try:
        rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    except FileNotFoundError:
        return SAMPLE_QUESTIONS
    return [r["q"] for r in rows if r.get("answerable", True)] or SAMPLE_QUESTIONS


def _real_kb_service(cfg: Settings, llm_min_s: float, llm_max_s: float) -> ConversationService:
    """Base de conocimiento completa y embeddings reales; LLM simulado y BD aparte.
    Solo LEE el índice (no lo guarda), así que no toca la base de conocimiento."""
    from app.runtime import build_embedder

    emb = build_embedder(cfg)
    kb = KnowledgeBase.open(cfg, emb)
    pipeline = RagPipeline(cfg, emb, kb.store, FakeGroundedLLM(llm_min_s, llm_max_s))
    return ConversationService(cfg, _isolated_repo(cfg), pipeline)


def _real_service(cfg: Settings) -> ConversationService:
    """Entorno real: tu base de conocimiento, tu BD, tu cadena de proveedores."""
    rt = build_runtime(cfg)
    return ConversationService(cfg, rt.repo, rt.pipeline)


def _one_user(svc: ConversationService, user_id: str, n_messages: int,
              questions: list[str] = SAMPLE_QUESTIONS) -> list[RequestResult]:
    out = []
    for _ in range(n_messages):
        q = random.choice(questions)
        t0 = time.perf_counter()
        try:
            result, _ = svc.handle_message(user_id, user_id, q)
            out.append(RequestResult(user_id, True, time.perf_counter() - t0,
                                     outcome=result.outcome.value))
        except Exception as e:
            out.append(RequestResult(user_id, False, time.perf_counter() - t0,
                                     error=f"{type(e).__name__}: {e}"))
    return out


def run(cfg: Settings, n_users: int = 20, messages_per_user: int = 1,
        real_llm: bool = False, llm_min_s: float = 0.3,
        llm_max_s: float = 1.2, real_kb: bool = False) -> tuple[list[RequestResult], float]:
    questions = SAMPLE_QUESTIONS
    if real_llm:
        svc = _real_service(cfg)
    elif real_kb:
        svc = _real_kb_service(cfg, llm_min_s, llm_max_s)
        questions = _load_questions()
    else:
        svc = _sandbox_service(cfg, llm_min_s, llm_max_s)
    t0 = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=n_users) as pool:
        futures = [pool.submit(_one_user, svc, f"carga-{i:03d}", messages_per_user, questions)
                  for i in range(n_users)]
        for f in as_completed(futures):
            results.extend(f.result())
    wall = time.perf_counter() - t0
    svc.repo.close()
    return results, wall


def _p95(times: list[float]) -> float:
    """Percentil 95 por rango más cercano (con 20 consultas: la 19.ª más lenta)."""
    return sorted(times)[max(0, int(len(times) * 0.95) - 1)]


def mode_label(real_llm: bool, real_kb: bool = False) -> str:
    if real_llm:
        return "LLM real (.env)"
    if real_kb:
        return "base de conocimiento completa + LLM simulado (BD de carga aparte)"
    return "LLM simulado (sandbox aislado, no toca tus datos)"


def save_report(results: list[RequestResult], wall: float, n_users: int, mode: str,
                directory: Path = Path("data"), extra: dict | None = None, now=None) -> Path:
    """Guarda el resultado en data/loadtest_AAAA-MM-DD_HHMM.json."""
    from collections import Counter
    from dataclasses import asdict
    from datetime import datetime

    now = now or datetime.now().astimezone()
    times = [r.seconds for r in results]
    path = Path(directory) / f"loadtest_{now:%Y-%m-%d_%H%M}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fecha": now.isoformat(timespec="seconds"),
        "modo": mode,
        "usuarios_concurrentes": n_users,
        "consultas": len(results),
        "tiempo_total_s": round(wall, 3),
        "promedio_s": round(statistics.mean(times), 3),
        "p95_s": round(_p95(times), 3),
        "max_s": round(max(times), 3),
        "errores": sum(not r.ok for r in results),
        "resultados": dict(Counter(r.outcome or "error" for r in results)),
        **(extra or {}),
        "consultas_detalle": [asdict(r) for r in results],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_report(results: list[RequestResult], wall: float, n_users: int, real_llm: bool,
                 real_kb: bool = False) -> None:
    times = [r.seconds for r in results]
    errors = [r for r in results if not r.ok]
    modo = mode_label(real_llm, real_kb)
    print(f"\n=== Prueba de carga: {n_users} usuarios concurrentes — {modo} ===")
    print(f"Consultas totales        : {len(results)}")
    print(f"Tiempo total (pared)     : {wall:.2f} s")
    print(f"Tiempo de respuesta prom.: {statistics.mean(times):.2f} s")
    if len(times) > 1:
        print(f"Tiempo de respuesta p95  : {_p95(times):.2f} s")
    print(f"Tiempo de respuesta máx. : {max(times):.2f} s  (RNF-EF-01: meta <= 30 s)")
    print(f"Rendimiento              : {len(results)/wall:.1f} consultas/segundo")
    from collections import Counter
    counts = Counter(r.outcome or "error" for r in results)
    print(f"Resultados               : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Errores                  : {len(errors)}/{len(results)}  (RNF-EF-02: meta 0)")
    for e in errors:
        print(f"  - {e.user}: {e.error}")
