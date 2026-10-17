"""Prueba de carga (RNF-EF-02: 20 consultas concurrentes, sin degradar el
tiempo de respuesta, sin errores). Usa una carpeta de datos AISLADA
(data/loadtest_*) para no tocar la base de conocimiento ni la base de datos
reales, y un LLM simulado con latencia realista por defecto, para medir la
concurrencia de nuestro propio código (SQLite, validación, pipeline) sin
depender de los límites de Groq (ver docs/capacidad-y-costos.md).

Con --real-llm usa el proveedor configurado en .env, sobre tus datos reales,
para una muestra pequeña de latencia auténtica — ten en cuenta el límite del
plan gratuito antes de pedir muchos usuarios con esta opción."""
from __future__ import annotations

import random
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

    def chat(self, messages, temperature: float = 0.3, max_tokens: int = 900) -> LLMReply:
        time.sleep(random.uniform(self.min_s, self.max_s))
        return LLMReply(
            "Conforme al Artículo 9, las sanciones incluyen amonestación verbal o por "
            "escrito y se registran en el expediente escolar. (Reglamento, Artículo 9)",
            "fake", "fake-fast")


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


def _real_service(cfg: Settings) -> ConversationService:
    """Entorno real: tu base de conocimiento, tu BD, tu cadena de proveedores."""
    rt = build_runtime(cfg)
    return ConversationService(cfg, rt.repo, rt.pipeline)


def _one_user(svc: ConversationService, user_id: str, n_messages: int) -> list[RequestResult]:
    out = []
    for _ in range(n_messages):
        q = random.choice(SAMPLE_QUESTIONS)
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
        llm_max_s: float = 1.2) -> tuple[list[RequestResult], float]:
    svc = _real_service(cfg) if real_llm else _sandbox_service(cfg, llm_min_s, llm_max_s)
    t0 = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=n_users) as pool:
        futures = [pool.submit(_one_user, svc, f"carga-{i:03d}", messages_per_user)
                  for i in range(n_users)]
        for f in as_completed(futures):
            results.extend(f.result())
    wall = time.perf_counter() - t0
    svc.repo.close()
    return results, wall


def print_report(results: list[RequestResult], wall: float, n_users: int, real_llm: bool) -> None:
    times = [r.seconds for r in results]
    errors = [r for r in results if not r.ok]
    modo = "LLM real (.env)" if real_llm else "LLM simulado (sandbox aislado, no toca tus datos)"
    print(f"\n=== Prueba de carga: {n_users} usuarios concurrentes — {modo} ===")
    print(f"Consultas totales        : {len(results)}")
    print(f"Tiempo total (pared)     : {wall:.2f} s")
    print(f"Tiempo de respuesta prom.: {statistics.mean(times):.2f} s")
    if len(times) > 1:
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        print(f"Tiempo de respuesta p95  : {p95:.2f} s")
    print(f"Tiempo de respuesta máx. : {max(times):.2f} s  (RNF-EF-01: meta <= 30 s)")
    print(f"Rendimiento              : {len(results)/wall:.1f} consultas/segundo")
    print(f"Errores                  : {len(errors)}/{len(results)}  (RNF-EF-02: meta 0)")
    for e in errors:
        print(f"  - {e.user}: {e.error}")
