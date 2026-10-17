from __future__ import annotations

from dataclasses import dataclass

from app.adapters.db_sqlite import SqliteRepository
from app.adapters.embedders import HashingEmbedder, SentenceTransformerEmbedder
from app.adapters.llm_openai_compat import FallbackLLM, build_llm
from app.config import Settings
from app.ports import Embedder
from app.rag.knowledge_base import KnowledgeBase
from app.rag.pipeline import RagPipeline
from app.kb_watcher import KBWatcher


@dataclass
class Runtime:
    cfg: Settings
    embedder: Embedder
    kb: KnowledgeBase
    llm: FallbackLLM
    pipeline: RagPipeline
    repo: SqliteRepository


def build_embedder(cfg: Settings, fake: bool = False) -> Embedder:
    if fake:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(cfg.embed_model, cfg.embed_query_prefix, cfg.embed_doc_prefix)


def build_runtime(cfg: Settings, fake_embeddings: bool = False, rebuild: bool = False) -> Runtime:
    emb = build_embedder(cfg, fake_embeddings)
    kb = KnowledgeBase.open(cfg, emb, rebuild=rebuild)
    llm = build_llm(cfg.llm_endpoints, cfg.llm_timeout, cfg.llm_max_retries, cfg.llm_total_budget)
    if cfg.database_url:
        from app.adapters.db_postgres import PostgresRepository
        repo = PostgresRepository(cfg.database_url, cfg.session_timeout_minutes)
    else:
        repo = SqliteRepository(cfg.db_path, cfg.session_timeout_minutes)
    watcher = KBWatcher(kb_dir=cfg.kb_dir)
    return Runtime(cfg, emb, kb, llm, RagPipeline(cfg, emb, kb.store, llm, watcher=watcher), repo)
