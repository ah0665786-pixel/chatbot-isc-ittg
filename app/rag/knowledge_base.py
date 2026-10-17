"""Base de conocimiento: ingesta de documentos + manifiesto de qué hay indexado.
El manifiesto responde: ¿qué documentos, qué versión, cuándo, cuántos fragmentos?"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.adapters.vectorstore_faiss import FaissStore, IncompatibleIndex
from app.config import Settings
from app.domain import Chunk
from app.ports import Embedder
from app.rag.chunking import chunk_text
from app.rag.loaders import SUPPORTED, load_sections


def slugify(name: str) -> str:
    n = unicodedata.normalize("NFD", name.lower())
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", n).strip("-")


@dataclass
class DocRecord:
    doc_id: str
    name: str
    version: str
    sha256: str
    n_chunks: int
    indexed_at: str
    active: bool = True


@dataclass
class IngestReport:
    name: str
    status: str  # added | replaced | unchanged | error
    n_chunks: int = 0
    message: str = ""


class KnowledgeBase:
    def __init__(self, cfg: Settings, embedder: Embedder, store: FaissStore,
                 docs: dict[str, DocRecord] | None = None):
        self.cfg, self.embedder, self.store = cfg, embedder, store
        self.docs: dict[str, DocRecord] = docs or {}

    # ---- apertura / guardado ----------------------------------------------
    @classmethod
    def open(cls, cfg: Settings, embedder: Embedder, rebuild: bool = False) -> "KnowledgeBase":
        d = cfg.kb_dir
        if not rebuild and FaissStore.exists(d):
            store = FaissStore.load(d, embedder.dim, embedder.name)  # puede lanzar IncompatibleIndex
            mp = d / "manifest.json"
            docs = ({r["doc_id"]: DocRecord(**r) for r in json.loads(mp.read_text("utf-8"))}
                    if mp.exists() else {})
            return cls(cfg, embedder, store, docs)
        return cls(cfg, embedder, FaissStore(embedder.dim, embedder.name))

    def save(self) -> None:
        self.store.save(self.cfg.kb_dir)
        (self.cfg.kb_dir / "manifest.json").write_text(
            json.dumps([asdict(r) for r in self.docs.values()], indent=2, ensure_ascii=False),
            encoding="utf-8")

    # ---- ingesta -----------------------------------------------------------
    def ingest_file(self, path: Path, version: str = "") -> IngestReport:
        path = Path(path)
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            doc_id = slugify(path.stem)
            prev = self.docs.get(doc_id)
            if prev and prev.sha256 == sha and prev.active:
                return IngestReport(path.name, "unchanged", prev.n_chunks, "sin cambios")

            chunks: list[Chunk] = []
            for section, text in load_sections(path):
                for piece in chunk_text(text, self.cfg.chunk_words, self.cfg.chunk_overlap_words):
                    chunks.append(Chunk(doc_id, path.name, len(chunks), piece, section))
            if not chunks:
                return IngestReport(path.name, "error", 0,
                                    "sin texto extraíble (¿PDF escaneado? requiere OCR)")

            vectors = self.embedder.embed_documents([c.text for c in chunks])
            replaced = prev is not None
            if replaced:
                self.store.remove_document(doc_id)
            self.store.add(chunks, vectors)
            self.docs[doc_id] = DocRecord(
                doc_id, path.name, version, sha, len(chunks),
                datetime.now(timezone.utc).isoformat(timespec="seconds"))
            return IngestReport(path.name, "replaced" if replaced else "added", len(chunks))
        except Exception as e:  # un documento malo no debe tumbar la carga completa
            return IngestReport(path.name, "error", 0, f"{type(e).__name__}: {e}")

    def ingest_dir(self, directory: Path, version: str = "") -> list[IngestReport]:
        files = sorted(p for p in Path(directory).iterdir()
                       if p.is_file() and p.suffix.lower() in SUPPORTED)
        reports = [self.ingest_file(p, version) for p in files]
        self.save()
        return reports
