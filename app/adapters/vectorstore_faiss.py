"""Índice vectorial FAISS (IndexFlatIP sobre vectores normalizados = coseno).
Guarda en disco el índice, los fragmentos y qué modelo de embeddings se usó,
para negarse a mezclar vectores de modelos distintos."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from app.domain import Chunk, Hit


class IncompatibleIndex(Exception):
    """El índice guardado se creó con otro modelo de embeddings: hay que reindexar."""


class FaissStore:
    def __init__(self, dim: int, embed_name: str):
        self.dim, self.embed_name = dim, embed_name
        self._chunks: list[Chunk] = []
        self._vecs = np.zeros((0, dim), dtype="float32")
        self._index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return len(self._chunks)

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if len(self._vecs):
            self._index.add(self._vecs)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks y vectores no coinciden")
        if len(chunks) == 0:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"dimensión {vectors.shape[1]} != {self.dim}")
        self._chunks.extend(chunks)
        self._vecs = np.vstack([self._vecs, vectors])
        self._index.add(vectors)

    def remove_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._vecs = self._vecs[keep] if keep else np.zeros((0, self.dim), "float32")
            self._rebuild()
        return removed

    def search(self, qvec: np.ndarray, k: int = 5) -> list[Hit]:
        if not len(self._chunks):
            return []
        q = np.ascontiguousarray(qvec, dtype="float32").reshape(1, -1)
        scores, idx = self._index.search(q, min(k, len(self._chunks)))
        return [Hit(self._chunks[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    # ---- persistencia (bytes en vez de rutas: evita problemas de FAISS con
    #      rutas con acentos/espacios en Windows) ---------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.faiss").write_bytes(faiss.serialize_index(self._index).tobytes())
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(
            json.dumps({"embed_model": self.embed_name, "dim": self.dim,
                        "n_chunks": len(self._chunks)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, dim: int, embed_name: str) -> "FaissStore":
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta["embed_model"] != embed_name or meta["dim"] != dim:
            raise IncompatibleIndex(
                f"El índice se creó con '{meta['embed_model']}' ({meta['dim']} dim) pero "
                f"ahora se usa '{embed_name}' ({dim} dim). Reindexa: python -m app.cli ingest --rebuild")
        store = cls(dim, embed_name)
        raw = np.frombuffer((directory / "index.faiss").read_bytes(), dtype="uint8")
        store._index = faiss.deserialize_index(raw)
        n = store._index.ntotal
        store._vecs = store._index.reconstruct_n(0, n) if n else np.zeros((0, dim), "float32")
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            store._chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return store

    @staticmethod
    def exists(directory: Path) -> bool:
        return (directory / "meta.json").exists()
