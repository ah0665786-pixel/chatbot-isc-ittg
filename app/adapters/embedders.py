from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Sequence

import numpy as np


class SentenceTransformerEmbedder:
    """Embeddings locales: sin API, sin costo, sin fecha de retiro.
    El modelo se descarga una vez (Hugging Face) y luego funciona sin internet."""

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "",
                 batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # import perezoso

        self.name = model_name
        self._q, self._d, self._bs = query_prefix, doc_prefix, batch_size
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension",
                          self._model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())

    def _enc(self, texts: Sequence[str]) -> np.ndarray:
        v = self._model.encode(list(texts), normalize_embeddings=True,
                               batch_size=self._bs, show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc([self._d + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._enc([self._q + text])[0]


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", t)


class HashingEmbedder:
    """Embedder determinista por bolsa de palabras. SOLO para pruebas y para
    correr el sistema sin descargar modelos; no entiende sinónimos."""

    def __init__(self, dim: int = 512):
        self.name, self.dim = f"hashing-{dim}", dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in _tokens(text):
            if len(tok) <= 3:  # ignora artículos/preposiciones (ruido léxico)
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
