"""Puertos: contratos que separan el núcleo RAG de las tecnologías externas.
Cambiar de proveedor, de modelo de embeddings o de canal = cambiar un adaptador."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass
class LLMReply:
    text: str
    provider: str
    model: str


class LLMProvider(Protocol):
    def chat(self, messages: Sequence[dict], temperature: float = 0.3,
             max_tokens: int = 900, deadline: float | None = None) -> LLMReply:
        """deadline: instante de time.monotonic() en que hay que rendirse."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
