"""Vigilancia del índice FAISS para recarga en caliente (RF-I-05 CA-03).

Cada proceso (bot de Telegram, panel de administración, CLI) construye su
propio Runtime con su propia copia en memoria del FaissStore. Cuando el
panel de administración (u otro proceso) agrega o actualiza un documento,
escribe un nuevo index.faiss / chunks.jsonl / manifest.json en disco, pero
los demás procesos ya en ejecución siguen usando la copia vieja cargada en
memoria hasta que se reinician.

KBWatcher resuelve esto sin reiniciar nada: antes de responder cada
pregunta, RagPipeline le pregunta "¿cambió el knowledge base en disco desde
la última vez?" (comparando la fecha de modificación de manifest.json). Si
cambió, reconstruye el FaissStore desde disco y lo reemplaza en caliente en
el propio pipeline (pipeline.store = nuevo_store). El chequeo de disco se
limita a como máximo una vez cada `min_interval` segundos para no golpear el
sistema de archivos en cada mensaje.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class KBWatcher:
    """Detecta cambios en el knowledge base en disco y recarga el índice
    FAISS de un RagPipeline en caliente, sin reiniciar el proceso."""

    kb_dir: Path
    min_interval: float = 2.0
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.kb_dir = Path(self.kb_dir)
        self._manifest_path = self.kb_dir / "manifest.json"
        self._last_check: float = 0.0
        self._last_mtime: Optional[float] = self._manifest_mtime()

    def _manifest_mtime(self) -> Optional[float]:
        try:
            return self._manifest_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def changed(self) -> bool:
        """True si el manifest.json cambió desde la última vez que se
        confirmó una recarga, respetando el throttle de min_interval.
        No tiene efectos secundarios sobre el estado "último mtime visto";
        solo maybe_reload() actualiza ese estado."""
        now = self.clock()
        if now - self._last_check < self.min_interval:
            return False
        self._last_check = now
        mtime = self._manifest_mtime()
        return mtime is not None and mtime != self._last_mtime

    def maybe_reload(self, pipeline) -> bool:
        """Si el knowledge base cambió en disco, reconstruye el FaissStore
        y lo asigna a pipeline.store. Devuelve True si hubo recarga."""
        now = self.clock()
        if now - self._last_check < self.min_interval:
            return False
        self._last_check = now

        mtime = self._manifest_mtime()
        if mtime is None or mtime == self._last_mtime:
            return False

        from app.adapters.vectorstore_faiss import FaissStore

        embedder = pipeline.embedder
        new_store = FaissStore.load(self.kb_dir, embedder.dim, embedder.name)
        pipeline.store = new_store
        self._last_mtime = mtime
        return True
