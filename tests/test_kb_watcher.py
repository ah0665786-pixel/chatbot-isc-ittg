"""Pruebas para KBWatcher: detección de cambios en el knowledge base en
disco y recarga en caliente del FaissStore de un RagPipeline, sin reiniciar
el proceso (RF-I-05 CA-03)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.kb_watcher import KBWatcher


class _FakeClock:
    """Reloj controlable a mano para probar el throttle sin usar time.sleep."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeEmbedder:
    def __init__(self, name: str = "fake", dim: int = 8) -> None:
        self.name = name
        self.dim = dim


class _FakeStore:
    """Sustituto de FaissStore.load para no depender de faiss real en esta
    prueba unitaria: solo necesitamos verificar que maybe_reload llama a
    load() con los argumentos correctos y reemplaza pipeline.store."""

    last_load_args = None

    @classmethod
    def load(cls, kb_dir, dim, embed_name):
        cls.last_load_args = (Path(kb_dir), dim, embed_name)
        return cls()


class _FakePipeline:
    def __init__(self, store) -> None:
        self.store = store
        self.embedder = _FakeEmbedder()


def _write_manifest(kb_dir: Path, content: str = "{}") -> None:
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "manifest.json").write_text(content, encoding="utf-8")


_MTIME_BASE = 2_000_000_000.0  # fecha ficticia lejana, nunca choca con la real


def _touch_manifest_with_new_mtime(kb_dir: Path, clock: _FakeClock) -> None:
    """Simula una nueva escritura del manifest con una fecha de modificación
    distinta. La fecha se deriva del reloj FALSO de la prueba (no del reloj
    real de Windows/Linux): usar el reloj real es inestable porque dos
    'toques' seguidos pueden caer en el mismo instante real (milisegundos
    de diferencia), haciendo que la prueba falle de forma intermitente."""
    import os

    path = kb_dir / "manifest.json"
    path.write_text(json.dumps({"version": clock.now}), encoding="utf-8")
    new_mtime = _MTIME_BASE + clock.now
    os.utime(path, (new_mtime, new_mtime))


def test_no_reload_when_manifest_unchanged(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    _write_manifest(kb_dir)

    clock = _FakeClock()
    watcher = KBWatcher(kb_dir=kb_dir, min_interval=0.0, clock=clock)
    pipeline = _FakePipeline(store="original")

    monkeypatch.setattr("app.adapters.vectorstore_faiss.FaissStore", _FakeStore)

    clock.advance(10)
    reloaded = watcher.maybe_reload(pipeline)

    assert reloaded is False
    assert pipeline.store == "original"


def test_reload_when_manifest_changes(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    _write_manifest(kb_dir)

    clock = _FakeClock()
    watcher = KBWatcher(kb_dir=kb_dir, min_interval=0.0, clock=clock)
    pipeline = _FakePipeline(store="original")

    monkeypatch.setattr("app.adapters.vectorstore_faiss.FaissStore", _FakeStore)

    # Simula que el panel de administración subió un documento nuevo:
    # se reescribe manifest.json con una fecha de modificación más reciente.
    clock.advance(10)
    _touch_manifest_with_new_mtime(kb_dir, clock)

    reloaded = watcher.maybe_reload(pipeline)

    assert reloaded is True
    assert isinstance(pipeline.store, _FakeStore)
    assert _FakeStore.last_load_args == (kb_dir, 8, "fake")


def test_throttle_skips_check_before_min_interval(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    _write_manifest(kb_dir)

    clock = _FakeClock()
    watcher = KBWatcher(kb_dir=kb_dir, min_interval=5.0, clock=clock)
    pipeline = _FakePipeline(store="original")

    monkeypatch.setattr("app.adapters.vectorstore_faiss.FaissStore", _FakeStore)

    clock.advance(1.0)
    _touch_manifest_with_new_mtime(kb_dir, clock)

    # Todavía no pasó min_interval desde la última revisión (t=0), así que
    # no debe ni siquiera mirar el disco.
    reloaded = watcher.maybe_reload(pipeline)
    assert reloaded is False
    assert pipeline.store == "original"

    # Ahora sí pasó el intervalo: debe detectar el cambio.
    clock.advance(5.0)
    reloaded = watcher.maybe_reload(pipeline)
    assert reloaded is True
    assert pipeline.store != "original"


def test_no_reload_when_manifest_file_missing(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(parents=True, exist_ok=True)
    # No se crea manifest.json a propósito.

    clock = _FakeClock()
    watcher = KBWatcher(kb_dir=kb_dir, min_interval=0.0, clock=clock)
    pipeline = _FakePipeline(store="original")

    monkeypatch.setattr("app.adapters.vectorstore_faiss.FaissStore", _FakeStore)

    clock.advance(10)
    reloaded = watcher.maybe_reload(pipeline)

    assert reloaded is False
    assert pipeline.store == "original"


def test_second_change_after_successful_reload_is_detected(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    _write_manifest(kb_dir)

    clock = _FakeClock()
    watcher = KBWatcher(kb_dir=kb_dir, min_interval=0.0, clock=clock)
    pipeline = _FakePipeline(store="original")

    monkeypatch.setattr("app.adapters.vectorstore_faiss.FaissStore", _FakeStore)

    clock.advance(1)
    _touch_manifest_with_new_mtime(kb_dir, clock)
    assert watcher.maybe_reload(pipeline) is True

    # Sin cambios nuevos, una segunda llamada no debe recargar otra vez.
    clock.advance(1)
    assert watcher.maybe_reload(pipeline) is False

    # Un segundo cambio real sí debe detectarse.
    clock.advance(1)
    _touch_manifest_with_new_mtime(kb_dir, clock)
    assert watcher.maybe_reload(pipeline) is True
