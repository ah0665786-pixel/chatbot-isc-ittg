import pytest

from app.adapters.embedders import HashingEmbedder
from app.adapters.llm_openai_compat import LLMUnavailable
from app.config import Settings
from app.ports import LLMReply
from app.rag.knowledge_base import KnowledgeBase

REGLAMENTO = """Artículo 30. La baja temporal se solicita ante la Coordinación de la carrera
mediante el formato de baja temporal, dentro de los primeros 20 días hábiles del semestre.
El estudiante deberá entregar copia de su credencial vigente y firmar el formato. La baja
temporal no podrá exceder de dos semestres consecutivos.

Artículo 31. La baja definitiva procede por solicitud expresa del estudiante o por
incumplimiento del reglamento escolar. Servicios Escolares registra la baja definitiva."""

RESIDENCIA = """Lineamientos de Residencia Profesional. Para iniciar la residencia profesional
el estudiante debe haber acreditado el 80% de los créditos de su plan de estudios y haber
cumplido el servicio social. La residencia profesional tiene una duración mínima de 500 horas
y se registra mediante el anteproyecto autorizado por la Coordinación."""


class ScriptedLLM:
    """LLM falso: devuelve respuestas guionadas y guarda las llamadas."""

    def __init__(self, *replies, error=None):
        self.replies, self.calls, self.error = list(replies), [], error
        self.deadlines: list = []

    def chat(self, messages, temperature=0.3, max_tokens=700, deadline=None):
        self.calls.append(list(messages))
        self.deadlines.append(deadline)
        if self.error:
            raise self.error
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return LLMReply(text, "fake", "fake-model")


@pytest.fixture
def cfg(tmp_path):
    return Settings(kb_dir=tmp_path / "kb", docs_dir=tmp_path / "docs",
                    db_path=tmp_path / "app.db",
                    sim_threshold=0.20, chunk_words=60, chunk_overlap_words=10,
                    contact_email="coord.isc@example.mx", outbox_dir=tmp_path / "outbox")


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def kb(cfg, embedder, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "Reglamento Escolar.txt").write_text(REGLAMENTO, encoding="utf-8")
    (docs / "Residencia.txt").write_text(RESIDENCIA, encoding="utf-8")
    base = KnowledgeBase.open(cfg, embedder)
    base.ingest_dir(docs, "test")
    return base
