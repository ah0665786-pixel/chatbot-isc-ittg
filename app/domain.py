from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_name: str
    chunk_no: int
    text: str
    section: str = ""  # p. ej. "pág. 3"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Outcome(str, Enum):
    ANSWERED = "answered"
    DERIVED = "derived"


class Reason(str, Enum):
    """Por qué se derivó a atención humana (se registra en METRICAS)."""

    NONE = "none"
    OUT_OF_SCOPE = "out_of_scope"          # datos personales / documentos oficiales
    INJECTION = "injection"                # intento de manipular el prompt
    LOW_SIMILARITY = "low_similarity"      # nada en la base supera el umbral
    NOT_IN_CONTEXT = "not_in_context"      # el LLM dijo que el contexto no alcanza
    VALIDATION_FAILED = "validation_failed"
    LLM_UNAVAILABLE = "llm_unavailable"
    TOO_LONG = "too_long"                  # HU-01 CA-04


# Reason en palabras de la coordinación (CSV del panel y asunto del aviso por correo)
REASON_LABELS = {
    "out_of_scope": "Fuera de alcance (datos personales o documentos oficiales)",
    "injection": "Intento de manipular al asistente",
    "low_similarity": "Tema no encontrado en los documentos",
    "not_in_context": "Sin información suficiente en los documentos",
    "validation_failed": "Respuesta no verificable contra los documentos",
    "llm_unavailable": "Servicio de IA no disponible",
    "too_long": "Pregunta demasiado larga",
}


@dataclass
class RagResult:
    outcome: Outcome
    reason: Reason
    text: str                       # mensaje final para el estudiante
    hits: list[Hit] = field(default_factory=list)
    max_score: float = 0.0
    provider: str = ""
    model: str = ""
    attempts: int = 0               # generaciones intentadas (1 o 2)
    timings: dict[str, float] = field(default_factory=dict)
    detail: str = ""
    debug: list[str] = field(default_factory=list)  # borradores rechazados, para diagnóstico

    @property
    def derived(self) -> bool:
        return self.outcome is Outcome.DERIVED
