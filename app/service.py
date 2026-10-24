"""Servicio de conversación: la misma lógica de negocio que usan tanto el
CLI (comando 'chat') como el bot de Telegram. Vive aquí para no duplicarla
entre los dos canales (RNF-PO-03: el motor no depende del canal) y para que
se pueda probar sin necesitar la librería de Telegram."""
from __future__ import annotations

from dataclasses import dataclass

from app.notify import notify_coordinadora, should_notify

from app.adapters.db_sqlite import SqliteRepository
from app.config import Settings
from app.domain import Outcome, RagResult, Reason
from app.rag.messages import derivation_message
from app.rag.knowledge_base import KnowledgeBase
from app.rag.pipeline import RagPipeline


def sync_kb_to_repo(kb: KnowledgeBase, repo: SqliteRepository) -> None:
    """Refleja el manifiesto de la base de conocimiento (FAISS) en la tabla
    DOCUMENTOS de la base de datos, para que las consultas de reportes
    (HU-13) puedan unir métricas con documentos sin abrir el índice."""
    repo.sync_documents([
        {"doc_id": d.doc_id, "name": d.name, "version": d.version,
         "n_chunks": d.n_chunks, "indexed_at": d.indexed_at}
        for d in kb.docs.values()])


@dataclass
class ConversationService:
    cfg: Settings
    repo: SqliteRepository
    pipeline: RagPipeline

    def handle_message(self, external_id: str, display_name: str, text: str,
                       canal: str = "telegram") -> tuple[RagResult, str]:
        """Procesa un mensaje de un usuario y devuelve (resultado, id_metrica).
        Crea/reusa usuario y sesión, guarda ambos mensajes y registra la
        métrica de la interacción (RF-I-01, RF-I-02, RF-I-04, RF-C-06)."""
        user_id = self.repo.get_or_create_user(external_id, display_name)
        conv_id = self.repo.get_or_create_conversation(user_id, canal)
        history = self.repo.get_history(conv_id, self.cfg.history_turns)
        question_id = self.repo.add_message(conv_id, "usuario", text)

        if len(text) > self.cfg.max_question_chars:  # HU-01 CA-04
            result = RagResult(Outcome.DERIVED, Reason.TOO_LONG,
                               derivation_message(Reason.TOO_LONG, self.cfg),
                               [], 0.0, "", "", 0, {"total": 0.0}, f"len={len(text)}")
        else:
            result = self.pipeline.answer(text, history)

        message_id = self.repo.add_message(conv_id, "asistente", result.text)
        if should_notify(result):
            notify_coordinadora(self.cfg, external_id, display_name, text, result)
        metric_id = self.repo.record_metric(conv_id, message_id, result,
                                            question_id=question_id)
        return result, metric_id

    def rate(self, metric_id: str, score: int) -> None:
        self.repo.rate_response(metric_id, score)
