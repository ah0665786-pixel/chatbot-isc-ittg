"""Lógica del panel de administración: cargar documentos y consultar
métricas (RF-I-05, HU-04, HU-13, HU-14). Separado de la app FastAPI para
poder probarlo sin levantar un servidor HTTP real — mismo patrón que
app.service.ConversationService para el bot."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.adapters.db_sqlite import MetricsSummary, SqliteRepository
from app.config import Settings
from app.rag.knowledge_base import IngestReport, KnowledgeBase
from app.rag.loaders import SUPPORTED
from app.service import sync_kb_to_repo

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB: generoso para un reglamento de 50 páginas

# Caracteres prohibidos en nombres de archivo de Windows (NTFS). Linux/macOS
# son más permisivos, pero saneamos siempre así: no sabemos con certeza en
# qué sistema operativo se desplegará, y la máquina de desarrollo SÍ es
# Windows — sin esto, un nombre con "<", ">", etc. tumba la carga con un
# OSError sin control.
_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = {"con", "prn", "aux", "nul",
                     *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}


def sanitize_filename(name: str) -> str:
    """Nombre seguro para guardar en disco en Windows, macOS o Linux.

    OJO: separamos manualmente por "/" y "\\" ANTES de usar pathlib. Si se
    le pasa a Path() un nombre con ":" sin quitar primero (p. ej.
    "a:b.txt"), WindowsPath lo interpreta como letra de unidad ("a:") y
    descarta esa parte silenciosamente — un nombre de archivo normal, sin
    ninguna intención de ruta, quedaba mutilado por esta razón."""
    name = re.split(r"[/\\]", name)[-1]  # último segmento: sin pasar por pathlib todavía
    name = _WINDOWS_FORBIDDEN.sub("_", name).strip(" .")  # Windows recorta espacios/puntos al final
    if not name:
        return "documento"
    stem, suffix = Path(name).stem, Path(name).suffix
    if stem.lower() in _WINDOWS_RESERVED:  # CON, NUL, COM1... son nombres reservados en Windows
        name = f"{stem}_{suffix}"
    return name


@dataclass
class DocumentInfo:
    doc_id: str
    name: str
    version: str
    n_chunks: int
    indexed_at: str


@dataclass
class AdminService:
    cfg: Settings
    kb: KnowledgeBase
    repo: SqliteRepository

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(
            (DocumentInfo(d.doc_id, d.name, d.version, d.n_chunks, d.indexed_at)
             for d in self.kb.docs.values()),
            key=lambda d: d.name.lower())

    def upload_document(self, filename: str, content: bytes, version: str = "") -> IngestReport:
        """Guarda el archivo recibido, lo indexa y sincroniza el manifiesto
        con la base de datos. No modifica nada si el formato no es válido
        o el archivo excede el tamaño máximo (RF-I-05 CA de error)."""
        filename = sanitize_filename(filename)
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED:
            return IngestReport(filename, "error", 0,
                                f"formato no soportado ({ext or 'sin extensión'}); "
                                f"usa PDF, DOCX, TXT o MD")
        if len(content) > MAX_UPLOAD_BYTES:
            mb = len(content) / (1024 * 1024)
            return IngestReport(filename, "error", 0,
                                f"archivo de {mb:.1f} MB excede el máximo permitido "
                                f"({MAX_UPLOAD_BYTES // (1024*1024)} MB)")
        if not content.strip():
            return IngestReport(filename, "error", 0, "el archivo llegó vacío")

        self.cfg.docs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cfg.docs_dir / filename
        dest.write_bytes(content)
        report = self.kb.ingest_file(dest, version)
        if report.status in ("added", "replaced"):
            self.kb.save()
            sync_kb_to_repo(self.kb, self.repo)
        return report

    def metrics_summary(self, date_from: str | None = None,
                        date_to: str | None = None) -> MetricsSummary:
        return self.repo.metrics_summary(date_from, date_to)
