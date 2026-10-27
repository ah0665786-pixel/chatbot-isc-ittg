"""Lógica del panel de administración: cargar documentos y consultar
métricas y reportes (RF-I-05, HU-04, HU-09, HU-13, HU-14, HU-15). Separado de la app FastAPI para
poder probarlo sin levantar un servidor HTTP real — mismo patrón que
app.service.ConversationService para el bot."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path

from app.adapters.db_sqlite import MetricsSummary, SqliteRepository
from app.config import Settings
from app.domain import REASON_LABELS
from app.rag.knowledge_base import IngestReport, KnowledgeBase
from app.rag.loaders import SUPPORTED
from app.rag.query_expansion import topic_label
from app.reports import GROUPINGS, Report, build_report, fmt_day
from app.service import sync_kb_to_repo

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB: generoso para un reglamento de 50 páginas

# ---- exportación de métricas a CSV (HU-04, HU-13) ---------------------------
CSV_HEADER = ["Fecha", "Pregunta", "Tema", "Resultado", "Motivo de derivación",
              "Tiempo de respuesta (s)", "Calificación (1-5)", "Tiempo de búsqueda (s)",
              "Aviso a la coordinadora"]
RESULT_LABELS = {"answered": "Respondida", "derived": "Derivada"}
NOTICE_LABELS = {  # tabla avisos (app/notify.py)
    "enviado": "Enviado",
    "guardado": "Guardado sin enviar",
    "fallido": "Falló el envío",
}
REPORT_COLUMNS = ["Consultas", "Respondidas", "Derivadas", "Tasa de derivación (%)",
                  "Tiempo promedio (s)", "Tiempo p95 (s)"]


def _stats_cells(s) -> list:
    return [s.total, s.answered, s.derived, _pct(s.derivation_rate),
            _secs(s.avg_seconds), _secs(s.p95_seconds)]


# Una celda que empieza con = + - @ (o tab/retorno) Excel la toma como fórmula
# al abrir el archivo: la pregunta la escribe el estudiante, así que se neutraliza.
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(text: str) -> str:
    return "'" + text if text.startswith(_FORMULA_START) else text


def _secs(value: float | None, digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _notice(estado: str | None, destinatario: str | None, tipo: str | None) -> str:
    if not estado:
        return ""
    label = NOTICE_LABELS.get(estado, estado)
    if tipo == "resumen":
        label += " en el resumen diario"
    return f"{label} ({destinatario})" if destinatario else label


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}"


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo  # TZ del contenedor (Tuxtla)


def _utc_bound(day: date, tz: tzinfo) -> str:
    """Medianoche local de `day`, en el mismo formato ISO-UTC que la base."""
    local = datetime.combine(day, time.min, tzinfo=tz)
    return local.astimezone(timezone.utc).isoformat(timespec="seconds")

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

    def export_metrics_csv(self, desde: date | None = None, hasta: date | None = None,
                           tz: tzinfo | None = None) -> str:
        """CSV con una fila por consulta (HU-04, HU-13). desde/hasta son días en
        hora local, ambos incluidos; la base guarda UTC, así que sin convertir
        una consulta de las 23:00 caería en el día siguiente. Empieza con BOM
        para que Excel muestre bien los acentos."""
        tz = tz or _local_tz()
        start = _utc_bound(desde, tz) if desde else None
        end = _utc_bound(hasta + timedelta(days=1), tz) if hasta else None
        out = io.StringIO()
        out.write("﻿")
        w = csv.writer(out, lineterminator="\r\n")
        w.writerow(CSV_HEADER)
        for r in self.repo.metric_rows(start, end):
            fecha = datetime.fromisoformat(r.fecha).astimezone(tz)
            w.writerow([
                fecha.strftime("%Y-%m-%d %H:%M:%S"),
                _csv_safe(r.pregunta or ""),
                topic_label(r.tema),
                RESULT_LABELS.get(r.resultado, r.resultado),
                REASON_LABELS.get(r.motivo_derivacion, r.motivo_derivacion or ""),
                _secs(r.tiempo_respuesta_seg),
                "" if r.calificacion is None else r.calificacion,
                _secs(r.tiempo_busqueda_seg, 3),  # la búsqueda tarda milisegundos
                _notice(r.aviso, r.aviso_destinatario, r.aviso_tipo),
            ])
        return out.getvalue()

    def report(self, desde: date | None = None, hasta: date | None = None,
               agrupacion: str = "mes", tz: tzinfo | None = None,
               today: date | None = None) -> Report:
        """Reporte de la coordinadora (HU-09, HU-13, HU-15). desde/hasta: días en
        hora local, ambos incluidos, como en el CSV. Se piden también las
        consultas anteriores a `desde`: con ellas se calcula el periodo mínimo
        necesario y el último reporte disponible cuando el periodo está vacío."""
        tz = tz or _local_tz()
        end = _utc_bound(hasta + timedelta(days=1), tz) if hasta else None
        return build_report(self.repo.metric_rows(None, end), desde, hasta, agrupacion, tz,
                            self.cfg.reportes_min_consultas, today)

    def export_report_csv(self, report: Report) -> str:
        """El reporte agrupado en CSV (HU-13 CA-04): una fila por periodo con
        tiempo promedio y p95, el total y la tabla de temas. Mismas protecciones
        que export_metrics_csv: BOM para Excel y celdas de texto neutralizadas."""
        out = io.StringIO()
        out.write("﻿")
        w = csv.writer(out, lineterminator="\r\n")
        w.writerow([GROUPINGS[report.grouping], "Desde", "Hasta", *REPORT_COLUMNS,
                    "Tema más frecuente", f"Datos suficientes (mín. {report.min_queries})"])
        for g in report.groups:
            w.writerow([_csv_safe(g.period.label), fmt_day(g.period.start), fmt_day(g.period.end),
                        *_stats_cells(g.stats), _csv_safe(g.stats.top_topic),
                        "Sí" if g.enough else "No"])
        w.writerow(["Total", fmt_day(report.desde) if report.desde else "",
                    fmt_day(report.hasta) if report.hasta else "", *_stats_cells(report.totals),
                    _csv_safe(report.totals.top_topic),
                    "No" if report.insufficient else "Sí"])
        if report.notice:
            w.writerow([]); w.writerow(["Aviso", _csv_safe(report.notice)])
        w.writerow([])
        w.writerow(["Tema", "Consultas", "Derivadas", "Tasa de derivación (%)"])
        for t in report.topics:
            w.writerow([_csv_safe(t.label), t.total, t.derived, _pct(t.derivation_rate)])
        return out.getvalue()
