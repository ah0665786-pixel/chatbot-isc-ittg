"""Reportes de la coordinadora (HU-09, HU-13, HU-15, UC-14): tipos de consulta
más frecuentes, temas con más derivaciones, tasa de derivación, tiempo
promedio y p95, agrupados por semana, mes o ciclo escolar, y aviso cuando el
periodo elegido no tiene datos suficientes.

Se calcula en Python sobre las filas que ya devuelven los dos repositorios
(metric_rows): SQLite y PostgreSQL dan el mismo reporte sin SQL nuevo. Con el
volumen esperado (cientos de consultas por semestre) traer las filas es barato.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from typing import Iterable, Sequence

from app.rag.query_expansion import topic_label

GROUPINGS = {"semana": "Semana", "mes": "Mes", "ciclo": "Ciclo escolar"}

# ---- ciclo escolar del TecNM (ÚNICO lugar donde se define) -------------------
# El TecNM divide el año en tres periodos, que el SII numera 1, 2 y 3 (p. ej.
# 20263 = agosto-diciembre de 2026):
#   1  Enero-Junio       semestre de enero a junio
#   2  Verano            cursos de verano (julio)
#   3  Agosto-Diciembre  semestre de agosto a diciembre
# Los cortes van por mes completo (1 ene, 1 jul, 1 ago) para que todo día caiga
# en un ciclo, aunque las clases empiecen después: una pregunta de inscripción
# de principios de agosto es del semestre agosto-diciembre. Si el calendario
# del ITTG cambia (p. ej. el verano se extiende a agosto), se ajusta solo aquí.
# (mes de inicio, número, nombre), en orden.
SCHOOL_CYCLES = [(1, 1, "Enero-Junio"), (7, 2, "Verano"), (8, 3, "Agosto-Diciembre")]

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre"]


def fmt_day(d: date) -> str:
    return d.strftime("%d/%m/%Y")


@dataclass(frozen=True)
class Period:
    key: str        # ordena cronológicamente: "2026-10-19", "2026-10", "2026-3"
    label: str
    start: date
    end: date       # incluido


def period_of(day: date, grouping: str) -> Period:
    """Semana (lunes a domingo), mes o ciclo escolar al que pertenece `day`
    (día en hora local)."""
    if grouping == "semana":
        start = day - timedelta(days=day.weekday())
        end = start + timedelta(days=6)
        return Period(start.isoformat(), f"Semana del {fmt_day(start)} al {fmt_day(end)}",
                      start, end)
    if grouping == "mes":
        start = day.replace(day=1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        return Period(f"{day.year}-{day.month:02d}",
                      f"{MONTHS[day.month - 1].capitalize()} {day.year}", start, end)
    if grouping == "ciclo":
        i = max(i for i, (month, _, _) in enumerate(SCHOOL_CYCLES) if month <= day.month)
        month, number, name = SCHOOL_CYCLES[i]
        start = date(day.year, month, 1)
        end = (date(day.year, SCHOOL_CYCLES[i + 1][0], 1) if i + 1 < len(SCHOOL_CYCLES)
               else date(day.year + 1, 1, 1)) - timedelta(days=1)
        return Period(f"{day.year}-{number}", f"{name} {day.year}", start, end)
    raise ValueError(f"agrupación desconocida: {grouping!r} (usa {', '.join(GROUPINGS)})")


def percentile(values: Iterable[float], q: float = 0.95) -> float | None:
    """Percentil por rango más cercano: el valor que deja al menos el q de las
    mediciones en o por debajo de él (RNF-EF-01: "95 % de consultas ≤ 30 s").
    Con menos de 20 valores el p95 es simplemente el máximo."""
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(math.ceil(q * len(ordered)), 1) - 1]


# ---- estadísticas -------------------------------------------------------------
@dataclass
class Stats:
    total: int = 0
    derived: int = 0
    times: list[float] = field(default_factory=list, repr=False)
    topics: Counter = field(default_factory=Counter, repr=False)

    def add(self, row) -> None:
        self.total += 1
        self.derived += row.resultado == "derived"
        if row.tiempo_respuesta_seg is not None:
            self.times.append(row.tiempo_respuesta_seg)
        self.topics[row.tema] += 1

    @property
    def answered(self) -> int:
        return self.total - self.derived

    @property
    def derivation_rate(self) -> float:
        return self.derived / self.total if self.total else 0.0

    @property
    def avg_seconds(self) -> float | None:
        return sum(self.times) / len(self.times) if self.times else None

    @property
    def p95_seconds(self) -> float | None:
        return percentile(self.times)

    @property
    def top_topic(self) -> str:
        return topic_label(self.topics.most_common(1)[0][0]) if self.topics else ""


@dataclass
class TopicStat:
    tema: str | None
    label: str
    total: int
    derived: int

    @property
    def derivation_rate(self) -> float:
        return self.derived / self.total if self.total else 0.0


@dataclass
class GroupStat:
    period: Period
    stats: Stats
    enough: bool    # ¿alcanza el mínimo de consultas para leer sus tasas?


@dataclass
class Report:
    desde: date | None
    hasta: date | None
    grouping: str
    min_queries: int
    totals: Stats
    topics: list[TopicStat]           # tipos de consulta, el más frecuente primero
    derivations: list[TopicStat]      # temas con derivaciones, el que más deriva primero
    groups: list[GroupStat]           # periodos con datos, en orden cronológico
    notice: str | None = None         # qué falta (sin datos o periodo insuficiente)
    minimum_from: date | None = None  # el periodo debe empezar este día o antes
    last_available: GroupStat | None = None  # HU-09 CA-04: si no hay datos en el periodo

    @property
    def insufficient(self) -> bool:
        return self.totals.total < self.min_queries


def _local_day(row, tz: tzinfo) -> date:
    return datetime.fromisoformat(row.fecha).astimezone(tz).date()


def _groups(rows: Sequence, grouping: str, tz: tzinfo, min_queries: int) -> list[GroupStat]:
    by_key: dict[str, GroupStat] = {}
    for r in rows:
        p = period_of(_local_day(r, tz), grouping)
        by_key.setdefault(p.key, GroupStat(p, Stats(), False)).stats.add(r)
    groups = [by_key[k] for k in sorted(by_key)]
    for g in groups:
        g.enough = g.stats.total >= min_queries
    return groups


def build_report(rows: Sequence, desde: date | None, hasta: date | None, grouping: str,
                 tz: tzinfo, min_queries: int, today: date | None = None) -> Report:
    """rows: TODAS las consultas hasta el fin del periodo (las anteriores a
    `desde` sirven para el periodo mínimo y el último reporte disponible), en
    orden cronológico. desde/hasta: días en hora local, ambos incluidos."""
    period_of(date.today(), grouping)  # agrupación inválida: error antes de calcular
    selected = [r for r in rows if desde is None or _local_day(r, tz) >= desde]
    totals = Stats()
    for r in selected:
        totals.add(r)
    derived_by_topic = Counter(r.tema for r in selected if r.resultado == "derived")
    topics = [TopicStat(t, topic_label(t), n, derived_by_topic[t])
              for t, n in sorted(totals.topics.items(), key=lambda x: (-x[1], topic_label(x[0])))]
    derivations = sorted((t for t in topics if t.derived),
                         key=lambda t: (-t.derived, -t.derivation_rate, t.label))
    report = Report(desde, hasta, grouping, min_queries, totals, topics, derivations,
                    _groups(selected, grouping, tz, min_queries))

    end_day = hasta or today or datetime.now(tz).date()
    rango = (f"del {fmt_day(desde)} al {fmt_day(end_day)}" if desde
             else f"hasta el {fmt_day(end_day)}")
    if not rows:
        report.notice = (f"Todavía no hay consultas registradas {rango}. Se necesitan al "
                         f"menos {min_queries} para que el reporte sea confiable.")
        return report
    if not selected:  # HU-09 CA-04: mostrar el último reporte disponible
        report.last_available = _groups(rows, grouping, tz, min_queries)[-1]
        last = report.last_available
        report.notice = (f"No hay consultas {rango}. Último reporte disponible: "
                         f"{last.period.label} ({last.stats.total} consultas).")
        return report
    if report.insufficient:  # HU-15 CA-04: periodo mínimo necesario
        if len(rows) >= min_queries:
            report.minimum_from = _local_day(rows[-min_queries], tz)
            report.notice = (
                f"Datos insuficientes: hay {totals.total} consultas {rango} y se necesitan "
                f"al menos {min_queries}. Periodo mínimo necesario: del "
                f"{fmt_day(report.minimum_from)} al {fmt_day(end_day)}.")
        else:
            report.notice = (
                f"Datos insuficientes: en total hay {len(rows)} consultas registradas "
                f"(desde el {fmt_day(_local_day(rows[0], tz))}) y se necesitan al menos "
                f"{min_queries}. Ningún periodo las alcanza todavía; las cifras son solo "
                "de referencia.")
    return report
