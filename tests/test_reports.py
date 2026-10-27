"""Reportes de la coordinadora (HU-09 CA-02/CA-04, HU-13, HU-15, RNF-EF-01):
agrupación por semana, mes y ciclo escolar en hora de Tuxtla, temas más
frecuentes y con más derivaciones, tiempo promedio y p95, y periodo
insuficiente o sin datos."""
import csv
import io
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.adapters.db_sqlite import SCHEMA, SqliteRepository
from app.admin_service import AdminService
from app.reports import period_of, percentile
from app.web_admin import build_app
from tests.test_metrics_export import AUTH, TUXTLA, answered, ask, derived, parse, utc


@pytest.fixture
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "app.db")
    yield r
    r.close()


def service(cfg, kb, repo, min_queries=1):
    cfg = type(cfg)(**{**cfg.__dict__, "reportes_min_consultas": min_queries})
    return AdminService(cfg, kb, repo)


@pytest.fixture
def conv(repo):
    return repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))


# ---- periodos -------------------------------------------------------------------
@pytest.mark.parametrize("day,grouping,label,start,end", [
    (date(2026, 10, 26), "semana", "Semana del 26/10/2026 al 01/11/2026",
     date(2026, 10, 26), date(2026, 11, 1)),                       # lunes
    (date(2026, 11, 1), "semana", "Semana del 26/10/2026 al 01/11/2026",
     date(2026, 10, 26), date(2026, 11, 1)),                       # domingo
    (date(2026, 2, 14), "mes", "Febrero 2026", date(2026, 2, 1), date(2026, 2, 28)),
    (date(2026, 12, 31), "mes", "Diciembre 2026", date(2026, 12, 1), date(2026, 12, 31)),
    (date(2026, 1, 1), "ciclo", "Enero-Junio 2026", date(2026, 1, 1), date(2026, 6, 30)),
    (date(2026, 6, 30), "ciclo", "Enero-Junio 2026", date(2026, 1, 1), date(2026, 6, 30)),
    (date(2026, 7, 1), "ciclo", "Verano 2026", date(2026, 7, 1), date(2026, 7, 31)),
    (date(2026, 8, 1), "ciclo", "Agosto-Diciembre 2026", date(2026, 8, 1), date(2026, 12, 31)),
    (date(2026, 12, 31), "ciclo", "Agosto-Diciembre 2026", date(2026, 8, 1), date(2026, 12, 31)),
])
def test_periodos(day, grouping, label, start, end):
    p = period_of(day, grouping)
    assert (p.label, p.start, p.end) == (label, start, end)


def test_claves_ordenan_cronologicamente():
    keys = [period_of(d, "ciclo").key for d in (date(2026, 1, 5), date(2026, 7, 5),
                                                  date(2026, 8, 5), date(2027, 1, 5))]
    assert keys == sorted(keys) == ["2026-1", "2026-2", "2026-3", "2027-1"]


def test_agrupacion_desconocida():
    with pytest.raises(ValueError):
        period_of(date(2026, 1, 1), "anio")


def test_p95_por_rango_mas_cercano():
    assert percentile([]) is None
    assert percentile([3.0]) == 3.0
    assert percentile(range(1, 20)) == 19        # menos de 20 valores: el máximo
    assert percentile(range(1, 21)) == 19        # 20 valores: el 19.º
    assert percentile(range(1, 101)) == 95


# ---- cortes de fecha en hora local -----------------------------------------------
@pytest.mark.parametrize("grouping,when,label_local", [
    # domingo 01/11 a las 23:30 en Tuxtla = lunes 02/11 05:30 UTC: semana del 26/10
    ("semana", utc(2026, 11, 2, 5, 30), "Semana del 26/10/2026 al 01/11/2026"),
    # 31/10 a las 22:00 en Tuxtla = 01/11 04:00 UTC: octubre
    ("mes", utc(2026, 11, 1, 4), "Octubre 2026"),
    # 31/07 a las 20:00 en Tuxtla = 01/08 02:00 UTC: verano, no agosto-diciembre
    ("ciclo", utc(2026, 8, 1, 2), "Verano 2026"),
    # 31/12/2026 a las 19:00 en Tuxtla = 01/01/2027 01:00 UTC: ago-dic 2026
    ("ciclo", utc(2027, 1, 1, 1), "Agosto-Diciembre 2026"),
])
def test_cortes_en_hora_de_tuxtla(cfg, kb, repo, conv, grouping, when, label_local):
    ask(repo, conv, "¿Qué es el servicio social?", answered(), when, tema="servicio_social")
    r = service(cfg, kb, repo).report(agrupacion=grouping, tz=TUXTLA)
    assert [g.period.label for g in r.groups] == [label_local]


def test_agrupa_por_mes_con_tasas_tiempos_y_tema(cfg, kb, repo, conv):
    for i, secs in enumerate((1.0, 2.0, 3.0)):  # octubre: 3 respondidas de servicio social
        ask(repo, conv, "servicio social", answered(secs), utc(2026, 10, 5 + i, 18),
            tema="servicio_social")
    ask(repo, conv, "kárdex", derived(), utc(2026, 10, 20, 18), tema="otros")
    ask(repo, conv, "baja", derived(), utc(2026, 11, 3, 18), tema="bajas")
    r = service(cfg, kb, repo).report(agrupacion="mes", tz=TUXTLA)
    octubre, noviembre = r.groups
    assert (octubre.period.label, octubre.stats.total, octubre.stats.derived) == \
        ("Octubre 2026", 4, 1)
    assert octubre.stats.derivation_rate == 0.25
    assert octubre.stats.avg_seconds == pytest.approx((1 + 2 + 3 + 0.4) / 4)
    assert octubre.stats.p95_seconds == 3.0
    assert octubre.stats.top_topic == "Servicio social"
    assert (noviembre.period.label, noviembre.stats.top_topic) == ("Noviembre 2026", "Bajas")
    assert r.totals.total == 5 and r.totals.derivation_rate == 0.4


def test_tipos_mas_frecuentes_y_temas_con_mas_derivaciones(cfg, kb, repo, conv):
    t = utc(2026, 10, 22, 18)
    for _ in range(3):
        ask(repo, conv, "titulación", answered(), t, tema="titulacion")
    ask(repo, conv, "titulación", derived(), t, tema="titulacion")
    ask(repo, conv, "permiso", derived(), t, tema="permisos")
    ask(repo, conv, "permiso", derived(), t, tema="permisos")
    ask(repo, conv, "vieja", answered(), t)  # sin tema: anterior a la clasificación
    r = service(cfg, kb, repo).report(tz=TUXTLA)
    assert [(x.label, x.total) for x in r.topics] == [
        ("Titulación", 4), ("Permisos", 2), ("Sin clasificar", 1)]
    assert [(x.label, x.derived, x.derivation_rate) for x in r.derivations] == [
        ("Permisos", 2, 1.0), ("Titulación", 1, 0.25)]


def test_filtro_por_fechas_en_hora_local(cfg, kb, repo, conv):
    ask(repo, conv, "22 de noche", answered(), utc(2026, 10, 23, 5, 30))  # 22/10 23:30 local
    ask(repo, conv, "23", answered(), utc(2026, 10, 23, 18))
    svc = service(cfg, kb, repo)
    assert svc.report(date(2026, 10, 22), date(2026, 10, 22), tz=TUXTLA).totals.total == 1
    assert svc.report(date(2026, 10, 23), None, tz=TUXTLA).totals.total == 1


# ---- periodo insuficiente y sin datos (HU-15 CA-04, HU-09 CA-04) -----------------
def test_sin_ninguna_consulta(cfg, kb, repo):
    r = service(cfg, kb, repo, 20).report(tz=TUXTLA, today=date(2026, 10, 26))
    assert r.totals.total == 0 and r.groups == [] and r.last_available is None
    assert r.notice == ("Todavía no hay consultas registradas hasta el 26/10/2026. "
                        "Se necesitan al menos 20 para que el reporte sea confiable.")


def test_periodo_vacio_muestra_el_ultimo_reporte_disponible(cfg, kb, repo, conv):
    ask(repo, conv, "septiembre", answered(), utc(2026, 9, 10, 18), tema="titulacion")
    ask(repo, conv, "octubre", derived(), utc(2026, 10, 5, 18), tema="bajas")
    r = service(cfg, kb, repo).report(date(2026, 11, 1), date(2026, 11, 30), "mes", tz=TUXTLA)
    assert r.totals.total == 0
    assert r.last_available.period.label == "Octubre 2026"
    assert r.last_available.stats.total == 1
    assert r.notice == ("No hay consultas del 01/11/2026 al 30/11/2026. Último reporte "
                        "disponible: Octubre 2026 (1 consultas).")


def test_periodo_vacio_antes_de_cualquier_consulta(cfg, kb, repo, conv):
    ask(repo, conv, "octubre", answered(), utc(2026, 10, 5, 18))
    r = service(cfg, kb, repo).report(date(2026, 1, 1), date(2026, 1, 31), tz=TUXTLA)
    assert r.last_available is None and r.notice.startswith("Todavía no hay consultas")


def test_periodo_insuficiente_dice_el_periodo_minimo(cfg, kb, repo, conv):
    # 5 consultas, una por día del 01/10 al 05/10, y 2 en la semana del 19/10
    for d in range(1, 6):
        ask(repo, conv, f"día {d}", answered(), utc(2026, 10, d, 18))
    ask(repo, conv, "día 20", answered(), utc(2026, 10, 20, 18))
    ask(repo, conv, "día 21", answered(), utc(2026, 10, 21, 18))
    r = service(cfg, kb, repo, 4).report(date(2026, 10, 19), date(2026, 10, 25), "semana",
                                         tz=TUXTLA)
    assert r.insufficient and r.totals.total == 2
    # Para juntar 4 hay que empezar el 04/10: 04, 05, 20 y 21
    assert r.minimum_from == date(2026, 10, 4)
    assert r.notice == ("Datos insuficientes: hay 2 consultas del 19/10/2026 al 25/10/2026 y "
                        "se necesitan al menos 4. Periodo mínimo necesario: del 04/10/2026 "
                        "al 25/10/2026.")
    # Con ese periodo ya alcanza
    ok = service(cfg, kb, repo, 4).report(date(2026, 10, 4), date(2026, 10, 25), tz=TUXTLA)
    assert not ok.insufficient and ok.notice is None


def test_periodo_minimo_en_hora_local(cfg, kb, repo, conv):
    # La consulta que completa el mínimo fue el 03/10 a las 23:30 en Tuxtla (04/10 en UTC)
    ask(repo, conv, "noche", answered(), utc(2026, 10, 4, 5, 30))
    ask(repo, conv, "después", answered(), utc(2026, 10, 10, 18))
    r = service(cfg, kb, repo, 2).report(date(2026, 10, 10), date(2026, 10, 10), tz=TUXTLA)
    assert r.minimum_from == date(2026, 10, 3)


def test_ningun_periodo_alcanza_el_minimo(cfg, kb, repo, conv):
    for d in (1, 2, 3):
        ask(repo, conv, f"día {d}", answered(), utc(2026, 10, d, 18))
    r = service(cfg, kb, repo, 20).report(tz=TUXTLA, today=date(2026, 10, 26))
    assert r.insufficient and r.minimum_from is None
    assert r.notice.startswith("Datos insuficientes: en total hay 3 consultas registradas "
                               "(desde el 01/10/2026) y se necesitan al menos 20.")
    assert [g.enough for g in r.groups] == [False]


def test_minimo_por_defecto_viene_de_la_configuracion(monkeypatch):
    from app.config import Settings
    assert Settings().reportes_min_consultas == 20
    monkeypatch.setenv("REPORTES_MIN_CONSULTAS", "50")
    assert Settings.from_env(load_dotenv_file=False).reportes_min_consultas == 50


# ---- p95 en el resumen y en los CSV (RNF-EF-01, N13) -----------------------------
def test_resumen_trae_p95(repo, conv):
    for secs in range(1, 21):
        ask(repo, conv, "q", answered(float(secs)), utc(2026, 10, 22, 18))
    s = repo.metrics_summary()
    assert s.avg_response_seconds == 10.5 and s.p95_response_seconds == 19.0


def test_csv_del_reporte(cfg, kb, repo, conv):
    ask(repo, conv, "q", answered(1.0), utc(2026, 10, 5, 18), tema="titulacion")
    ask(repo, conv, "q", derived(), utc(2026, 11, 5, 18), tema="=cmd")  # tema raro: neutralizado
    svc = service(cfg, kb, repo, 2)
    rows = parse(svc.export_report_csv(svc.report(agrupacion="mes", tz=TUXTLA)))
    assert rows[0] == ["Mes", "Desde", "Hasta", "Consultas", "Respondidas", "Derivadas",
                       "Tasa de derivación (%)", "Tiempo promedio (s)", "Tiempo p95 (s)",
                       "Tema más frecuente", "Datos suficientes (mín. 2)"]
    assert rows[1] == ["Octubre 2026", "01/10/2026", "31/10/2026", "1", "1", "0", "0.0",
                       "1.00", "1.00", "Titulación", "No"]
    assert rows[2][0] == "Noviembre 2026" and rows[2][9] == "'=cmd"
    assert rows[3][:4] == ["Total", "", "", "2"] and rows[3][8] == "1.00"
    i = rows.index(["Tema", "Consultas", "Derivadas", "Tasa de derivación (%)"])
    assert rows[i + 1:] == [["'=cmd", "1", "1", "100.0"], ["Titulación", "1", "0", "0.0"]]


def test_csv_del_reporte_trae_el_aviso(cfg, kb, repo, conv):
    ask(repo, conv, "q", answered(), utc(2026, 10, 5, 18))
    svc = service(cfg, kb, repo, 20)
    rows = parse(svc.export_report_csv(svc.report(tz=TUXTLA)))
    assert any(r and r[0] == "Aviso" and r[1].startswith("Datos insuficientes") for r in rows)


# ---- migración ------------------------------------------------------------------
def test_base_existente_recibe_la_columna_tema(tmp_path):
    path = tmp_path / "vieja.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executescript(
        "INSERT INTO usuarios VALUES ('u','tg',NULL,NULL,'estudiante','2026-10-01',1);"
        "INSERT INTO conversaciones VALUES ('c','u','2026-10-01',NULL,'telegram');"
        "INSERT INTO metricas(id_metrica,id_conversacion,timestamp,resultado) "
        "  VALUES ('x','c','2026-10-01T10:00:01+00:00','answered');")
    con.commit(); con.close()
    repo = SqliteRepository(path)
    cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(metricas)")}
    assert "tema" in cols
    [row] = repo.metric_rows()
    assert row.tema is None  # no se reclasifica: queda "Sin clasificar"
    repo.close()
    SqliteRepository(path).close()


# ---- panel (HTTP) ---------------------------------------------------------------
@pytest.fixture
def client(kb, cfg, repo):
    cfg = type(cfg)(**{**cfg.__dict__, "admin_user": "coordinadora", "admin_password": "clave123",
                       "reportes_min_consultas": 2})
    return TestClient(build_app(cfg, AdminService(cfg, kb, repo)))


def test_panel_muestra_los_reportes(client, repo, conv):
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    ask(repo, conv, "titulación", answered(), now, tema="titulacion")
    ask(repo, conv, "permiso", derived(), now, tema="permisos")
    html = client.get("/panel?agrupacion=ciclo", headers=AUTH).text
    for text in ("Reportes", "Tipos de consulta más frecuentes", "Temas con más derivaciones",
                 "Por ciclo escolar", "Titulación", "Permisos", "Tiempo p95",
                 'href="/reportes/exportar.csv?desde=&amp;hasta=&amp;agrupacion=ciclo"'):
        assert text in html, text


def test_panel_con_fechas_vacias_y_periodo_sin_datos(client, repo, conv):
    ask(repo, conv, "q", answered(), utc(2026, 10, 5, 18), tema="bajas")
    r = client.get("/panel?desde=&hasta=&agrupacion=semana", headers=AUTH)
    assert r.status_code == 200 and "Datos insuficientes" in r.text  # 1 consulta, mínimo 2
    html = client.get("/panel?desde=2026-11-01&hasta=2026-11-30", headers=AUTH).text
    assert "No hay consultas del 01/11/2026 al 30/11/2026" in html
    assert "Último reporte disponible: Octubre 2026" in html


def test_panel_fechas_invertidas_y_agrupacion_invalida(client):
    r = client.get("/panel?desde=2026-10-31&hasta=2026-10-01", headers=AUTH)
    assert r.status_code == 200 and "posterior a" in r.text
    assert client.get("/panel?agrupacion=anio", headers=AUTH).status_code == 422


def test_descarga_del_reporte(client, repo, conv):
    ask(repo, conv, "q", answered(), utc(2026, 10, 5, 18), tema="bajas")
    assert client.get("/reportes/exportar.csv").status_code == 401
    r = client.get("/reportes/exportar.csv?desde=2026-10-01&hasta=2026-10-31&agrupacion=semana",
                   headers=AUTH)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert r.headers["content-disposition"] == \
        'attachment; filename="reporte_semana_2026-10-01_a_2026-10-31.csv"'
    rows = parse(r.content.decode("utf-8"))
    assert rows[0][0] == "Semana" and rows[1][0] == "Semana del 05/10/2026 al 11/10/2026"
    assert client.get("/reportes/exportar.csv?desde=2026-10-31&hasta=2026-10-01",
                      headers=AUTH).status_code == 400
