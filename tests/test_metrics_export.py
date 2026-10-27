"""Exportación de métricas a CSV desde el panel (HU-04, HU-13)."""
import base64
import csv
import io
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.adapters.db_sqlite import SCHEMA, SqliteRepository
from app.admin_service import CSV_HEADER, AdminService
from app.domain import Chunk, Hit, Outcome, RagResult, Reason
from app.web_admin import build_app

TUXTLA = timezone(timedelta(hours=-6))  # sin horario de verano desde 2022


def utc(y, mo, d, h, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def answered(secs=1.25):
    hit = Hit(Chunk("manual", "Manual.pdf", 0, "texto"), 0.9)
    return RagResult(Outcome.ANSWERED, Reason.NONE, "respuesta", [hit], 0.9, "groq", "m", 1,
                     {"total": secs, "retrieval": 0.0123}, "ok")


def derived(reason=Reason.NOT_IN_CONTEXT):
    return RagResult(Outcome.DERIVED, reason, "deriva", [], 0.5, "", "", 1,
                     {"total": 0.4}, "detalle")


@pytest.fixture
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "app.db")
    yield r
    r.close()


def ask(repo, conv, question, result, when, link=True):
    """Simula lo que hace ConversationService.handle_message."""
    qid = repo.add_message(conv, "usuario", question, now=when)
    aid = repo.add_message(conv, "asistente", result.text, now=when)
    return repo.record_metric(conv, aid, result, now=when, question_id=qid if link else None)


def parse(csv_text):
    assert csv_text.startswith("﻿")  # BOM: Excel muestra bien los acentos
    return list(csv.reader(io.StringIO(csv_text[1:])))


# ---- repositorio --------------------------------------------------------------
def test_pregunta_correcta_aunque_dos_mensajes_lleguen_en_el_mismo_segundo(repo):
    # Telegram puede atender dos mensajes seguidos a la vez: mismas horas al segundo.
    # Con el enlace directo (id_mensaje_usuario) cada métrica trae SU pregunta.
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    t = utc(2026, 10, 22, 18)
    q1 = repo.add_message(conv, "usuario", "¿Cuántas horas dura el servicio social?", now=t)
    q2 = repo.add_message(conv, "usuario", "¿Y la residencia?", now=t)
    a2 = repo.add_message(conv, "asistente", "500 horas", now=t)
    a1 = repo.add_message(conv, "asistente", "480 horas", now=t)
    repo.record_metric(conv, a2, answered(), now=t, question_id=q2)
    repo.record_metric(conv, a1, answered(), now=t, question_id=q1)
    assert sorted(r.pregunta for r in repo.metric_rows()) == [
        "¿Cuántas horas dura el servicio social?", "¿Y la residencia?"]


def test_metricas_viejas_sin_enlace_usan_el_ultimo_mensaje_del_estudiante(repo):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    ask(repo, conv, "primera", answered(), utc(2026, 10, 22, 18, 0), link=False)
    ask(repo, conv, "segunda", derived(), utc(2026, 10, 22, 18, 5), link=False)
    assert [r.pregunta for r in repo.metric_rows()] == ["primera", "segunda"]


def test_base_existente_recibe_la_columna_nueva_sin_perder_datos(tmp_path):
    # Base creada con el esquema anterior (metricas sin id_mensaje_usuario)
    path = tmp_path / "vieja.db"
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.executescript(
        "INSERT INTO usuarios VALUES ('u','tg',NULL,NULL,'estudiante','2026-10-01',1);"
        "INSERT INTO conversaciones VALUES ('c','u','2026-10-01',NULL,'telegram');"
        "INSERT INTO mensajes VALUES ('m1','c','usuario','¿Qué es la convalidación?',"
        "  '2026-10-01T10:00:00+00:00',NULL);"
        "INSERT INTO mensajes VALUES ('m2','c','asistente','Es...','2026-10-01T10:00:01+00:00',NULL);"
        "INSERT INTO metricas(id_metrica,id_conversacion,id_mensaje,timestamp,resultado) "
        "  VALUES ('x','c','m2','2026-10-01T10:00:01+00:00','answered');")
    con.commit(); con.close()

    repo = SqliteRepository(path)  # aquí corre la migración
    cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(metricas)")}
    assert {"id_mensaje_usuario", "tiempo_busqueda_seg"} <= cols
    tablas = {r[0] for r in repo._conn.execute("SELECT name FROM sqlite_master")}
    assert {"avisos", "accesos_fallidos"} <= tablas
    [row] = repo.metric_rows()
    assert row.pregunta == "¿Qué es la convalidación?"
    assert (row.tiempo_busqueda_seg, row.aviso) == (None, None)  # la fila vieja sigue igual
    repo.close()
    SqliteRepository(path).close()  # abrirla otra vez no intenta agregarla de nuevo


def test_rango_inicio_incluido_fin_excluido(repo):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    for h in (10, 11, 12):
        ask(repo, conv, f"a las {h}", answered(), utc(2026, 10, 22, h))
    rows = repo.metric_rows("2026-10-22T11:00:00+00:00", "2026-10-22T12:00:00+00:00")
    assert [r.pregunta for r in rows] == ["a las 11"]


# ---- CSV (AdminService) -------------------------------------------------------
def test_csv_columnas_etiquetas_y_hora_local(repo, cfg, kb):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    mid = ask(repo, conv, "¿Cuántos créditos vale la residencia?", answered(1.234),
              utc(2026, 10, 22, 17, 30))
    repo.rate_response(mid, 5)
    did = ask(repo, conv, "¿Me das mi kárdex?", derived(Reason.OUT_OF_SCOPE),
              utc(2026, 10, 22, 18))
    repo.record_notice(did, "guardado", "vista_previa", "coord@test.local", "x.eml")

    rows = parse(AdminService(cfg, kb, repo).export_metrics_csv(tz=TUXTLA))
    assert rows[0] == CSV_HEADER
    assert rows[1] == ["2026-10-22 11:30:00", "¿Cuántos créditos vale la residencia?",
                       "Respondida", "", "1.23", "5", "0.012", ""]
    assert rows[2][2:] == ["Derivada",
                           "Fuera de alcance (datos personales o documentos oficiales)",
                           "0.40", "",  # sin calificación: celda vacía
                           "",          # no llegó a buscar: sin tiempo de búsqueda
                           "Guardado sin enviar (coord@test.local)"]


def test_filtro_por_dia_usa_la_hora_de_tuxtla(repo, cfg, kb):
    # 23:30 del 22 en Tuxtla = 05:30 UTC del 23: debe contar en el día 22
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    ask(repo, conv, "de noche", answered(), utc(2026, 10, 23, 5, 30))
    ask(repo, conv, "al día siguiente", answered(), utc(2026, 10, 23, 7))
    svc = AdminService(cfg, kb, repo)
    dia22 = parse(svc.export_metrics_csv(date(2026, 10, 22), date(2026, 10, 22), tz=TUXTLA))
    assert [r[1] for r in dia22[1:]] == ["de noche"]
    assert dia22[1][0] == "2026-10-22 23:30:00"
    dia23 = parse(svc.export_metrics_csv(date(2026, 10, 23), None, tz=TUXTLA))
    assert [r[1] for r in dia23[1:]] == ["al día siguiente"]


@pytest.mark.parametrize("malicious", ["=HYPERLINK(\"http://x\",\"clic\")", "+1+1", "-2+3",
                                       "@SUM(1)"])
def test_pregunta_con_formula_no_se_ejecuta_en_excel(repo, cfg, kb, malicious):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    ask(repo, conv, malicious, derived(), utc(2026, 10, 22, 18))
    rows = parse(AdminService(cfg, kb, repo).export_metrics_csv(tz=TUXTLA))
    assert rows[1][1] == "'" + malicious


def test_csv_vacio_trae_solo_encabezado(repo, cfg, kb):
    assert parse(AdminService(cfg, kb, repo).export_metrics_csv()) == [CSV_HEADER]


def test_el_bot_guarda_el_enlace_a_la_pregunta(cfg, kb, repo):
    from app.rag.pipeline import RagPipeline
    from app.service import ConversationService
    from tests.conftest import ScriptedLLM

    good = ("La baja temporal se solicita ante la Coordinación con el formato de baja "
            "temporal dentro de los primeros 20 días hábiles del semestre. (Reglamento Escolar.txt)")
    svc = ConversationService(cfg, repo, RagPipeline(cfg, kb.embedder, kb.store, ScriptedLLM(good)))
    svc.handle_message("tg-9", "Ana", "¿Cómo solicito la baja temporal?")
    linked = repo._conn.execute("SELECT id_mensaje_usuario FROM metricas").fetchone()[0]
    assert linked is not None
    assert [r.pregunta for r in repo.metric_rows()] == ["¿Cómo solicito la baja temporal?"]


# ---- panel (HTTP) -------------------------------------------------------------
@pytest.fixture
def client(kb, cfg, repo):
    cfg = type(cfg)(**{**cfg.__dict__, "admin_user": "coordinadora", "admin_password": "clave123"})
    return TestClient(build_app(cfg, AdminService(cfg, kb, repo)))


AUTH = {"Authorization": "Basic " + base64.b64encode(b"coordinadora:clave123").decode()}


def test_descarga_requiere_contrasena(client):
    assert client.get("/metricas/exportar.csv").status_code == 401


def test_descarga_csv_como_archivo(client, repo):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    ask(repo, conv, "¿Qué es el curso especial?", answered(), utc(2026, 10, 22, 18))
    r = client.get("/metricas/exportar.csv?desde=2026-10-01&hasta=2026-10-31", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == \
        'attachment; filename="metricas_2026-10-01_a_2026-10-31.csv"'
    assert r.headers["cache-control"] == "no-store"
    rows = parse(r.content.decode("utf-8"))
    assert rows[0] == CSV_HEADER and rows[1][1] == "¿Qué es el curso especial?"


def test_fechas_invertidas_o_invalidas(client):
    assert client.get("/metricas/exportar.csv?desde=2026-10-31&hasta=2026-10-01",
                      headers=AUTH).status_code == 400
    assert client.get("/metricas/exportar.csv?desde=ayer", headers=AUTH).status_code == 422


def test_panel_muestra_el_boton_de_descarga(client):
    html = client.get("/panel", headers=AUTH).text
    assert 'action="/metricas/exportar.csv"' in html and "Descargar CSV" in html


# ---- fechas vacías: así las manda el formulario del panel ---------------------
def two_days(repo):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-1"))
    ask(repo, conv, "pregunta del 22", answered(), utc(2026, 10, 22, 18))  # mediodía en Tuxtla
    ask(repo, conv, "pregunta del 24", answered(), utc(2026, 10, 24, 18))


def test_fechas_vacias_bajan_todas_las_consultas(client, repo):
    # El <input type="date"> vacío llega como "desde=&hasta=", no ausente (antes: 422)
    two_days(repo)
    r = client.get("/metricas/exportar.csv?desde=&hasta=", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith('attachment; filename="metricas_inicio_a_')
    rows = parse(r.content.decode("utf-8"))
    assert [x[1] for x in rows[1:]] == ["pregunta del 22", "pregunta del 24"]


@pytest.mark.parametrize("query,expected", [
    ("desde=2026-10-23&hasta=", ["pregunta del 24"]),   # solo "desde"
    ("desde=&hasta=2026-10-22", ["pregunta del 22"]),   # solo "hasta"
])
def test_solo_una_de_las_dos_fechas(client, repo, query, expected):
    two_days(repo)
    r = client.get(f"/metricas/exportar.csv?{query}", headers=AUTH)
    assert r.status_code == 200
    assert [x[1] for x in parse(r.content.decode("utf-8"))[1:]] == expected
