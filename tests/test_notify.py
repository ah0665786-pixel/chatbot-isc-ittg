"""Pruebas de los avisos por correo a la coordinadora (HU-11, UC-12). El correo
va a un servidor SMTP simulado en 127.0.0.1 (tests/fake_smtp.py): nada sale a
internet."""
from __future__ import annotations

import logging
import socket
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from email import message_from_bytes, policy
from pathlib import Path

import pytest

from app.adapters.db_sqlite import SqliteRepository
from app.admin_service import AdminService
from app.cli import probar_correo
from app.config import Settings
from app.domain import Outcome, Reason, RagResult
from app.notify import (TIPOS, Notifier, notify_coordinadora, should_notify, student_notice,
                        tipo_de, write_examples)
from app.rag.pipeline import RagPipeline
from app.service import ConversationService
from tests.conftest import ScriptedLLM
from tests.fake_smtp import FakeSMTPServer, client_tls_context

PASSWORD = "clave-de-aplicacion-de-prueba"  # solo existe en estas pruebas
COORD = "coordinadora@test.local"
MIO = "yo@test.local"
TUXTLA = timezone(timedelta(hours=-6))
QUESTION = "¿Cómo solicito la baja temporal?"  # con ScriptedLLM("NO_INFO") se deriva


def _cfg(tmp_path: Path, port: int = 25, **overrides) -> Settings:
    base = Settings(
        smtp_host="127.0.0.1", smtp_port=port, smtp_user="bot@test.local",
        smtp_password=PASSWORD, smtp_use_tls=False, coordinadora_email=COORD,
        outbox_dir=tmp_path / "outbox")
    return replace(base, **overrides)


def _derived(reason: Reason = Reason.LOW_SIMILARITY, text="mensaje de orientación") -> RagResult:
    return RagResult(Outcome.DERIVED, reason, text, [], 0.0, "", "", 0, {"total": 0.0},
                     "detalle de prueba", [])


def _notify(cfg, question="¿cómo tramito mi residencia?", result=None, **kw):
    return notify_coordinadora(cfg, "12345", "Ana", question, result or _derived(),
                               metric_id="m-1", **kw)


def _eml(path: str):
    return message_from_bytes(Path(path).read_bytes(), policy=policy.default)


def _parts(msg) -> tuple[str, str]:
    return msg.get_body(("plain",)).get_content(), msg.get_body(("html",)).get_content()


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nadie escucha: la conexión se rechaza
    return port


# ---- tipos ------------------------------------------------------------------------
@pytest.mark.parametrize("reason,tipo", [
    (Reason.OUT_OF_SCOPE, "datos_personales"),
    (Reason.LOW_SIMILARITY, "sin_informacion"),
    (Reason.NOT_IN_CONTEXT, "sin_informacion"),
    (Reason.VALIDATION_FAILED, "sin_informacion"),
    (Reason.LLM_UNAVAILABLE, "falla_tecnica"),
])
def test_cada_motivo_escalable_tiene_su_tipo(reason, tipo):
    assert should_notify(_derived(reason)) is True
    assert tipo_de(reason).clave == tipo == tipo_de(reason.value).clave


@pytest.mark.parametrize("reason", [Reason.INJECTION, Reason.TOO_LONG, Reason.NONE])
def test_should_not_notify_for_non_escalable_reasons(reason):
    assert should_notify(_derived(reason)) is False and tipo_de(reason) is None


def test_valores_por_defecto():
    cfg = Settings()
    assert cfg.avisos_modo == "vista_previa"
    assert cfg.avisos_inmediatos == ("falla_tecnica",) and cfg.avisos_resumen_hora == "08:00"


def test_variables_del_env(monkeypatch):
    monkeypatch.setenv("AVISOS_INMEDIATOS", " Falla_Tecnica, datos_personales ,")
    monkeypatch.setenv("AVISOS_RESUMEN_HORA", "07:30")
    cfg = Settings.from_env(load_dotenv_file=False)
    assert cfg.avisos_inmediatos == ("falla_tecnica", "datos_personales")
    assert cfg.avisos_resumen_hora == "07:30"


# ---- formato del aviso --------------------------------------------------------------
def test_vista_previa_es_el_modo_por_defecto_y_no_envia(tmp_path):
    with FakeSMTPServer() as srv:
        res = _notify(_cfg(tmp_path, srv.port))
    assert srv.received == [] and srv.auth_attempts == []
    assert (res.estado, res.modo, res.destinatario) == ("guardado", "vista_previa", COORD)
    path = Path(res.archivo)
    assert path.parent == tmp_path / "outbox" and path.name.startswith("aviso_")
    msg = _eml(res.archivo)
    assert msg["To"] == COORD and msg["X-Chatbot-Modo"] == "vista_previa"
    raw = path.read_text(encoding="utf-8")
    assert "este aviso no se envió" in raw  # legible sin un programa de correo
    assert PASSWORD not in raw


def test_asunto_remitente_y_partes(tmp_path):
    res = _notify(_cfg(tmp_path))
    msg = _eml(res.archivo)
    assert msg["Subject"] == ("[Chatbot ISC] Sin información en los documentos: "
                              "¿cómo tramito mi residencia?")
    [sender] = msg["From"].addresses
    assert (sender.display_name, sender.addr_spec) == ("Chatbot ISC – ITTG", "bot@test.local")
    assert msg.get_content_type() == "multipart/alternative"
    text, html = _parts(msg)
    for part in (text, html):
        assert "¿cómo tramito mi residencia?" in part     # pregunta
        assert "mensaje de orientación" in part           # respuesta que recibió
        assert "Qué hacer" in part
        assert "No respondas a este correo" in part and "por Telegram" in part
        assert "m-1" in part and "12345" in part          # folio e ID de Telegram
    assert text.startswith("VISTA PREVIA") or "VISTA PREVIA" in text.splitlines()[0]


@pytest.mark.parametrize("reason,clave,palabra", [
    (Reason.OUT_OF_SCOPE, "datos_personales", "Servicios Escolares"),
    (Reason.NOT_IN_CONTEXT, "sin_informacion", "documento"),
    (Reason.LLM_UNAVAILABLE, "falla_tecnica", "responsable técnico"),
])
def test_cada_tipo_lleva_su_etiqueta_de_color_y_que_hacer(tmp_path, reason, clave, palabra):
    tipo = TIPOS[clave]
    msg = _eml(_notify(_cfg(tmp_path), result=_derived(reason)).archivo)
    assert msg["Subject"].startswith(f"[Chatbot ISC] {tipo.etiqueta}: ")
    _, html = _parts(msg)
    assert f"background:{tipo.fondo};color:{tipo.color}" in html
    assert tipo.resumen in html and palabra in html


def test_asunto_sin_saltos_de_linea_y_recortado(tmp_path):
    malicious = "hola\r\nBcc: victima@x.com\r\n" + "x" * 200
    msg = _eml(_notify(_cfg(tmp_path), question=malicious).archivo)
    assert msg["Bcc"] is None
    subject = msg["Subject"]
    assert "\n" not in subject and "\r" not in subject
    resumen = subject.split(": ", 1)[1]
    assert resumen.startswith("hola Bcc: victima@x.com") and len(resumen) == 60
    assert resumen.endswith("…")


def test_html_escapa_lo_que_escribe_el_estudiante(tmp_path):
    ataque = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    res = _notify(_cfg(tmp_path), question=ataque,
                  result=_derived(text="respuesta <b>con etiquetas</b>"))
    _, html = _parts(_eml(res.archivo))
    assert "<script>" not in html and "<img" not in html and "<b>con" not in html
    assert "&lt;script&gt;" in html and "&lt;b&gt;con etiquetas&lt;/b&gt;" in html
    assert "<style" not in html  # los clientes de correo lo ignoran: estilos en línea


# ---- modos y envío ------------------------------------------------------------------
def test_prueba_envia_al_correo_de_prueba_con_starttls_y_login(tmp_path):
    with FakeSMTPServer(starttls=True, password=PASSWORD) as srv:
        cfg = _cfg(tmp_path, srv.port, smtp_use_tls=True, avisos_modo="prueba",
                   avisos_correo_prueba=MIO)
        res = _notify(cfg, ssl_context=client_tls_context())
    assert (res.estado, res.destinatario, res.archivo) == ("enviado", MIO, "")
    [got] = srv.received
    assert got.rcpt_to == [MIO]          # nunca a la coordinadora en modo prueba
    assert got.tls is True and got.auth_user == "bot@test.local"
    assert got.mail_from == "bot@test.local"
    msg = got.message
    assert msg["To"] == MIO and msg["Subject"].startswith("[PRUEBA] [Chatbot ISC] ")
    text, _ = _parts(msg)
    assert f"le llegaría a {COORD}" in text
    assert not (tmp_path / "outbox").exists()


def test_prueba_sin_correo_de_prueba_no_envia_y_lo_advierte(tmp_path, caplog):
    with FakeSMTPServer() as srv, caplog.at_level(logging.WARNING, "notify"):
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="prueba", avisos_correo_prueba=""))
    assert srv.received == []
    assert res.estado == "guardado" and "AVISOS_CORREO_PRUEBA" in res.detalle
    assert Path(res.archivo).exists()
    assert "AVISOS_CORREO_PRUEBA" in caplog.text


def test_coordinadora_solo_con_el_modo_explicito(tmp_path):
    # sin usuario SMTP ni TLS: el login es opcional
    with FakeSMTPServer() as srv:
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="coordinadora", smtp_user="",
                           smtp_password="", avisos_correo_prueba=MIO))
    assert res.estado == "enviado"
    [got] = srv.received
    assert got.rcpt_to == [COORD] and got.tls is False and got.auth_user is None
    assert got.message["Subject"].startswith("[Chatbot ISC] ")
    assert "VISTA PREVIA" not in _parts(got.message)[0]


def test_coordinadora_sin_correo_configurado_no_envia(tmp_path):
    with FakeSMTPServer() as srv:
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="coordinadora",
                           coordinadora_email=""))
    assert srv.received == [] and res.estado == "guardado"
    assert "COORDINADORA_EMAIL" in res.detalle


def test_sin_contrasena_no_intenta_entrar(tmp_path):
    with FakeSMTPServer(password=PASSWORD) as srv:
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="prueba", avisos_correo_prueba=MIO,
                           smtp_password=""))
    assert srv.received == [] and srv.auth_attempts == []
    assert res.estado == "guardado" and "SMTP_PASSWORD" in res.detalle


def test_modo_desconocido_se_trata_como_vista_previa(tmp_path):
    with FakeSMTPServer() as srv:
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="enviar-ya"))
    assert srv.received == []
    assert (res.estado, res.modo) == ("guardado", "vista_previa") and "enviar-ya" in res.detalle


def test_contrasena_incorrecta_falla_sin_mostrar_la_contrasena(tmp_path, caplog):
    with FakeSMTPServer(password="otra-clave") as srv, caplog.at_level(logging.INFO):
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="prueba", avisos_correo_prueba=MIO))
    assert srv.received == [] and set(srv.auth_attempts) == {"bot@test.local"}  # PLAIN y LOGIN
    assert res.estado == "fallido" and "SMTPAuthenticationError" in res.detalle
    assert PASSWORD not in res.detalle and PASSWORD not in caplog.text
    # el aviso no se pierde: queda una copia en la carpeta
    assert PASSWORD not in Path(res.archivo).read_text(encoding="utf-8")


def test_destinatario_rechazado_se_registra_como_fallido(tmp_path):
    with FakeSMTPServer(reject_rcpt=True) as srv:
        res = _notify(_cfg(tmp_path, srv.port, avisos_modo="coordinadora", smtp_user=""))
    assert res.estado == "fallido" and "550" in res.detalle


def test_smtp_caido_no_lanza_excepcion(tmp_path):
    res = _notify(_cfg(tmp_path, _closed_port(), avisos_modo="coordinadora"))
    assert res.estado == "fallido" and Path(res.archivo).exists()


# ---- en segundo plano, desde el servicio de conversación ------------------------
def _service(kb, cfg, repo, reply="NO_INFO"):
    return ConversationService(cfg, repo, RagPipeline(cfg, kb.embedder, kb.store,
                                                      ScriptedLLM(reply)))


def _with_kb(base: Settings, cfg: Settings) -> Settings:
    return replace(cfg, **{k: getattr(base, k) for k in
                           ("kb_dir", "db_path", "sim_threshold", "contact_email")})


def test_smtp_lento_no_retrasa_la_respuesta_y_el_aviso_queda_registrado(kb, cfg):
    repo = SqliteRepository(cfg.db_path)
    with FakeSMTPServer(delay=2.0) as srv:
        cfg = _with_kb(cfg, _cfg(cfg.db_path.parent, srv.port, avisos_modo="prueba",
                                 avisos_correo_prueba=MIO, avisos_resumen_hora=""))
        svc = _service(kb, cfg, repo)
        t = time.perf_counter()
        result, metric_id = svc.handle_message("tg-1", "Ana", QUESTION)
        elapsed = time.perf_counter() - t
        assert result.reason is Reason.NOT_IN_CONTEXT
        assert "Coordinación de ISC" in result.text
        assert elapsed < 1.0, f"handle_message esperó al SMTP ({elapsed:.2f} s)"
        assert svc.notifier.wait(15)
    [got] = srv.received
    assert got.rcpt_to == [MIO]
    assert QUESTION in _parts(got.message)[0]
    [notice] = repo.notices()
    assert notice["id_metrica"] == metric_id and notice["tipo_aviso"] == "inmediato"
    assert (notice["estado"], notice["modo"], notice["destinatario"]) == ("enviado", "prueba", MIO)
    assert notice["motivo_derivacion"] == "not_in_context"
    csv = AdminService(cfg, kb, repo).export_metrics_csv()
    assert f"Enviado ({MIO})" in csv


def test_smtp_caido_no_afecta_la_respuesta(kb, cfg):
    repo = SqliteRepository(cfg.db_path)
    cfg = replace(cfg, smtp_host="127.0.0.1", smtp_port=_closed_port(), smtp_use_tls=False,
                  coordinadora_email=COORD, avisos_modo="coordinadora", avisos_resumen_hora="")
    svc = _service(kb, cfg, repo)
    result, metric_id = svc.handle_message("tg-1", "Ana", QUESTION)
    assert result.outcome is Outcome.DERIVED and metric_id
    assert svc.notifier.wait(15)
    [notice] = repo.notices()
    assert notice["estado"] == "fallido" and notice["id_metrica"] == metric_id


def test_vista_previa_desde_el_servicio_guarda_eml_y_registro(kb, cfg):
    repo = SqliteRepository(cfg.db_path)
    svc = _service(kb, replace(cfg, avisos_inmediatos=("sin_informacion",)), repo)
    _, metric_id = svc.handle_message("tg-1", "Ana", QUESTION)
    assert svc.notifier.wait(15)
    [notice] = repo.notices()
    assert notice["estado"] == "guardado" and notice["id_metrica"] == metric_id
    assert list(cfg.outbox_dir.glob("aviso_*.eml")) == [Path(notice["archivo"])]


def test_tipo_que_no_es_inmediato_espera_al_resumen(kb, cfg):
    repo = SqliteRepository(cfg.db_path)
    svc = _service(kb, cfg, repo)  # por defecto solo falla_tecnica es inmediata
    svc.handle_message("tg-1", "Ana", QUESTION)
    svc.notifier.wait(5)
    assert repo.notices() == [] and not cfg.outbox_dir.exists()


def test_lo_que_no_se_escala_no_genera_aviso(kb, cfg):
    repo = SqliteRepository(cfg.db_path)
    svc = _service(kb, replace(cfg, avisos_resumen_hora=""), repo)
    svc.handle_message("tg-1", "Ana", "hola " * 400)  # demasiado largo: no se escala
    svc.notifier.wait(5)
    assert repo.notices() == [] and not cfg.outbox_dir.exists()


def test_modo_desactivado_no_avisa(tmp_path):
    n = Notifier(_cfg(tmp_path, avisos_modo="desactivado", avisos_resumen_hora=""))
    assert n.submit("m", "1", "Ana", "pregunta", _derived()) is None


# ---- frase al estudiante: solo si es cierta ---------------------------------------
@pytest.mark.parametrize("overrides,esperada", [
    ({"avisos_modo": "vista_previa"}, None),
    ({"avisos_modo": "prueba", "avisos_correo_prueba": MIO}, None),
    ({"avisos_modo": "desactivado"}, None),
    ({"avisos_modo": "coordinadora", "smtp_password": ""}, None),   # no se podría enviar
    ({"avisos_modo": "coordinadora", "coordinadora_email": ""}, None),
    ({"avisos_modo": "coordinadora"}, "resumen diario"),           # sin_informacion: resumen
    ({"avisos_modo": "coordinadora", "avisos_inmediatos": ("sin_informacion",)},
     "Le enviaremos un aviso"),
    ({"avisos_modo": "coordinadora", "avisos_resumen_hora": ""}, "Le enviaremos un aviso"),
])
def test_frase_al_estudiante_segun_el_modo(kb, cfg, overrides, esperada):
    repo = SqliteRepository(cfg.db_path)
    cfg = _with_kb(cfg, _cfg(cfg.db_path.parent, _closed_port(), **overrides))
    svc = _service(kb, cfg, repo)
    result, _ = svc.handle_message("tg-1", "Ana", QUESTION)
    svc.notifier.close()
    guardado = repo._conn.execute(
        "SELECT contenido FROM mensajes WHERE rol='asistente'").fetchone()[0]
    assert guardado == result.text  # el historial guarda lo mismo que vio el estudiante
    if esperada is None:
        assert "Coordinación de ISC recibe" not in result.text
        assert "Le enviaremos" not in result.text
    else:
        assert esperada in result.text
        assert "sigue la indicación de arriba" in result.text


def test_la_frase_no_aparece_si_no_se_escala(tmp_path):
    cfg = _cfg(tmp_path, avisos_modo="coordinadora")
    assert student_notice(cfg, Reason.INJECTION) == ""
    assert "Le enviaremos" in student_notice(cfg, Reason.LLM_UNAVAILABLE)


# ---- resumen diario -----------------------------------------------------------------
def _seed(repo, items):
    """items: (hora UTC, motivo, pregunta). Simula lo que hace handle_message."""
    from app.domain import Outcome as O
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-7", "Luis"))
    ids = []
    for when, reason, question in items:
        qid = repo.add_message(conv, "usuario", question, now=when)
        res = RagResult(O.DERIVED if reason is not Reason.NONE else O.ANSWERED, reason,
                        "orientación", [], 0.0, "", "", 0, {"total": 0.1}, "")
        aid = repo.add_message(conv, "asistente", res.text, now=when)
        ids.append(repo.record_metric(conv, aid, res, now=when, question_id=qid))
    return ids


def utc(d, h, mi=0):
    return datetime(2026, 10, d, h, mi, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "app.db")
    yield r
    r.close()


def test_resumen_del_dia_en_hora_local_y_sin_repetir(tmp_path, repo):
    ids = _seed(repo, [
        (utc(25, 5, 30), Reason.LOW_SIMILARITY, "del 24 por la noche (23:30 local)"),
        (utc(25, 6, 30), Reason.OUT_OF_SCOPE, "¿Me das mi kárdex?"),          # 00:30 del 25
        (utc(25, 18), Reason.NONE, "respondida: no va"),
        (utc(25, 19), Reason.INJECTION, "ataque: no va"),
        (utc(25, 20), Reason.NOT_IN_CONTEXT, "¿Puedo hacer la residencia fuera?"),
        (utc(26, 5, 59), Reason.LLM_UNAVAILABLE, "<b>falla</b> a las 23:59"),  # 23:59 del 25
        (utc(26, 6, 0), Reason.LOW_SIMILARITY, "ya es del 26"),
    ])
    repo.record_notice(ids[5], "enviado", "coordinadora", COORD, tipo_aviso="inmediato")
    n = Notifier(_cfg(tmp_path), repo)  # vista previa
    res = n.send_daily_summary(date(2026, 10, 25), TUXTLA)
    assert res.estado == "guardado" and Path(res.archivo).name.startswith("resumen_")
    msg = _eml(res.archivo)
    assert msg["Subject"].strip() == "[Chatbot ISC] Resumen diario: 3 consultas derivadas el 25/10/2026"
    text, html = _parts(msg)
    for q in ("¿Me das mi kárdex?", "¿Puedo hacer la residencia fuera?"):
        assert q in text
    assert "del 24" not in text and "ya es del 26" not in text and "no va" not in text
    assert "El domingo 25 de octubre de 2026 el chatbot derivó 3 consultas a atención" in text
    assert "los estudiantes no lo reciben" in text
    assert "Datos personales o trámite oficial: 1" in text
    assert "Sin información en los documentos: 1" in text and "Falla técnica: 1" in text
    assert "00:30" in text and "23:59" in text
    assert "&lt;b&gt;falla&lt;/b&gt;" in html and "<b>falla" not in html
    assert "<table" in html and "No respondas a este correo" in html
    # cada consulta queda enlazada al resumen; el CSV prefiere el aviso inmediato
    rows = {r["id_metrica"]: r for r in repo.notices() if r["tipo_aviso"] == "resumen"}
    assert set(rows) == {ids[1], ids[4], ids[5]}
    assert {r["fecha_resumen"] for r in rows.values()} == {"2026-10-25"}
    csv = AdminService(_cfg(tmp_path), None, repo).export_metrics_csv(tz=TUXTLA)
    assert f"Guardado sin enviar en el resumen diario ({COORD})" in csv
    assert f"Enviado ({COORD})" in csv
    # el bot se reinicia: otro Notifier no lo vuelve a mandar
    assert Notifier(_cfg(tmp_path), repo).send_daily_summary(date(2026, 10, 25), TUXTLA) is None
    assert len(list((tmp_path / "outbox").glob("resumen_*.eml"))) == 1


def test_dia_sin_derivaciones_no_manda_nada(tmp_path, repo, caplog):
    _seed(repo, [(utc(25, 18), Reason.NONE, "respondida")])
    n = Notifier(_cfg(tmp_path), repo)
    with caplog.at_level(logging.INFO, "notify"):
        for minute in range(3):  # el hilo revisa cada minuto: solo una línea en el log
            assert n.run_due_summary(datetime(2026, 10, 26, 8, minute, tzinfo=TUXTLA)) is None
    assert caplog.text.count("sin consultas derivadas") == 1
    assert repo.notices() == [] and not (tmp_path / "outbox").exists()


def test_el_resumen_sale_a_la_hora_configurada_con_el_dia_anterior(tmp_path, repo):
    _seed(repo, [(utc(25, 20), Reason.NOT_IN_CONTEXT, "¿pregunta del 25?")])
    with FakeSMTPServer() as srv:
        n = Notifier(_cfg(tmp_path, srv.port, avisos_modo="prueba", avisos_correo_prueba=MIO,
                          avisos_resumen_hora="08:00"), repo)
        assert n.run_due_summary(datetime(2026, 10, 26, 7, 59, tzinfo=TUXTLA)) is None
        res = n.run_due_summary(datetime(2026, 10, 26, 8, 0, tzinfo=TUXTLA))
        assert res.estado == "enviado"
        assert n.run_due_summary(datetime(2026, 10, 26, 8, 1, tzinfo=TUXTLA)) is None
    [got] = srv.received
    assert got.rcpt_to == [MIO]
    # un asunto largo solo ASCII se dobla tras "Subject:"; al desdoblarlo queda igual
    assert got.message["Subject"].strip().startswith("[PRUEBA] [Chatbot ISC] Resumen diario: 1 consulta")


@pytest.mark.parametrize("overrides", [{"avisos_resumen_hora": ""},
                                       {"avisos_resumen_hora": "8am"},
                                       {"avisos_modo": "desactivado"}])
def test_sin_resumen_diario(tmp_path, repo, overrides):
    _seed(repo, [(utc(25, 20), Reason.NOT_IN_CONTEXT, "¿pregunta?")])
    n = Notifier(_cfg(tmp_path, **overrides), repo)
    assert n.start_daily_summary() is False
    assert n.run_due_summary(datetime(2026, 10, 26, 12, tzinfo=TUXTLA)) is None


def test_sin_resumen_todos_los_tipos_son_inmediatos(tmp_path):
    n = Notifier(_cfg(tmp_path, avisos_resumen_hora=""))
    assert n.submit("m", "1", "Ana", "pregunta", _derived(Reason.OUT_OF_SCOPE)) is not None
    n.close()


# ---- ejemplos -----------------------------------------------------------------------
def test_ejemplos_uno_por_tipo_y_el_resumen_sin_tocar_la_base(tmp_path):
    cfg = _cfg(tmp_path, db_path=tmp_path / "no-debe-existir.db")
    paths = [Path(p) for p in write_examples(cfg, tmp_path / "ejemplos")]
    assert sorted(p.name for p in paths) == [
        "ejemplo_datos_personales.eml", "ejemplo_falla_tecnica.eml",
        "ejemplo_falla_tecnica_agrupada.eml", "ejemplo_resumen_diario.eml",
        "ejemplo_sin_informacion.eml"]
    assert not cfg.db_path.exists()
    for p in paths:
        msg = _eml(str(p))
        assert msg["Subject"].startswith("[Chatbot ISC] ") and msg["To"] == COORD
        assert "VISTA PREVIA" not in _parts(msg)[0]  # tal como le llegaría
    falla = _parts(_eml(str(tmp_path / "ejemplos" / "ejemplo_falla_tecnica.eml")))[0]
    assert "Le enviaremos un aviso" in falla  # la frase que vería el estudiante
    assert "1 consulta afectada en los últimos 30 minutos" in falla
    agrupada = _parts(_eml(str(tmp_path / "ejemplos" / "ejemplo_falla_tecnica_agrupada.eml")))[0]
    assert "7 consultas afectadas desde el último aviso de falla técnica" in agrupada


# ---- prueba de envío real (python -m app.cli probar-correo) ---------------------
@pytest.mark.parametrize("missing", ["smtp_password", "avisos_correo_prueba", "smtp_user"])
def test_probar_correo_se_niega_si_falta_algo(tmp_path, capsys, missing):
    with FakeSMTPServer() as srv:
        cfg = _cfg(tmp_path, srv.port, **{"avisos_correo_prueba": MIO, missing: ""})
        assert probar_correo(cfg) == 1
    assert srv.received == []
    out = capsys.readouterr().out
    assert "No se envió nada" in out and missing.upper() in out and PASSWORD not in out


def test_probar_correo_va_al_correo_de_prueba_aunque_el_modo_sea_coordinadora(tmp_path, capsys):
    with FakeSMTPServer(password=PASSWORD) as srv:
        cfg = _cfg(tmp_path, srv.port, avisos_modo="coordinadora", avisos_correo_prueba=MIO)
        assert probar_correo(cfg) == 0
    [got] = srv.received
    assert got.rcpt_to == [MIO] and got.message["Subject"].startswith("[PRUEBA]")
    out = capsys.readouterr().out
    assert "Enviado" in out and PASSWORD not in out


# ---- límite de avisos de falla técnica (reloj simulado) -----------------------------
def _falla_runner(repo, notifier, clock):
    conv = repo.get_or_create_conversation(repo.get_or_create_user("tg-9", "Ana"))

    def falla(minute, n=None):
        t = utc(26, 18) + timedelta(minutes=minute)
        clock["now"] = t
        res = _derived(Reason.LLM_UNAVAILABLE)
        mid = repo.record_metric(conv, None, res, now=t)
        return (n or notifier).submit(mid, "tg-9", "Ana", "¿Cuántas horas dura el servicio?",
                                      res).result(10)
    return falla


def _body(res) -> str:
    return _parts(_eml(res.archivo))[0]


def test_falla_tecnica_un_aviso_por_ventana_y_el_siguiente_trae_el_conteo(tmp_path, repo):
    clock = {"now": utc(26, 18)}
    cfg = _cfg(tmp_path)  # vista previa; por defecto 30 min y falla_tecnica inmediata
    n = Notifier(cfg, repo, clock=lambda: clock["now"])
    falla = _falla_runner(repo, n, clock)
    first = falla(0)
    assert first.estado == "guardado"
    assert "1 consulta afectada en los últimos 30 minutos, contando esta." in _body(first)
    for minute in (10, 20, 29):  # dentro de la ventana: sin aviso nuevo
        assert falla(minute) is None
    assert len(repo.notices()) == 1
    second = falla(31)  # pasó la ventana: sale uno con las 4 acumuladas
    body = _body(second)
    assert "4 consultas afectadas desde el último aviso de falla técnica" in body
    assert "como máximo un aviso cada 30 minutos" in body
    assert len(list((tmp_path / "outbox").glob("aviso_*.eml"))) == 2
    # el bot se reinicia: el estado está en la base, no en memoria
    n2 = Notifier(cfg, repo, clock=lambda: clock["now"])
    assert falla(40, n2) is None
    third = falla(62, n2)
    assert "2 consultas afectadas desde el último aviso" in _body(third)  # las de 40 y 62


def test_el_limite_solo_aplica_a_falla_tecnica(tmp_path, repo):
    clock = {"now": utc(26, 18)}
    n = Notifier(_cfg(tmp_path, avisos_resumen_hora=""), repo, clock=lambda: clock["now"])
    ids = _seed(repo, [(utc(26, 18, i), Reason.LOW_SIMILARITY, "¿pregunta?") for i in range(3)])
    results = [n.submit(mid, "1", "Ana", "¿pregunta?", _derived()).result(10) for mid in ids]
    assert all(r is not None and r.estado == "guardado" for r in results)
    assert "afectada" not in _body(results[-1])


def test_sin_limite_con_intervalo_cero(tmp_path, repo):
    clock = {"now": utc(26, 18)}
    n = Notifier(_cfg(tmp_path, avisos_falla_intervalo_min=0), repo, clock=lambda: clock["now"])
    falla = _falla_runner(repo, n, clock)
    assert falla(0) is not None and falla(1) is not None
    assert "afectada" not in _body(falla(2))


def test_un_envio_fallido_no_cuenta_como_aviso(tmp_path, repo):
    clock = {"now": utc(26, 18)}
    cfg = _cfg(tmp_path, _closed_port(), avisos_modo="coordinadora")
    n = Notifier(cfg, repo, clock=lambda: clock["now"])
    falla = _falla_runner(repo, n, clock)
    assert falla(0).estado == "fallido"
    assert falla(5).estado == "fallido"  # no llegó a nadie: se vuelve a intentar


def test_variable_del_intervalo(monkeypatch):
    monkeypatch.setenv("AVISOS_FALLA_INTERVALO_MIN", "45")
    assert Settings.from_env(load_dotenv_file=False).avisos_falla_intervalo_min == 45
    assert Settings().avisos_falla_intervalo_min == 30
