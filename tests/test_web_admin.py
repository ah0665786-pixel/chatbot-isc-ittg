import base64

import pytest
from fastapi.testclient import TestClient

from app.adapters.db_sqlite import SqliteRepository
from app.admin_service import AdminService
from app.web_admin import build_app


@pytest.fixture
def client(kb, cfg):
    cfg = type(cfg)(**{**cfg.__dict__, "admin_user": "coordinadora", "admin_password": "clave123"})
    svc = AdminService(cfg, kb, SqliteRepository(cfg.db_path, cfg.session_timeout_minutes))
    app = build_app(cfg, svc)
    return TestClient(app)


def auth_headers(user="coordinadora", password="clave123"):
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_panel_requires_authentication(client):
    r = client.get("/panel")
    assert r.status_code == 401


def test_panel_rejects_wrong_password(client):
    r = client.get("/panel", headers=auth_headers(password="incorrecta"))
    assert r.status_code == 401


def test_panel_rejects_wrong_username(client):
    r = client.get("/panel", headers=auth_headers(user="otro"))
    assert r.status_code == 401


def test_panel_loads_with_correct_credentials(client):
    r = client.get("/panel", headers=auth_headers())
    assert r.status_code == 200
    assert "Reglamento Escolar.txt" in r.text
    assert "Consultas totales" in r.text


def test_root_redirects_to_panel(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/panel"


def test_upload_valid_document_appears_in_panel(client):
    files = {"archivo": ("Titulacion.txt", b"Requisitos de titulacion de la carrera. " * 5,
                         "text/plain")}
    r = client.post("/documentos/cargar", files=files, data={"version": "2026-1"},
                    headers=auth_headers(), follow_redirects=False)
    assert r.status_code == 303 and "ok=1" in r.headers["location"]

    panel = client.get("/panel", headers=auth_headers())
    assert "Titulacion.txt" in panel.text and "2026-1" in panel.text


def test_upload_unsupported_extension_shows_error_and_is_not_indexed(client):
    files = {"archivo": ("virus.exe", b"contenido", "application/octet-stream")}
    r = client.post("/documentos/cargar", files=files, data={}, headers=auth_headers(),
                    follow_redirects=False)
    assert "ok=0" in r.headers["location"]

    panel = client.get("/panel", headers=auth_headers())
    assert "virus.exe" not in panel.text


def test_upload_without_auth_is_rejected(client):
    files = {"archivo": ("x.txt", b"contenido de prueba largo " * 5, "text/plain")}
    r = client.post("/documentos/cargar", files=files)
    assert r.status_code == 401


def test_document_name_with_html_is_neutralized_in_panel(client):
    # Defensa en dos capas: sanitize_filename() quita "<"/">" ANTES de
    # guardar (también evita el OSError real de Windows con esos
    # caracteres), y lo que sí llegara a la plantilla Jinja2 lo escapa.
    # Ninguna de las dos formas del ataque debe aparecer sin neutralizar.
    nombre = "<img src=x onerror=alert(1)>.txt"
    files = {"archivo": (nombre, b"contenido inofensivo de prueba " * 5, "text/plain")}
    r = client.post("/documentos/cargar", files=files, data={}, headers=auth_headers(),
                    follow_redirects=False)
    assert "ok=1" in r.headers["location"]

    panel = client.get("/panel", headers=auth_headers())
    assert "<img src=x onerror=alert(1)>" not in panel.text       # crudo: no aparece
    assert "&lt;img src=x onerror=alert(1)&gt;" not in panel.text  # tampoco hacía falta escapar
    assert "_img src=x onerror=alert(1)_.txt" in panel.text        # forma saneada, la que se guardó


def test_version_field_is_escaped_if_it_contains_html(client):
    # El nombre de archivo ya no puede llevar "<"/">" (se sanea), pero el
    # campo de texto libre "versión" no pasa por ese saneo — este es el
    # caso real que sí depende de que Jinja2 escape correctamente.
    files = {"archivo": ("Normal.txt", b"contenido de prueba " * 5, "text/plain")}
    client.post("/documentos/cargar", files=files,
               data={"version": "<b>2026</b>"}, headers=auth_headers())
    panel = client.get("/panel", headers=auth_headers())
    assert "<b>2026</b>" not in panel.text
    assert "&lt;b&gt;2026&lt;/b&gt;" in panel.text


def test_api_documentos_json(client):
    r = client.get("/api/documentos", headers=auth_headers())
    assert r.status_code == 200
    names = {d["name"] for d in r.json()}
    assert "Reglamento Escolar.txt" in names


def test_api_metricas_json(client):
    r = client.get("/api/metricas", headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0 and "precision_rate" in body
