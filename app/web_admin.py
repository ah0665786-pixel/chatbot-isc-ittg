"""Capa de administración: panel por navegador para que la coordinadora
cargue documentos y consulte métricas sin usar la consola (RF-I-05,
RNF-US-04, HU-04, HU-13). FastAPI es solo el adaptador HTTP; toda la lógica
vive en app.admin_service.AdminService, ya probada por separado.

Uso:
  python -m app.web_admin
Requiere ADMIN_PASSWORD en el .env (si falta, el servidor no arranca).
"""
from __future__ import annotations

import logging
import os
import secrets
from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from app.admin_service import AdminService
from app.config import Settings
from app.rag.knowledge_base import KnowledgeBase

log = logging.getLogger("web_admin")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Un <input type="date"> vacío llega como "desde=" (cadena vacía), no ausente:
# sin esto FastAPI responde 422. Vacío = sin fecha; una fecha mal escrita sigue
# dando 422.
OptionalDate = Annotated[date | None, BeforeValidator(lambda v: v or None)]
security = HTTPBasic()
MAX_LOGGED_USER = 100  # recorta el usuario intentado antes de guardarlo


def build_app(cfg: Settings, svc: AdminService) -> FastAPI:
    app = FastAPI(title="Panel de Administración ISC")
    app.state.svc = svc

    def _record_failed_login(username: str, request: Request) -> None:
        """Acceso no autorizado (HU-01 CA-03, RNF-SE-01): usuario intentado, IP y
        ruta al log y a la tabla accesos_fallidos. La contraseña intentada no se
        guarda en ningún lado. En Docker la IP es la de la red de Docker."""
        usuario = username[:MAX_LOGGED_USER]
        ip = request.client.host if request.client else None
        log.warning("acceso no autorizado al panel: usuario=%r ip=%s ruta=%s",
                    usuario, ip, request.url.path)
        try:
            svc.repo.record_failed_login(usuario, ip, request.url.path)
        except Exception:  # noqa: BLE001 — la respuesta sigue siendo 401 aunque falle la BD
            log.exception("no se pudo registrar el acceso no autorizado")

    def require_auth(request: Request,
                     credentials: HTTPBasicCredentials = Depends(security)) -> str:
        # Sin encabezado de autenticación HTTPBasic responde 401 antes de llegar
        # aquí: es el navegador pidiendo usuario y contraseña, no un intento.
        user_ok = secrets.compare_digest(credentials.username.encode(), cfg.admin_user.encode())
        pass_ok = secrets.compare_digest(credentials.password.encode(),
                                         cfg.admin_password.encode())
        if not (user_ok and pass_ok):
            _record_failed_login(credentials.username, request)
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Credenciales inválidas",
                                headers={"WWW-Authenticate": "Basic"})
        return credentials.username

    @app.get("/")
    def root():
        return RedirectResponse("/panel")

    @app.get("/panel")
    def panel(request: Request, msg: str = "", ok: int = 1, user: str = Depends(require_auth)):
        return templates.TemplateResponse(request, "panel.html", {
            "documents": svc.list_documents(),
            "metrics": svc.metrics_summary(),
            "message": msg or None,
            "message_ok": bool(ok),
        })

    @app.post("/documentos/cargar")
    async def cargar(archivo: UploadFile = File(...), version: str = Form(""),
                     user: str = Depends(require_auth)):
        content = await archivo.read()
        report = svc.upload_document(archivo.filename or "sin_nombre", content, version)
        ok = report.status in ("added", "replaced")
        detalle = f" ({report.n_chunks} fragmentos)" if ok else ""
        texto = (f"'{report.name}' {report.status}{detalle}. {report.message}".strip()
                if not ok else f"'{report.name}' {report.status}{detalle} correctamente.")
        return RedirectResponse(f"/panel?msg={texto}&ok={int(ok)}",
                               status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/api/documentos")
    def api_documentos(user: str = Depends(require_auth)):
        return [d.__dict__ for d in svc.list_documents()]

    @app.get("/api/metricas")
    def api_metricas(desde: str | None = None, hasta: str | None = None,
                     user: str = Depends(require_auth)):
        m = svc.metrics_summary(desde, hasta)
        return m.__dict__

    @app.get("/metricas/exportar.csv")
    def exportar_csv(desde: OptionalDate = None, hasta: OptionalDate = None,
                     user: str = Depends(require_auth)):
        """Descarga de métricas por consulta (HU-04, HU-13). desde/hasta: AAAA-MM-DD,
        en hora local y ambos incluidos; vacíos = todas las consultas."""
        if desde and hasta and desde > hasta:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                "La fecha 'desde' es posterior a 'hasta'.")
        content = svc.export_metrics_csv(desde, hasta)
        rango = f"{desde or 'inicio'}_a_{hasta or date.today()}"
        return Response(content.encode("utf-8"), media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="metricas_{rango}.csv"',
            "Cache-Control": "no-store",  # trae preguntas de estudiantes: sin copias en caché
        })

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Settings.from_env()
    if not cfg.admin_password:
        raise SystemExit(
            "Falta ADMIN_PASSWORD en el .env. El panel no arranca sin contraseña: "
            "va a poder cargar documentos y ver datos de estudiantes, así que no debe "
            "quedar abierto sin protección.")

    from app.adapters.embedders import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(cfg.embed_model, cfg.embed_query_prefix,
                                           cfg.embed_doc_prefix)
    from app.runtime import build_repository

    kb = KnowledgeBase.open(cfg, embedder)
    repo = build_repository(cfg)  # misma base que el bot (PostgreSQL si hay DATABASE_URL)
    app = build_app(cfg, AdminService(cfg, kb, repo))

    log.info("Panel disponible en http://%s:%s/panel (usuario: %s)",
            cfg.admin_host, cfg.admin_port, cfg.admin_user)
    uvicorn.run(app, host=cfg.admin_host, port=cfg.admin_port)


if __name__ == "__main__":
    main()
