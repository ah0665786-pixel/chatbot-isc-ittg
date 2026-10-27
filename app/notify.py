"""Avisos por correo a la coordinadora cuando el sistema escala una consulta
a atención humana (HU-11, UC-12). Usa smtplib de la librería estándar: cero
dependencias nuevas.

Modos (AVISOS_MODO en el .env):
  vista_previa  (por defecto) no envía nada: guarda cada aviso como archivo
                .eml en AVISOS_CARPETA (data/outbox/) para revisarlo antes.
  prueba        envía de verdad, pero a AVISOS_CORREO_PRUEBA (tu correo), no a
                la coordinadora. Si está vacío, NO envía: guarda el .eml.
  coordinadora  envía a COORDINADORA_EMAIL. Solo con este valor escrito a mano
                le llegan los avisos a la coordinadora.
  desactivado   no avisa (lo usan las pruebas de carga).

Cada consulta derivada tiene un tipo (ver TIPOS). Los tipos de
AVISOS_INMEDIATOS se avisan al momento; el resto va al resumen diario, que se
envía a la hora AVISOS_RESUMEN_HORA con el día anterior completo. Sin hora de
resumen, todo se avisa al momento.

Cada aviso (enviado, guardado o fallido) queda en el log y en la tabla
`avisos` (HU-11 CA-03). El aviso inmediato corre en un hilo aparte
(Notifier): un SMTP lento o caído NUNCA retrasa ni tumba la respuesta al
estudiante. Un aviso que no se pudo enviar también se guarda como .eml.

La contraseña SMTP (SMTP_PASSWORD) solo se le pasa a smtplib: no se escribe
en el log, en el .eml ni en la base de datos.
"""
from __future__ import annotations

import logging
import re
import smtplib
import ssl
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from html import escape
from pathlib import Path
from typing import Callable, Sequence

from app.config import Settings
from app.domain import Outcome, RagResult, Reason

log = logging.getLogger("notify")

SMTP_TIMEOUT = 10  # segundos por operación con el servidor de correo
SENDER_NAME = "Chatbot ISC – ITTG"
SUBJECT_PREFIX = "[Chatbot ISC]"
SUBJECT_QUESTION_CHARS = 60

VISTA_PREVIA, PRUEBA, COORDINADORA, DESACTIVADO = (
    "vista_previa", "prueba", "coordinadora", "desactivado")
MODOS = (VISTA_PREVIA, PRUEBA, COORDINADORA, DESACTIVADO)

# Estados que se guardan en la tabla avisos
ENVIADO, GUARDADO, FALLIDO = "enviado", "guardado", "fallido"
INMEDIATO, RESUMEN = "inmediato", "resumen"  # avisos.tipo_aviso
_FALLA_MOTIVOS = [Reason.LLM_UNAVAILABLE.value]


@dataclass(frozen=True)
class Tipo:
    clave: str        # la que se escribe en AVISOS_INMEDIATOS
    etiqueta: str     # asunto y etiqueta de color
    color: str        # texto de la etiqueta
    fondo: str        # fondo de la etiqueta
    resumen: str      # primera línea del aviso
    que_hacer: str


TIPOS = {t.clave: t for t in (
    Tipo("datos_personales", "Datos personales o trámite oficial", "#92400e", "#fef3c7",
         "Un estudiante pidió datos personales o un documento oficial, que el chatbot no maneja.",
         "Ya se le indicó acudir a Servicios Escolares. Solo requiere acción de la coordinación "
         "si el trámite le corresponde."),
    Tipo("sin_informacion", "Sin información en los documentos", "#1e40af", "#dbeafe",
         "Un estudiante hizo una pregunta que el chatbot no pudo responder con los documentos "
         "cargados.",
         "Si es una duda frecuente, carga o actualiza desde el panel el documento que la "
         "responda, para que el chatbot la conteste la próxima vez."),
    Tipo("falla_tecnica", "Falla técnica", "#991b1b", "#fee2e2",
         "El chatbot no pudo responder a un estudiante por una falla técnica.",
         "Avisa al responsable técnico para que revise el bot y el servicio de IA. Mientras "
         "tanto, los estudiantes reciben la indicación de acudir a la coordinación."),
)}

# Motivo de derivación -> tipo de aviso. INJECTION es un intento de ataque (no
# una consulta real que necesite su atención) y TOO_LONG es un error del
# propio mensaje del estudiante: ninguno de los dos se escala.
_REASON_TIPO = {
    Reason.OUT_OF_SCOPE.value: "datos_personales",
    Reason.LOW_SIMILARITY.value: "sin_informacion",
    Reason.NOT_IN_CONTEXT.value: "sin_informacion",
    Reason.VALIDATION_FAILED.value: "sin_informacion",
    Reason.LLM_UNAVAILABLE.value: "falla_tecnica",
}

NO_REPLY_NOTE = ("No respondas a este correo: el estudiante no lo recibe. El estudiante ya "
                 "recibió por Telegram la orientación que aparece arriba.")
SUMMARY_NO_REPLY_NOTE = ("No respondas a este correo: los estudiantes no lo reciben. Cada uno ya "
                         "recibió por Telegram la indicación de a dónde acudir.")
_DIAS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_MESES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
          "septiembre", "octubre", "noviembre", "diciembre")


def tipo_de(reason: Reason | str | None) -> Tipo | None:
    """Tipo de aviso de un motivo de derivación (None = no se escala)."""
    return TIPOS.get(_REASON_TIPO.get(getattr(reason, "value", reason)))


def should_notify(result: RagResult) -> bool:
    return tipo_de(result.reason) is not None


def summary_time(cfg: Settings) -> time | None:
    """Hora del resumen diario, o None si no hay resumen (vacía o mal escrita)."""
    try:
        h, m = cfg.avisos_resumen_hora.split(":")
        return time(int(h), int(m))
    except ValueError:
        return None


def is_immediate(cfg: Settings, tipo: Tipo) -> bool:
    return summary_time(cfg) is None or tipo.clave in cfg.avisos_inmediatos


def reaches_coordinadora(cfg: Settings) -> bool:
    """¿Los avisos le llegan de verdad a la coordinadora con esta configuración?"""
    return (cfg.avisos_modo == COORDINADORA and bool(cfg.coordinadora_email and cfg.smtp_host)
            and not (cfg.smtp_user and not cfg.smtp_password))


def notice_phrase(immediate: bool) -> str:
    """Frase para el estudiante. En futuro: el aviso sale después de su respuesta."""
    if immediate:
        inicio = "Le enviaremos un aviso con tu consulta a la Coordinación de ISC"
    else:
        inicio = "Tu consulta se incluirá en el resumen diario que recibe la Coordinación de ISC"
    return f"{inicio}; aun así, para recibir respuesta sigue la indicación de arriba."


def student_notice(cfg: Settings, reason: Reason) -> str:
    """Frase "tu consulta fue notificada", solo si es cierta: modo coordinadora
    con el correo configurado. En vista previa, prueba o desactivado: vacía."""
    tipo = tipo_de(reason)
    if tipo is None or not reaches_coordinadora(cfg):
        return ""
    return notice_phrase(is_immediate(cfg, tipo))


@dataclass(frozen=True)
class AvisoResultado:
    """Qué pasó con un aviso; es lo que se guarda en la tabla avisos."""
    estado: str          # enviado | guardado | fallido
    modo: str
    destinatario: str    # a quién se envió (o a quién iría, en vista previa)
    archivo: str = ""    # .eml guardado, si lo hubo
    detalle: str = ""    # advertencia o error


# ---- contenido -------------------------------------------------------------------
def _one_line(text: str, limit: int) -> str:
    """Para el asunto: sin saltos de línea ni caracteres de control (no se
    pueden colar cabeceras) y recortado."""
    text = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _html_text(text: str) -> str:
    """Texto del estudiante o del chatbot, escapado y con sus saltos de línea."""
    return escape(text or "").replace("\n", "<br>\n")


def _fecha_larga(day: date) -> str:
    return f"{_DIAS[day.weekday()]} {day.day} de {_MESES[day.month - 1]} de {day.year}"


def _banner(cfg: Settings, modo: str) -> str:
    if modo == PRUEBA:
        real = cfg.coordinadora_email or "(COORDINADORA_EMAIL vacío)"
        return f"AVISO DE PRUEBA: con AVISOS_MODO=coordinadora este correo le llegaría a {real}."
    if modo == VISTA_PREVIA:
        return "VISTA PREVIA: este aviso no se envió; se guardó para revisarlo."
    return ""


def _label(tipo: Tipo, extra: str = "") -> str:
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;'
            f'background:{tipo.fondo};color:{tipo.color};font-weight:bold;font-size:13px">'
            f'{escape(tipo.etiqueta)}{extra}</span>')


def _page(cfg: Settings, modo: str, body: str, tech: str, note: str = NO_REPLY_NOTE) -> str:
    banner = _banner(cfg, modo)
    banner_html = (f'<p style="margin:0 0 16px;padding:8px 12px;background:#f3f4f6;'
                   f'border:1px dashed #9ca3af;font-size:13px">{escape(banner)}</p>\n'
                   if banner else "")
    return (
        '<!doctype html>\n<html lang="es"><head><meta charset="utf-8"></head>\n'
        '<body style="margin:0;padding:16px;background:#ffffff">\n'
        '<div style="font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.45;'
        'color:#1f2937;max-width:640px">\n'
        f"{banner_html}{body}"
        '<p style="margin:0 0 16px;padding:10px 12px;background:#fffbeb;border:1px solid #fde68a;'
        f'font-size:13px">{escape(note)}</p>\n'
        '<hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0 8px">\n'
        f'<p style="margin:0;font-size:12px;color:#6b7280">{tech}</p>\n'
        "</div>\n</body></html>\n")


def _affected_line(affected: int, since: datetime, since_last: bool, minutes: int) -> str:
    desde = (f"desde el último aviso de falla técnica ({since.strftime('%d/%m/%Y %H:%M')})"
             if since_last else f"en los últimos {minutes} minutos")
    consultas = "1 consulta afectada" if affected == 1 else f"{affected} consultas afectadas"
    limite = (f" Mientras siga la falla, llega como máximo un aviso cada {minutes} minutos."
              if minutes else "")
    return f"{consultas.capitalize()} {desde}, contando esta.{limite}"


def render_notice(cfg: Settings, modo: str, telegram_id: str, display_name: str,
                  question: str, result: RagResult, metric_id: str | None,
                  now: datetime, affected: tuple[int, datetime, bool] | None = None
                  ) -> tuple[str, str, str]:
    """(asunto, texto plano, HTML) del aviso inmediato de una consulta.
    affected: (consultas afectadas, desde cuándo, si "desde" es el último aviso)."""
    tipo = tipo_de(result.reason)
    afectadas = (_affected_line(*affected, cfg.avisos_falla_intervalo_min)
                 if affected else "")
    subject = f"{SUBJECT_PREFIX} {tipo.etiqueta}: {_one_line(question, SUBJECT_QUESTION_CHARS)}"
    fecha = now.strftime("%d/%m/%Y %H:%M")
    tech = [("Folio", metric_id or "-"), ("ID de Telegram", telegram_id),
            ("Nombre", display_name or "(no proporcionado)"), ("Fecha", fecha),
            ("Motivo técnico", f"{result.reason.value}"
                               f"{' (' + result.detail + ')' if result.detail else ''}")]
    banner = _banner(cfg, modo)
    text = (
        (f"{banner}\n\n" if banner else "")
        + f"{tipo.resumen}\n\n" + (f"{afectadas}\n\n" if afectadas else "")
        + f"Tipo: {tipo.etiqueta}\n\n"
        f"Pregunta del estudiante:\n{question}\n\n"
        f"Respuesta que recibió por Telegram:\n{result.text}\n\n"
        f"Qué hacer: {tipo.que_hacer}\n\n{NO_REPLY_NOTE}\n\n--\n"
        + "\n".join(f"{k}: {v}" for k, v in tech) + "\n")
    small = "<p style=\"margin:0 0 4px;font-size:13px;color:#6b7280\">"
    body = (
        f'<p style="margin:0 0 12px;font-size:16px">{escape(tipo.resumen)}</p>\n'
        + (f'<p style="margin:0 0 12px;padding:8px 12px;background:{tipo.fondo};'
           f'color:{tipo.color};font-weight:bold">{escape(afectadas)}</p>\n'
           if afectadas else "")
        + f'<p style="margin:0 0 16px">{_label(tipo)}</p>\n'
        f"{small}Pregunta del estudiante</p>\n"
        f'<div style="margin:0 0 16px;padding:12px 16px;background:#f9fafb;'
        f'border-left:4px solid {tipo.color};font-size:17px;font-weight:bold">'
        f"{_html_text(question)}</div>\n"
        f"{small}Respuesta que recibió por Telegram</p>\n"
        '<div style="margin:0 0 16px;padding:12px 16px;border:1px solid #e5e7eb;font-size:14px">'
        f"{_html_text(result.text)}</div>\n"
        f'<p style="margin:0 0 16px"><strong>Qué hacer:</strong> {escape(tipo.que_hacer)}</p>\n')
    tech_html = " · ".join(f"{escape(k)}: {escape(str(v))}" for k, v in tech)
    return subject, text, _page(cfg, modo, body, tech_html)


def render_summary(cfg: Settings, modo: str, day: date, rows: Sequence, tz: tzinfo,
                   now: datetime) -> tuple[str, str, str]:
    """(asunto, texto plano, HTML) del resumen diario. rows: DerivationRow."""
    n = len(rows)
    consultas = "1 consulta" if n == 1 else f"{n} consultas"
    derivadas = "derivada" if n == 1 else "derivadas"
    subject = (f"{SUBJECT_PREFIX} Resumen diario: {consultas} {derivadas} el "
               f"{day.strftime('%d/%m/%Y')}")
    intro = f"El {_fecha_larga(day)} el chatbot derivó {consultas} a atención humana."
    by_tipo: dict[str, int] = {}
    items = []
    for r in rows:
        tipo = tipo_de(r.motivo_derivacion)
        by_tipo[tipo.clave] = by_tipo.get(tipo.clave, 0) + 1
        hora = datetime.fromisoformat(r.fecha).astimezone(tz).strftime("%H:%M")
        items.append((hora, tipo, r.pregunta or "(sin texto)", r.nombre or "-",
                      "Sí" if r.aviso_inmediato == ENVIADO else "No"))
    present = [TIPOS[k] for k in TIPOS if k in by_tipo]

    text = ((f"{_banner(cfg, modo)}\n\n" if _banner(cfg, modo) else "") + f"{intro}\n\n"
            + "".join(f"  {t.etiqueta}: {by_tipo[t.clave]}\n" for t in present)
            + "\nConsultas:\n"
            + "".join(f"  {h}  [{t.etiqueta}] {q}  (estudiante: {nm}; aviso inmediato: {av})\n"
                      for h, t, q, nm, av in items)
            + "\nQué hacer:\n"
            + "".join(f"  {t.etiqueta}: {t.que_hacer}\n" for t in present)
            + f"\n{SUMMARY_NO_REPLY_NOTE}\n\n--\n"
            f"Periodo: {day.isoformat()} de 00:00 a 24:00 (hora local)\n"
            f"Generado: {now.strftime('%d/%m/%Y %H:%M')}\n")

    cell = 'style="padding:6px 8px;border-bottom:1px solid #e5e7eb;vertical-align:top"'
    head = ('style="padding:6px 8px;border-bottom:2px solid #d1d5db;text-align:left;'
            'font-size:12px;color:#6b7280"')
    counts = " ".join(_label(t, f": {by_tipo[t.clave]}") for t in present)
    table = (
        '<table style="border-collapse:collapse;width:100%;font-size:14px;margin:0 0 16px">\n'
        f"<tr><th {head}>Hora</th><th {head}>Tipo</th><th {head}>Pregunta</th>"
        f"<th {head}>Estudiante</th><th {head}>Aviso inmediato</th></tr>\n"
        + "".join(
            f'<tr><td {cell}>{h}</td><td {cell}><span style="color:{t.color};font-weight:bold">'
            f"{escape(t.etiqueta)}</span></td><td {cell}>{_html_text(q)}</td>"
            f"<td {cell}>{escape(nm)}</td><td {cell}>{av}</td></tr>\n"
            for h, t, q, nm, av in items)
        + "</table>\n")
    todo = "".join(f"<li><strong>{escape(t.etiqueta)}:</strong> {escape(t.que_hacer)}</li>\n"
                   for t in present)
    body = (f'<p style="margin:0 0 12px;font-size:16px">{escape(intro)}</p>\n'
            f'<p style="margin:0 0 16px;line-height:2">{counts}</p>\n{table}'
            f'<p style="margin:0 0 4px"><strong>Qué hacer:</strong></p>\n'
            f'<ul style="margin:0 0 16px;padding-left:20px">\n{todo}</ul>\n')
    tech = (f"Periodo: {day.isoformat()} de 00:00 a 24:00 (hora local) · "
            f"Generado: {now.strftime('%d/%m/%Y %H:%M')}")
    return subject, text, _page(cfg, modo, body, escape(tech), SUMMARY_NO_REPLY_NOTE)


def _message(cfg: Settings, content: tuple[str, str, str], to: str, modo: str,
             now: datetime, cte: str | None = None) -> EmailMessage:
    subject, text, html = content
    msg = EmailMessage()
    msg["Subject"] = ("[PRUEBA] " if modo == PRUEBA else "") + subject
    # Gmail exige que la dirección sea la cuenta autenticada (SMTP_USER)
    msg["From"] = Address(SENDER_NAME, addr_spec=cfg.smtp_user or "chatbot-isc@noreply.local")
    if to:
        msg["To"] = to
    msg["Date"] = formatdate(now.timestamp(), localtime=True)
    msg["Message-ID"] = make_msgid(domain="chatbot-isc")
    msg["X-Chatbot-Modo"] = modo
    # En los archivos el texto va tal cual (8bit) y no como "envi=C3=B3", para
    # que también se lean en el Bloc de notas
    kw = {"cte": cte} if cte else {}
    msg.set_content(text, **kw)
    msg.add_alternative(html, subtype="html", **kw)
    return msg


# ---- envío -----------------------------------------------------------------------
def _recipient(cfg: Settings, modo: str) -> tuple[str, str]:
    """(destinatario, advertencia). Sin destinatario no se envía."""
    if modo == PRUEBA:
        if cfg.avisos_correo_prueba:
            return cfg.avisos_correo_prueba, ""
        return "", ("AVISOS_MODO=prueba pero AVISOS_CORREO_PRUEBA está vacío: "
                    "se guardó sin enviar")
    if cfg.coordinadora_email:
        return cfg.coordinadora_email, ""
    return "", "AVISOS_MODO=coordinadora pero COORDINADORA_EMAIL está vacío: se guardó sin enviar"


def _save(folder: Path, msg: EmailMessage, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(msg.as_bytes())
    return str(path)


def _send(cfg: Settings, msg: EmailMessage, to: str, ssl_context: ssl.SSLContext | None) -> None:
    with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=SMTP_TIMEOUT) as server:
        if cfg.smtp_use_tls:
            server.starttls(context=ssl_context or ssl.create_default_context())
        if cfg.smtp_user:
            server.login(cfg.smtp_user, cfg.smtp_password)
        server.send_message(msg, to_addrs=[to])


def _sin_secretos(text: str, cfg: Settings) -> str:
    """Por si un error del servidor repitiera la contraseña."""
    return text.replace(cfg.smtp_password, "***") if cfg.smtp_password else text


def _deliver(cfg: Settings, render: Callable[[str], tuple[str, str, str]], kind: str,
             ssl_context: ssl.SSLContext | None, now: datetime) -> AvisoResultado:
    """Envía o guarda según AVISOS_MODO. render(modo) arma el contenido. Nunca
    lanza excepción: el resultado dice qué pasó."""
    modo = cfg.avisos_modo
    aviso = ""
    if modo not in MODOS or modo == DESACTIVADO:
        aviso, modo = f"AVISOS_MODO={modo!r} no envía: se usó vista_previa", VISTA_PREVIA
    if modo in (PRUEBA, COORDINADORA):
        to, falta = _recipient(cfg, modo)
        if to and not cfg.smtp_host:
            falta = "falta SMTP_HOST: se guardó sin enviar"
        elif to and cfg.smtp_user and not cfg.smtp_password:
            falta = "falta SMTP_PASSWORD: se guardó sin enviar"
        aviso = falta or aviso
    else:
        to = cfg.coordinadora_email  # vista previa: a quién iría
    content = render(modo)
    name = f"{kind}_{now.strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.eml"

    if modo not in (PRUEBA, COORDINADORA) or aviso:
        try:
            path = _save(cfg.outbox_dir, _message(cfg, content, to, modo, now, "8bit"), name)
        except OSError as e:
            log.error("no se pudo guardar el %s en %s: %s", kind, cfg.outbox_dir, e)
            return AvisoResultado(FALLIDO, modo, to, "", f"no se pudo guardar: {e}")
        if aviso:
            log.warning("%s NO enviado (%s); guardado en %s", kind, aviso, path)
        else:
            log.info("%s guardado en %s (vista previa, no se envió)", kind, path)
        return AvisoResultado(GUARDADO, modo, to, path, aviso)

    try:
        _send(cfg, _message(cfg, content, to, modo, now), to, ssl_context)
    except Exception as e:  # noqa: BLE001 — nunca debe tumbar la respuesta al estudiante
        error = _sin_secretos(f"{type(e).__name__}: {e}", cfg)
        try:
            path = _save(cfg.outbox_dir, _message(cfg, content, to, modo, now, "8bit"), name)
        except OSError:
            path = ""
        log.warning("no se pudo enviar el %s a %s (modo %s): %s; copia en %s",
                    kind, to, modo, error, path or "(no se pudo guardar)")
        return AvisoResultado(FALLIDO, modo, to, path, error)
    log.info("%s enviado a %s (modo %s)", kind, to, modo)
    return AvisoResultado(ENVIADO, modo, to)


def notify_coordinadora(cfg: Settings, telegram_id: str, display_name: str,
                        question: str, result: RagResult, metric_id: str | None = None,
                        ssl_context: ssl.SSLContext | None = None,
                        now: datetime | None = None,
                        affected: tuple[int, datetime, bool] | None = None) -> AvisoResultado:
    """Aviso inmediato de una consulta (síncrono: lo llama el hilo de Notifier)."""
    now = (now or datetime.now(timezone.utc)).astimezone()
    return _deliver(cfg, lambda modo: render_notice(cfg, modo, telegram_id, display_name,
                                                    question, result, metric_id, now,
                                                    affected),
                    "aviso", ssl_context, now)


def notify_summary(cfg: Settings, day: date, rows: Sequence, tz: tzinfo,
                   ssl_context: ssl.SSLContext | None = None,
                   now: datetime | None = None) -> AvisoResultado:
    now = now or datetime.now().astimezone()
    return _deliver(cfg, lambda modo: render_summary(cfg, modo, day, rows, tz, now),
                    "resumen", ssl_context, now)


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


class Notifier:
    """Hace los avisos inmediatos en un hilo aparte, manda el resumen diario y
    registra cada resultado en la tabla avisos. Un solo hilo para los
    inmediatos: salen en orden y un SMTP caído no abre una conexión por
    estudiante."""

    def __init__(self, cfg: Settings, repo=None, ssl_context: ssl.SSLContext | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        self.cfg, self.repo, self.ssl_context = cfg, repo, ssl_context
        self._clock = clock  # reemplazable en pruebas
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avisos")
        self._pending: set[Future] = set()
        self._stop = threading.Event()
        self._summary_at = summary_time(cfg)
        self._empty_days: set[str] = set()  # días ya revisados sin derivaciones

    def submit(self, metric_id: str | None, telegram_id: str, display_name: str,
               question: str, result: RagResult) -> Future | None:
        """Aviso inmediato si el tipo lo pide; si no, la consulta espera al
        resumen diario (se toma de la base a la hora del resumen)."""
        tipo = tipo_de(result.reason)
        if self.cfg.avisos_modo == DESACTIVADO or tipo is None:
            return None
        if not is_immediate(self.cfg, tipo):
            log.info("consulta %s (%s): va al resumen diario", metric_id, tipo.clave)
            return None
        fut = self._pool.submit(self._run, metric_id, telegram_id, display_name, question,
                                result)
        self._pending.add(fut)
        fut.add_done_callback(self._pending.discard)
        return fut

    def _falla_window(self, now: datetime) -> tuple[int, datetime, bool] | None | bool:
        """Límite de avisos de falla técnica, con el estado en la base (sobrevive a
        un reinicio): False si dentro de la ventana ya hubo aviso; si no, el
        conteo de consultas afectadas para el aviso que sale."""
        minutes = self.cfg.avisos_falla_intervalo_min
        if minutes <= 0 or self.repo is None:
            return None
        window = timedelta(minutes=minutes)
        last = self.repo.last_notice_at(_FALLA_MOTIVOS)
        last_dt = datetime.fromisoformat(last) if last else None
        if last_dt and now - last_dt < window:
            return False
        since = last_dt or now - window
        n = self.repo.count_derived(_FALLA_MOTIVOS, _utc_iso(since), _utc_iso(now))
        return max(n, 1), since.astimezone(), last_dt is not None

    def _run(self, metric_id, telegram_id, display_name, question,
             result) -> AvisoResultado | None:
        now = self._clock()
        affected = None
        if tipo_de(result.reason).clave == "falla_tecnica":
            affected = self._falla_window(now)
            if affected is False:
                log.info("falla técnica (consulta %s): sin aviso nuevo, ya hubo uno hace menos "
                         "de %d min; se contará en el siguiente", metric_id,
                         self.cfg.avisos_falla_intervalo_min)
                return None
        try:
            res = notify_coordinadora(self.cfg, telegram_id, display_name, question, result,
                                      metric_id, self.ssl_context, now, affected)
        except Exception as e:  # noqa: BLE001 — p. ej. un error al armar el mensaje
            log.exception("fallo inesperado al preparar el aviso")
            res = AvisoResultado(FALLIDO, self.cfg.avisos_modo, "", "", type(e).__name__)
        if self.repo is not None:
            try:
                self.repo.record_notice(metric_id, res.estado, res.modo, res.destinatario,
                                        res.archivo, res.detalle, result.reason.value,
                                        now=now, tipo_aviso=INMEDIATO)
            except Exception:  # noqa: BLE001
                log.exception("no se pudo registrar el aviso en la base de datos")
        return res

    # ---- resumen diario ----------------------------------------------------------
    def send_daily_summary(self, day: date, tz: tzinfo,
                           now: datetime | None = None) -> AvisoResultado | None:
        """Resumen de las derivaciones escalables de `day` (00:00 a 24:00 en tz).
        No hace nada si ya se procesó ese día o si no hubo derivaciones."""
        key = day.isoformat()
        if key in self._empty_days or self.repo.summary_sent(key):
            return None
        start = datetime.combine(day, time.min, tzinfo=tz)
        rows = [r for r in self.repo.derived_between(_utc_iso(start),
                                                     _utc_iso(start + timedelta(days=1)))
                if tipo_de(r.motivo_derivacion)]
        if not rows:  # un día que ya pasó no recibe derivaciones nuevas
            log.info("resumen del %s: sin consultas derivadas, no se envía", key)
            self._empty_days.add(key)
            return None
        res = notify_summary(self.cfg, day, rows, tz, self.ssl_context, now)
        for r in rows:  # una fila por consulta: el CSV muestra en qué resumen fue
            self.repo.record_notice(r.id_metrica, res.estado, res.modo, res.destinatario,
                                    res.archivo, res.detalle, r.motivo_derivacion,
                                    tipo_aviso=RESUMEN, fecha_resumen=key)
        return res

    def run_due_summary(self, now: datetime | None = None) -> AvisoResultado | None:
        """A partir de la hora del resumen, manda el del día anterior si falta."""
        if self.cfg.avisos_modo == DESACTIVADO or self._summary_at is None or self.repo is None:
            return None
        now = now or datetime.now().astimezone()  # TZ del contenedor (Tuxtla)
        if now.time() < self._summary_at:
            return None
        return self.send_daily_summary(now.date() - timedelta(days=1), now.tzinfo, now)

    def start_daily_summary(self, interval: float = 60.0) -> bool:
        """Revisa cada minuto si toca mandar el resumen (lo arranca el bot)."""
        if self.cfg.avisos_resumen_hora and self._summary_at is None:
            log.warning("AVISOS_RESUMEN_HORA=%r no es HH:MM: no hay resumen diario y todos los "
                        "tipos se avisan al momento", self.cfg.avisos_resumen_hora)
        if self.cfg.avisos_modo == DESACTIVADO or self._summary_at is None or self.repo is None:
            return False

        def loop() -> None:
            while True:
                try:
                    self.run_due_summary()
                except Exception:  # noqa: BLE001 — el hilo no debe morir
                    log.exception("fallo al preparar el resumen diario")
                if self._stop.wait(interval):
                    return

        threading.Thread(target=loop, name="resumen-diario", daemon=True).start()
        log.info("resumen diario a las %s con el día anterior; avisos al momento: %s",
                 self._summary_at.strftime("%H:%M"), ", ".join(self.cfg.avisos_inmediatos) or "ninguno")
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Espera a que terminen los avisos pendientes (pruebas y CLI)."""
        _, pending = wait(list(self._pending), timeout)
        return not pending

    def close(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=True)


# ---- ejemplos para revisar el formato (python -m app.cli ejemplos-avisos) ------------
def write_examples(cfg: Settings, folder: Path) -> list[str]:
    """Un .eml de ejemplo por tipo y uno del resumen diario, con datos
    inventados. No toca la base de datos ni las métricas."""
    from app.adapters.db_sqlite import DerivationRow
    from app.rag.messages import derivation_message

    now = datetime.now().astimezone()
    cfg = replace(cfg, avisos_modo=COORDINADORA)  # tal como le llegaría a la coordinadora
    to = cfg.coordinadora_email or "coordinadora@ejemplo.mx"
    casos = [
        ("datos_personales", Reason.OUT_OF_SCOPE, "datos_personales",
         "¿Me pueden dar mi kárdex con las calificaciones del semestre pasado?"),
        ("sin_informacion", Reason.LOW_SIMILARITY, "max=0.771 < umbral 0.8",
         "¿Cuándo es la fecha límite para pagar la reinscripción de enero?"),
        ("falla_tecnica", Reason.LLM_UNAVAILABLE, "todos los proveedores fallaron: groq (timeout)",
         "¿Cuántas horas dura el servicio social?"),
    ]
    window = timedelta(minutes=cfg.avisos_falla_intervalo_min or 30)
    falla = {  # (consultas afectadas, desde, desde el último aviso)
        "falla_tecnica": (1, now - window, False),
        "falla_tecnica_agrupada": (7, now - window - timedelta(minutes=4), True),
    }
    casos.append(("falla_tecnica_agrupada", *casos[-1][1:]))
    paths = []
    for clave, reason, detail, question in casos:
        texto = derivation_message(reason, cfg, "datos_personales"
                                   if reason is Reason.OUT_OF_SCOPE else None)
        texto += "\n\n" + notice_phrase(True)  # si llega como aviso inmediato, es inmediato
        result = RagResult(Outcome.DERIVED, reason, texto, detail=detail)
        content = render_notice(cfg, COORDINADORA, "123456789", "Estudiante de ejemplo",
                                question, result, f"ejemplo-{clave}", now, falla.get(clave))
        paths.append(_save(folder, _message(cfg, content, to, COORDINADORA, now, "8bit"),
                           f"ejemplo_{clave}.eml"))
    ayer = now.date() - timedelta(days=1)

    def at(h, m):
        return _utc_iso(datetime.combine(ayer, time(h, m), tzinfo=now.tzinfo))

    rows = [
        DerivationRow("e1", at(9, 12), "¿Cuándo es la fecha límite para pagar la reinscripción "
                      "de enero?", "", Reason.LOW_SIMILARITY.value, "1", "Ana", None),
        DerivationRow("e2", at(10, 40), "¿Me pueden dar mi kárdex?", "",
                      Reason.OUT_OF_SCOPE.value, "2", "Luis", None),
        DerivationRow("e3", at(13, 5), "¿Puedo hacer la residencia en otra ciudad?", "",
                      Reason.NOT_IN_CONTEXT.value, "3", "María", None),
        DerivationRow("e4", at(17, 58), "¿Cuántas horas dura el servicio social?", "",
                      Reason.LLM_UNAVAILABLE.value, "4", "Jorge", ENVIADO),
    ]
    content = render_summary(cfg, COORDINADORA, ayer, rows, now.tzinfo, now)
    paths.append(_save(folder, _message(cfg, content, to, COORDINADORA, now, "8bit"),
                       "ejemplo_resumen_diario.eml"))
    return paths

