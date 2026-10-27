"""Servidor SMTP simulado para las pruebas del aviso por correo: escucha en
127.0.0.1 en un hilo, sin salir a la red. Entiende lo que usa smtplib con
Gmail (EHLO, STARTTLS, AUTH PLAIN/LOGIN, MAIL, RCPT, DATA, QUIT) y guarda lo
que recibe para revisarlo en la prueba. Con `delay` tarda en saludar, como un
SMTP que no responde."""
from __future__ import annotations

import base64
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
# Certificado autofirmado SOLO de prueba (CN=localhost, IP 127.0.0.1); no sirve
# para nada fuera de estas pruebas.
CERT = FIXTURES / "smtp_test_cert.pem"
KEY = FIXTURES / "smtp_test_key.pem"


def client_tls_context() -> ssl.SSLContext:
    """Contexto del lado del cliente que confía solo en el certificado de prueba."""
    return ssl.create_default_context(cafile=str(CERT))


@dataclass
class Received:
    mail_from: str
    rcpt_to: list[str]
    data: bytes
    tls: bool
    auth_user: str | None

    @property
    def message(self):
        return message_from_bytes(self.data, policy=policy.default)


@dataclass
class FakeSMTPServer:
    delay: float = 0.0                 # segundos antes de saludar
    starttls: bool = False             # anuncia STARTTLS (con el certificado de prueba)
    password: str | None = None        # si tiene valor, exige AUTH con esa contraseña
                                       # (sin valor, AUTH es opcional y acepta cualquiera)
    reject_rcpt: bool = False          # responde 550 al destinatario
    received: list[Received] = field(default_factory=list)
    auth_attempts: list[str] = field(default_factory=list)  # usuarios que intentaron AUTH

    def __post_init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "FakeSMTPServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(5)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    def _session(self, conn: socket.socket) -> None:
        conn.settimeout(10)
        tls, authed, auth_user = False, self.password is None, None
        mail_from, rcpts = "", []
        rfile = conn.makefile("rb")

        def send(line: str) -> None:
            conn.sendall(line.encode() + b"\r\n")

        def readline() -> str:
            return rfile.readline().decode(errors="replace").rstrip("\r\n")

        try:
            time.sleep(self.delay)
            send("220 fake.smtp.local ESMTP listo")
            while True:
                line = readline()
                if not line and rfile.closed:
                    return
                cmd = line.split(" ", 1)[0].upper()
                arg = line[len(cmd):].strip()
                if cmd in ("EHLO", "HELO"):
                    ext = []
                    if self.starttls and not tls:
                        ext.append("STARTTLS")  # como Gmail: AUTH solo después de TLS
                    else:
                        ext.append("AUTH PLAIN LOGIN")
                    lines = ["fake.smtp.local", *ext]
                    for i, l in enumerate(lines):
                        send(("250 " if i == len(lines) - 1 else "250-") + l)
                elif cmd == "STARTTLS":
                    send("220 inicia TLS")
                    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    ctx.load_cert_chain(str(CERT), str(KEY))
                    conn = ctx.wrap_socket(conn, server_side=True)
                    rfile = conn.makefile("rb")
                    tls = True
                elif cmd == "AUTH":
                    mech, _, initial = arg.partition(" ")
                    if mech.upper() == "PLAIN":
                        if not initial:
                            send("334 ")
                            initial = readline()
                        _, user, pwd = base64.b64decode(initial).decode().split("\0")
                    else:  # LOGIN (smtplib manda el usuario junto con el comando)
                        if not initial:
                            send("334 " + base64.b64encode(b"Username:").decode())
                            initial = readline()
                        user = base64.b64decode(initial).decode()
                        send("334 " + base64.b64encode(b"Password:").decode())
                        pwd = base64.b64decode(readline()).decode()
                    self.auth_attempts.append(user)
                    if self.password is None or pwd == self.password:
                        authed, auth_user = True, user
                        send("235 autenticado")
                    else:
                        send("535 credenciales incorrectas")
                elif cmd == "MAIL":
                    if not authed:
                        send("530 autenticacion requerida")
                        continue
                    mail_from, rcpts = arg.split(":", 1)[1].strip().strip("<>"), []
                    send("250 OK")
                elif cmd == "RCPT":
                    if self.reject_rcpt:
                        send("550 buzon no existe")
                        continue
                    rcpts.append(arg.split(":", 1)[1].strip().strip("<>"))
                    send("250 OK")
                elif cmd == "DATA":
                    send("354 termina con <CRLF>.<CRLF>")
                    chunks = []
                    while (raw := rfile.readline()) not in (b".\r\n", b""):
                        chunks.append(raw[1:] if raw.startswith(b"..") else raw)
                    self.received.append(Received(mail_from, rcpts, b"".join(chunks), tls,
                                                  auth_user))
                    send("250 recibido")
                elif cmd in ("RSET", "NOOP"):
                    send("250 OK")
                elif cmd == "QUIT":
                    send("221 adios")
                    return
                elif not line:
                    return
                else:
                    send("502 no implementado")
        except (OSError, ssl.SSLError, ValueError):
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass
