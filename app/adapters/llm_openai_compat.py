"""Cliente LLM para cualquier API "OpenAI-compatible" (Groq, OpenAI, Gemini
compat, Ollama, vLLM...). Cambiar de proveedor = cambiar 3 variables de entorno.

Incluye lo que exige RNF-FI-02: reintentos con espera exponencial y tope de
espera total; y detecta modelos retirados para no reintentar en vano.

El tope es una fecha límite (reloj monotónico) que se comparte entre todos los
reintentos, todos los proveedores y los 2 intentos del pipeline: ninguna
espera ni petición se pasa de ella, así que la consulta completa respeta el
tope aunque cada pieza tenga su propio timeout.

Límite de uso (HTTP 429): no se reintenta a ciegas. El proveedor dice cuánto
esperar (encabezado retry-after o x-ratelimit-reset-*); se pasa al siguiente
proveedor y, si todos están en su límite, la consulta espera en una fila
(RateLimitQueue) dentro del mismo tope. Si la fila está llena o la espera no
cabe, se rinde con HighDemand ("alta demanda"), que no es una falla técnica."""
from __future__ import annotations

import contextvars
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Callable, Mapping, Sequence

import httpx

from app.config import LLMEndpoint
from app.ports import LLMReply

log = logging.getLogger("llm")

_RETIRED_MARKERS = ("decommission", "model_not_found", "does not exist",
                    "not found", "deprecated", "no longer supported")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """Fallo genérico del proveedor."""


class ModelUnavailable(LLMError):
    """El modelo ya no existe (retirado). No tiene caso reintentar."""


class LLMUnavailable(LLMError):
    """Ningún proveedor de la cadena pudo responder."""


class RateLimited(LLMError):
    """El proveedor devolvió límite de uso (429). retry_after: segundos que pide esperar."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class HighDemand(LLMUnavailable):
    """Todos los proveedores están en su límite de uso y la consulta no pudo
    esperar más (fila llena, espera vencida o sin tiempo en el tope). Es alta
    demanda, no una falla técnica: se resuelve sola en unos minutos."""


# Espera cuando un 429 no trae encabezados que digan cuánto esperar
DEFAULT_RETRY_AFTER = 5.0
# Mínimo que se salta a un proveedor tras un 429 (un retry-after de 0 no debe
# provocar un ciclo de peticiones seguidas)
MIN_COOLDOWN = 1.0
# Un proveedor con falla técnica (clave inválida, caído) no cuenta como "por
# liberarse" durante este tiempo: la fila espera al que está en su límite de uso
# en vez de dar vueltas pidiéndole al que está roto. Se le sigue intentando.
BROKEN_HOLD = 30.0
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(value: str | None) -> float | None:
    """"7.66s", "2m59.56s", "1h2m", "120ms" o "2" (segundos) -> segundos."""
    v = (value or "").strip().lower()
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    parts = _DURATION.findall(v)
    if not parts or "".join(n + u for n, u in parts) != v:
        return None
    mult = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * mult[u] for n, u in parts)


def retry_after_seconds(headers: Mapping[str, str], now: Callable[[], float] = time.time,
                        default: float = DEFAULT_RETRY_AFTER) -> float:
    """Cuánto pide esperar un 429. Primero retry-after (segundos o fecha HTTP);
    si no viene, el reinicio del límite que se agotó (x-ratelimit-reset-tokens /
    -requests, formato de Groq y OpenAI: "7.66s", "2m59.56s")."""
    h = {k.lower(): v for k, v in headers.items()}
    ra = h.get("retry-after", "").strip()
    if ra:
        secs = parse_duration(ra)
        if secs is None:
            try:
                secs = max(0.0, parsedate_to_datetime(ra).timestamp() - now())
            except (TypeError, ValueError):
                secs = None
        if secs is not None:
            return secs
    resets = {}
    for kind in ("tokens", "requests"):
        secs = parse_duration(h.get(f"x-ratelimit-reset-{kind}"))
        if secs is not None:
            resets[kind] = secs
    exhausted = [resets[k] for k in resets
                 if (parse_duration(h.get(f"x-ratelimit-remaining-{k}")) or 0) <= 0
                 and h.get(f"x-ratelimit-remaining-{k}") is not None]
    if exhausted:
        return max(exhausted)
    if resets:
        return min(resets.values())
    return default


class OpenAICompatLLM:
    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 25.0, max_retries: int = 3, extra: dict | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.name, self.model, self.extra = name, model, dict(extra or {})
        self.max_retries = max(1, max_retries)
        self.timeout = timeout
        self._sleep, self._clock = sleep, clock
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers=headers, transport=transport)

    # ---- utilidades -------------------------------------------------------
    def list_models(self) -> list[str]:
        r = self._client.get("/models")
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    # ---- chat -------------------------------------------------------------
    def chat(self, messages: Sequence[dict], temperature: float = 0.3,
             max_tokens: int = 900, deadline: float | None = None) -> LLMReply:
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": temperature, "max_tokens": max_tokens, **self.extra}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            timeout = self.timeout
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    break
                timeout = min(timeout, remaining)  # un intento no se come el tope
            try:
                r = self._client.post("/chat/completions", json=payload, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = LLMError(f"{self.name}: red/timeout ({type(e).__name__})")
            else:
                if r.status_code == 200:
                    return LLMReply(self._extract(r), self.name, self.model)
                body = r.text[:500].lower()
                if r.status_code in (400, 404) and any(m in body for m in _RETIRED_MARKERS):
                    raise ModelUnavailable(f"{self.name}/{self.model}: {r.status_code} {body}")
                if r.status_code in (401, 403):
                    raise LLMError(f"{self.name}: credenciales inválidas ({r.status_code})")
                if r.status_code == 429:
                    if "insufficient_quota" in body:  # sin saldo (OpenAI): no se arregla esperando
                        raise LLMError(f"{self.name}: sin saldo o cuota agotada (429)")
                    wait = retry_after_seconds(r.headers)
                    raise RateLimited(f"{self.name}: HTTP 429, límite de uso (esperar {wait:.1f} s)",
                                      wait)
                if r.status_code >= 500:
                    last = LLMError(f"{self.name}: HTTP {r.status_code}")
                else:
                    raise LLMError(f"{self.name}: HTTP {r.status_code} {body}")
            if attempt < self.max_retries - 1:
                wait = 2 ** attempt  # 1 s, 2 s, 4 s...
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - self._clock()))
                self._sleep(wait)
        raise last or LLMError(f"{self.name}: sin respuesta dentro del tiempo permitido")

    @staticmethod
    def _extract(r: httpx.Response) -> str:
        try:
            content = r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"respuesta con formato inesperado: {e}") from e
        return _THINK.sub("", content).strip()  # algunos modelos devuelven <think>


@dataclass(frozen=True)
class QueueInfo:
    """Lo que se le dice a quien escucha la fila (el bot avisa al estudiante)."""
    position: int          # 1 = la siguiente en pasar
    estimated_wait: float  # segundos hasta que un proveedor vuelve a aceptar (aprox.)


# Quien atiende la consulta (ConversationService) pone aquí una función que se
# llama UNA vez, cuando la consulta entra a la fila. Es un ContextVar para que
# cada hilo (asyncio.to_thread copia el contexto) avise a su propio estudiante.
queue_listener: contextvars.ContextVar[Callable[[QueueInfo], None] | None] = (
    contextvars.ContextVar("queue_listener", default=None))


@dataclass(frozen=True)
class QueuePolicy:
    """Límites de la fila (LLM_COLA_* en el .env).
    max_len: consultas esperando a la vez; 0 = sin fila (alta demanda al momento).
    max_wait: segundos que una consulta puede esperar en la fila.
    reserve: segundos del tope que deben quedar para generar después de esperar."""
    max_len: int = 5
    max_wait: float = 60.0
    reserve: float = 15.0


class _Ticket:
    __slots__ = ("joined",)

    def __init__(self, joined: float):
        self.joined = joined


class FallbackLLM:
    """Prueba los proveedores en orden. Si el modelo de uno fue retirado, lo
    marca como muerto durante el proceso y avisa fuerte en el log.

    Si un proveedor devuelve límite de uso, se "enfría" el tiempo que pidió y
    se salta mientras tanto. Si todos están enfriándose, la consulta espera en
    una fila FIFO: pasa una a la vez cuando un proveedor vuelve a aceptar (si
    pasaran todas juntas, se volverían a topar con el límite). La espera cuenta
    dentro del mismo tope total (RNF-FI-02). La fila vive en memoria: si el bot
    se reinicia, lo que estaba esperando se pierde."""

    def __init__(self, providers: Sequence[OpenAICompatLLM], total_budget: float = 80.0,
                 clock: Callable[[], float] = time.monotonic,
                 queue: QueuePolicy | None = None,
                 wait: Callable[[threading.Condition, float], None] | None = None):
        self.providers = list(providers)
        self.total_budget = total_budget
        self.queue = queue or QueuePolicy()
        self._clock = clock
        self._dead: set[str] = set()
        self._cooling: dict[str, float] = {}  # proveedor -> instante en que vuelve a aceptar
        self._broken: dict[str, float] = {}   # proveedor -> hasta cuándo no se le espera
        self._cond = threading.Condition()
        self._line: deque[_Ticket] = deque()
        # Cómo se espera en la fila; las pruebas lo cambian por uno que solo
        # adelanta un reloj falso.
        self._wait = wait or (lambda cond, seconds: cond.wait(seconds))

    # ---- estado de los proveedores (con self._cond tomado) -------------------
    def _ready_at(self, now: float) -> float | None:
        """Instante en que algún proveedor vivo vuelve a aceptar; None si alguno
        ya acepta (o si no queda ninguno: eso no es límite de uso, es falla).
        No cuenta los que tuvieron una falla técnica hace poco."""
        live = [p for p in self.providers if p.name not in self._dead
                and self._broken.get(p.name, 0.0) <= now]
        if not live:
            return None
        soonest = min(self._cooling.get(p.name, 0.0) for p in live)
        return soonest if soonest > now else None

    @property
    def queued(self) -> int:
        """Consultas esperando en la fila ahora mismo."""
        with self._cond:
            return len(self._line)

    # ---- fila (con self._cond tomado) -----------------------------------------
    def _limit(self, joined: float, deadline: float) -> float:
        """Hasta cuándo puede esperar una consulta que entró a la fila en `joined`."""
        return min(joined + self.queue.max_wait, deadline - self.queue.reserve)

    def _join(self, deadline: float) -> tuple[_Ticket, QueueInfo]:
        now = self._clock()
        if self.queue.max_len <= 0:
            raise HighDemand("límite de uso y la fila está desactivada (LLM_COLA_MAX=0)")
        if len(self._line) >= self.queue.max_len:
            raise HighDemand(f"fila llena ({len(self._line)} consultas esperando)")
        ready = self._ready_at(now)
        estimated = (ready - now) if ready else 0.0
        limit = self._limit(now, deadline)
        if now + estimated > limit:
            # p. ej. se agotó el límite DIARIO: esperar no sirve, mejor decirlo ya
            raise HighDemand(f"el proveedor pide esperar {estimated:.0f} s y la espera "
                             f"permitida es de {max(0.0, limit - now):.0f} s")
        ticket = _Ticket(now)
        self._line.append(ticket)
        return ticket, QueueInfo(len(self._line), estimated)

    def _await_turn(self, ticket: _Ticket, deadline: float) -> None:
        """Espera a ser la primera de la fila con un proveedor que acepte."""
        while True:
            now = self._clock()
            ready = self._ready_at(now)
            first = self._line[0] is ticket
            if first and ready is None:
                return
            limit = self._limit(ticket.joined, deadline)
            if now >= limit:
                raise HighDemand(f"espera en la fila vencida ({now - ticket.joined:.0f} s)")
            timeout = limit - now
            if first and ready is not None:
                timeout = min(timeout, ready - now)
            self._wait(self._cond, max(timeout, 0.0))

    def _leave(self, ticket: _Ticket) -> None:
        with self._cond:
            try:
                self._line.remove(ticket)
            except ValueError:
                pass
            self._cond.notify_all()  # la siguiente revisa si ya es su turno

    # ---- chat --------------------------------------------------------------------
    def chat(self, messages: Sequence[dict], temperature: float = 0.3,
             max_tokens: int = 900, deadline: float | None = None) -> LLMReply:
        """Sin deadline, el tope cuenta desde esta llamada. El pipeline pasa el
        suyo para que sus 2 intentos compartan un mismo tope (RNF-FI-02)."""
        own = self._clock() + self.total_budget
        deadline = own if deadline is None else min(deadline, own)
        ticket: _Ticket | None = None
        try:
            while True:
                info = None
                with self._cond:
                    # Si ya hay fila, se forma aunque un proveedor esté libre: no
                    # se adelanta a quien llegó antes.
                    if ticket is None and (self._line or self._ready_at(self._clock())):
                        ticket, info = self._join(deadline)
                if info is not None:
                    self._announce(info)
                if ticket is not None:
                    with self._cond:
                        self._await_turn(ticket, deadline)
                reply = self._try_providers(messages, temperature, max_tokens, deadline)
                if reply is not None:
                    return reply
                # Todos en su límite de uso: a la fila (o a seguir en ella)
                if ticket is None and self.queue.max_len <= 0:
                    raise HighDemand("límite de uso y la fila está desactivada (LLM_COLA_MAX=0)")
        finally:
            if ticket is not None:
                self._leave(ticket)

    @staticmethod
    def _announce(info: QueueInfo) -> None:
        log.info("consulta en fila: lugar %d, espera estimada %.1f s",
                 info.position, info.estimated_wait)
        listener = queue_listener.get()
        if listener is not None:
            try:
                listener(info)
            except Exception:  # noqa: BLE001 — el aviso nunca tumba la consulta
                log.exception("no se pudo avisar que la consulta está en fila")

    def _try_providers(self, messages, temperature, max_tokens, deadline) -> LLMReply | None:
        """La respuesta del primer proveedor que conteste; None si los que no
        fallaron están en su límite de uso; LLMUnavailable si todos fallaron."""
        errors: list[str] = []
        limited = False
        for p in self.providers:
            if p.name in self._dead:
                continue
            with self._cond:
                cooling = self._cooling.get(p.name, 0.0) > self._clock()
            if cooling:
                limited = True
                continue
            try:
                reply = p.chat(messages, temperature, max_tokens, deadline=deadline)
                with self._cond:
                    self._broken.pop(p.name, None)
                return reply
            except RateLimited as e:
                with self._cond:
                    self._cooling[p.name] = self._clock() + max(e.retry_after, MIN_COOLDOWN)
                log.warning("proveedor %s en su límite de uso: %s", p.name, e)
                limited = True
            except ModelUnavailable as e:
                self._dead.add(p.name)
                log.critical("MODELO RETIRADO, cambiar en .env: %s", e)
                errors.append(str(e))
            except LLMError as e:
                log.warning("proveedor %s falló: %s", p.name, e)
                errors.append(str(e))
                with self._cond:
                    self._broken[p.name] = self._clock() + BROKEN_HOLD
        if limited:
            return None
        raise LLMUnavailable("; ".join(errors) or "no hay proveedores configurados")


def build_llm(endpoints: Sequence[LLMEndpoint], timeout: float, max_retries: int,
              total_budget: float, queue: QueuePolicy | None = None) -> FallbackLLM:
    return FallbackLLM(
        [OpenAICompatLLM(e.name, e.base_url, e.api_key, e.model, timeout, max_retries, e.extra)
         for e in endpoints],
        total_budget, queue=queue,
    )
