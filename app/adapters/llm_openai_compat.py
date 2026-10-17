"""Cliente LLM para cualquier API "OpenAI-compatible" (Groq, OpenAI, Gemini
compat, Ollama, vLLM...). Cambiar de proveedor = cambiar 3 variables de entorno.

Incluye lo que exige RNF-FI-02: reintentos con espera exponencial y tope de
espera total; y detecta modelos retirados para no reintentar en vano."""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Sequence

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


class OpenAICompatLLM:
    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 25.0, max_retries: int = 3, extra: dict | None = None,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.name, self.model, self.extra = name, model, dict(extra or {})
        self.max_retries = max(1, max_retries)
        self._sleep = sleep
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
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                r = self._client.post("/chat/completions", json=payload)
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
                if r.status_code == 429 or r.status_code >= 500:
                    last = LLMError(f"{self.name}: HTTP {r.status_code}")
                else:
                    raise LLMError(f"{self.name}: HTTP {r.status_code} {body}")
            if attempt < self.max_retries - 1:
                self._sleep(2 ** attempt)  # 1 s, 2 s, 4 s...
        raise last or LLMError(f"{self.name}: sin respuesta dentro del tiempo permitido")

    @staticmethod
    def _extract(r: httpx.Response) -> str:
        try:
            content = r.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"respuesta con formato inesperado: {e}") from e
        return _THINK.sub("", content).strip()  # algunos modelos devuelven <think>


class FallbackLLM:
    """Prueba los proveedores en orden. Si el modelo de uno fue retirado, lo
    marca como muerto durante el proceso y avisa fuerte en el log."""

    def __init__(self, providers: Sequence[OpenAICompatLLM], total_budget: float = 80.0):
        self.providers = list(providers)
        self.total_budget = total_budget
        self._dead: set[str] = set()

    def chat(self, messages: Sequence[dict], temperature: float = 0.3,
             max_tokens: int = 900) -> LLMReply:
        deadline = time.monotonic() + self.total_budget
        errors: list[str] = []
        for p in self.providers:
            if p.name in self._dead:
                continue
            try:
                return p.chat(messages, temperature, max_tokens, deadline=deadline)
            except ModelUnavailable as e:
                self._dead.add(p.name)
                log.critical("MODELO RETIRADO, cambiar en .env: %s", e)
                errors.append(str(e))
            except LLMError as e:
                log.warning("proveedor %s falló: %s", p.name, e)
                errors.append(str(e))
        raise LLMUnavailable("; ".join(errors) or "no hay proveedores configurados")


def build_llm(endpoints: Sequence[LLMEndpoint], timeout: float, max_retries: int,
              total_budget: float) -> FallbackLLM:
    return FallbackLLM(
        [OpenAICompatLLM(e.name, e.base_url, e.api_key, e.model, timeout, max_retries, e.extra)
         for e in endpoints],
        total_budget,
    )
