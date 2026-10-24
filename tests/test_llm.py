import httpx
import pytest

from app.adapters.llm_openai_compat import (FallbackLLM, LLMUnavailable, ModelUnavailable,
                                            OpenAICompatLLM)

MSG = [{"role": "user", "content": "hola"}]


def _llm(handler, name="p", retries=3, sleeps=None):
    return OpenAICompatLLM(name, "http://x/v1", "k", "m", max_retries=retries,
                           transport=httpx.MockTransport(handler),
                           sleep=(sleeps.append if sleeps is not None else lambda s: None))


def ok(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_retries_with_exponential_backoff_then_succeeds():
    calls, sleeps = [], []

    def h(req):
        calls.append(1)
        return httpx.Response(500) if len(calls) < 3 else ok("listo")

    assert _llm(h, sleeps=sleeps).chat(MSG).text == "listo"
    assert len(calls) == 3 and sleeps == [1, 2]


def test_gives_up_after_max_retries():
    with pytest.raises(Exception) as e:
        _llm(lambda r: httpx.Response(429), retries=3).chat(MSG)
    assert "429" in str(e.value)


def test_decommissioned_model_is_not_retried():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(400, json={"error": {"code": "model_decommissioned",
                                                    "message": "The model has been decommissioned"}})

    with pytest.raises(ModelUnavailable):
        _llm(h).chat(MSG)
    assert len(calls) == 1


def test_strips_think_blocks():
    assert _llm(lambda r: ok("<think>razono...</think>Respuesta final")).chat(MSG).text == "Respuesta final"


def test_fallback_skips_retired_model_and_remembers():
    dead = _llm(lambda r: httpx.Response(404, text="model_not_found"), name="viejo")
    live = _llm(lambda r: ok("desde el respaldo"), name="respaldo")
    fb = FallbackLLM([dead, live])
    r = fb.chat(MSG)
    assert (r.text, r.provider) == ("desde el respaldo", "respaldo")
    assert "viejo" in fb._dead


def test_fallback_all_fail():
    bad = _llm(lambda r: httpx.Response(401), name="a")
    with pytest.raises(LLMUnavailable):
        FallbackLLM([bad]).chat(MSG)
    with pytest.raises(LLMUnavailable):
        FallbackLLM([]).chat(MSG)


def test_extra_params_are_merged_into_payload():
    seen = {}

    def h(req):
        import json
        seen.update(json.loads(req.content))
        return ok("con razonamiento bajo")

    llm = OpenAICompatLLM("groq", "http://x/v1", "k", "m", transport=httpx.MockTransport(h))
    llm.extra = {"reasoning_effort": "low"}
    r = llm.chat(MSG)
    assert r.text == "con razonamiento bajo"
    assert seen["reasoning_effort"] == "low" and seen["max_tokens"] == 900


# ---- tope de espera total (RNF-FI-02) con reloj simulado -------------------
class FakeClock:
    """Reloj falso: el tiempo solo avanza cuando la prueba lo pide."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def _slow_llm(clock, name="p", seconds=25.0, timeouts=None, retries=3, request_timeout=25.0):
    """Cada petición "tarda" lo que permita su timeout (máx. `seconds`) y truena."""

    def h(req):
        t = req.extensions["timeout"]["read"]
        if timeouts is not None:
            timeouts.append(t)
        clock.now += min(seconds, t)
        raise httpx.ReadTimeout("lento", request=req)

    return OpenAICompatLLM(name, "http://x/v1", "k", "m", timeout=request_timeout,
                           max_retries=retries, transport=httpx.MockTransport(h),
                           sleep=clock.sleep, clock=clock)


def test_timeout_de_cada_peticion_se_recorta_al_tiempo_restante():
    clock, timeouts = FakeClock(), []
    llm = _slow_llm(clock, timeouts=timeouts)
    with pytest.raises(Exception):
        llm.chat(MSG, deadline=clock() + 30)
    # 25 s del 1er intento, espera de 1 s, y al 2o solo le quedan 4 s
    assert timeouts == [25.0, 4.0]
    assert clock() <= 1000 + 30


def test_fallback_con_varios_proveedores_lentos_respeta_el_tope():
    clock = FakeClock()
    providers = [_slow_llm(clock, name=n) for n in ("groq", "openai", "ollama")]
    fb = FallbackLLM(providers, total_budget=80, clock=clock)
    with pytest.raises(LLMUnavailable):
        fb.chat(MSG)
    # Sin el recorte serían 3 proveedores x 3 intentos x 25 s (> 3 min)
    assert clock() - 1000 <= 80


def test_fallback_usa_el_deadline_recibido_si_es_mas_cercano():
    clock, timeouts = FakeClock(), []
    fb = FallbackLLM([_slow_llm(clock, timeouts=timeouts, retries=1)], total_budget=80,
                     clock=clock)
    with pytest.raises(LLMUnavailable):
        fb.chat(MSG, deadline=clock() + 10)
    assert timeouts == [10.0]


def test_deadline_vencido_no_hace_peticiones():
    clock, timeouts = FakeClock(), []
    llm = _slow_llm(clock, timeouts=timeouts)
    with pytest.raises(Exception):
        llm.chat(MSG, deadline=clock() - 1)
    assert timeouts == []
