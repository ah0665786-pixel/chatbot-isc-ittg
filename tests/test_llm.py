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
