import os

from app.config import Settings


def test_llm_extra_parsed_from_json_env(monkeypatch):
    monkeypatch.setenv("LLM_1_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_1_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_1_API_KEY", "x")
    monkeypatch.setenv("LLM_1_EXTRA", '{"reasoning_effort":"low"}')
    for i in (2, 3):
        monkeypatch.delenv(f"LLM_{i}_BASE_URL", raising=False)
        monkeypatch.delenv(f"LLM_{i}_MODEL", raising=False)
    cfg = Settings.from_env(load_dotenv_file=False)
    assert len(cfg.llm_endpoints) == 1
    assert cfg.llm_endpoints[0].extra == {"reasoning_effort": "low"}


def test_llm_extra_defaults_to_empty_dict_when_absent(monkeypatch):
    monkeypatch.setenv("LLM_1_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_1_MODEL", "openai/gpt-oss-120b")
    monkeypatch.delenv("LLM_1_EXTRA", raising=False)
    for i in (2, 3):
        monkeypatch.delenv(f"LLM_{i}_BASE_URL", raising=False)
        monkeypatch.delenv(f"LLM_{i}_MODEL", raising=False)
    cfg = Settings.from_env(load_dotenv_file=False)
    assert cfg.llm_endpoints[0].extra == {}


def test_llm_extra_invalid_json_falls_back_to_empty(monkeypatch):
    monkeypatch.setenv("LLM_1_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_1_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("LLM_1_EXTRA", "{not valid json")
    for i in (2, 3):
        monkeypatch.delenv(f"LLM_{i}_BASE_URL", raising=False)
        monkeypatch.delenv(f"LLM_{i}_MODEL", raising=False)
    cfg = Settings.from_env(load_dotenv_file=False)
    assert cfg.llm_endpoints[0].extra == {}
