from __future__ import annotations

import json

from src.llm import LLMConfig
from src.llm import chat_json
from src.llm import load_config


def test_load_config_uses_openrouter_by_default(monkeypatch):
    monkeypatch.delenv("LLM", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    config = load_config()

    assert config.provider == "OPENROUTER"
    assert config.model == "openai/gpt-4o-mini"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.chat_path == "/chat/completions"


def test_load_config_uses_local_provider(monkeypatch):
    monkeypatch.setenv("LLM", "LOCAL")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080")

    config = load_config()

    assert config.provider == "LOCAL"
    assert config.model == "llama3.1:8b"
    assert config.base_url == "http://127.0.0.1:8080"
    assert config.chat_path == "/api/chat"


def test_chat_json_uses_local_ollama_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {
                        "content": json.dumps({"blocks": [{"kind": "action", "text": "db.commit()"}]})
                    }
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("src.llm.urllib.request.urlopen", fake_urlopen)

    config = LLMConfig(
        api_key="",
        model="llama3.1:8b",
        base_url="http://127.0.0.1:8080",
        provider="LOCAL",
        chat_path="/api/chat",
        timeout_seconds=90.0,
    )

    result = chat_json(
        config,
        system_prompt="system",
        user_prompt="user",
        node_name="test_local_llm",
    )

    assert result == {"blocks": [{"kind": "action", "text": "db.commit()"}]}
    assert captured["url"] == "http://127.0.0.1:8080/api/chat"
    assert captured["timeout"] == 90.0
    assert captured["headers"]["Content-type"] == "application/json"
    assert captured["payload"] == {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }
