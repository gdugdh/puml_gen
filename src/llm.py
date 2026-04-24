from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.logging_utils import log_event


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str


def load_config() -> LLMConfig:
    _load_dotenv(Path(".env"))

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return LLMConfig(
        api_key=api_key,
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def chat_json(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    node_name: str = "llm",
) -> dict[str, object]:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    request = urllib.request.Request(
        url=f"{config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.synthetic-generator",
            "X-Title": "synthetic_generator_with_ML",
        },
        method="POST",
    )
    log_event(
        f"{node_name} llm request",
        {
            "payload": payload,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        log_event(
            f"{node_name} llm response",
            {
                "parsed_content": parsed,
            },
        )
        return parsed
    except Exception:
        raise


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip())
