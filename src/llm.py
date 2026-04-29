from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.logging_utils import log_event


LLMProvider = Literal["OPENROUTER", "LOCAL"]

    
@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str
    provider: LLMProvider = "OPENROUTER"
    chat_path: str = "/chat/completions"
    timeout_seconds: float = 60.0
    stream: bool = False
    options: dict[str, Any] = field(default_factory=dict)


def build_llm_config(
    requested_model: str,
    *,
    options: dict[str, Any] | None = None,
    stream: bool = False,
) -> LLMConfig:
    _load_dotenv(Path(".env"))

    normalized_model = requested_model.strip()
    normalized_options = _normalize_options(options)
    if normalized_model == "openai/gpt-4o-mini":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return LLMConfig(
            api_key=api_key,
            model=normalized_model,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            provider="OPENROUTER",
            chat_path=os.getenv("OPENROUTER_CHAT_PATH", "/chat/completions"),
            timeout_seconds=_load_timeout("OPENROUTER_TIMEOUT_SECONDS", default=60.0),
            stream=stream,
            options=normalized_options,
        )

    if normalized_model == "local":
        local_model = os.getenv("LOCAL_LLM_MODEL", "").strip()
        if not local_model:
            raise RuntimeError("LOCAL_LLM_MODEL is not set")
        return LLMConfig(
            api_key=os.getenv("LOCAL_LLM_API_KEY", "").strip(),
            model=local_model,
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080"),
            provider="LOCAL",
            chat_path=os.getenv("LOCAL_LLM_CHAT_PATH", "/api/chat"),
            timeout_seconds=_load_timeout("LOCAL_LLM_TIMEOUT_SECONDS", default=120.0),
            stream=stream,
            options=normalized_options,
        )

    raise RuntimeError("model must be 'openai/gpt-4o-mini' or 'local'")


def chat_json(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    node_name: str = "llm",
) -> dict[str, object]:
    payload = _build_payload(config, system_prompt=system_prompt, user_prompt=user_prompt)
    request = _build_request(config, payload)
    log_event(
        f"{node_name} llm request",
        {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "chat_path": config.chat_path,
            "requested_stream": config.stream,
            "payload": payload,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = _extract_response_content(config, body)
        parsed = json.loads(content)
        log_event(
            f"{node_name} llm response",
            {
                "provider": config.provider,
                "parsed_content": parsed,
            },
        )
        return parsed
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        log_event(
            f"{node_name} llm error",
            {
                "provider": config.provider,
                "status": getattr(error, "code", None),
                "reason": str(error),
                "response_body": response_body,
            },
        )
        raise RuntimeError(f"{config.provider} request failed with HTTP {getattr(error, 'code', 'unknown')}") from error
    except json.JSONDecodeError as error:
        log_event(
            f"{node_name} llm error",
            {
                "provider": config.provider,
                "reason": "invalid_json_response",
                "error": str(error),
            },
        )
        raise RuntimeError(f"{config.provider} returned invalid JSON content") from error
    except Exception as error:
        log_event(
            f"{node_name} llm error",
            {
                "provider": config.provider,
                "reason": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _build_payload(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if config.provider == "OPENROUTER":
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        openrouter_options = _openrouter_options(config.options)
        if openrouter_options:
            payload.update(openrouter_options)
        else:
            payload["temperature"] = 0.2
        return payload

    if config.provider == "LOCAL":
        local_options = dict(config.options)
        if "temperature" not in local_options:
            local_options["temperature"] = 0.2
        return {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": local_options,
        }

    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")


def _openrouter_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed = ("temperature", "top_p", "top_k", "repeat_penalty", "stop", "seed")
    result = {
        key: value
        for key, value in options.items()
        if key in allowed and value is not None
    }
    if "num_predict" in options and options["num_predict"] is not None:
        result["max_tokens"] = options["num_predict"]
    return result


def _build_request(config: LLMConfig, payload: dict[str, Any]) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if config.provider == "OPENROUTER":
        headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://local.synthetic-generator",
                "X-Title": "puml_gen",
            }
        )
    elif config.provider == "LOCAL" and config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    return urllib.request.Request(
        url=f"{config.base_url.rstrip('/')}{config.chat_path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def _extract_response_content(config: LLMConfig, body: dict[str, Any]) -> str:
    if config.provider == "OPENROUTER":
        return str(body["choices"][0]["message"]["content"])
    if config.provider == "LOCAL":
        return str(body["message"]["content"])
    raise RuntimeError(f"Unsupported LLM provider: {config.provider}")


def _normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return {}
    return {
        key: value
        for key, value in options.items()
        if value is not None
    }


def _load_timeout(env_name: str, *, default: float) -> float:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{env_name} must be a number") from error


def _load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        os.environ.setdefault(key.strip(), value.strip())
