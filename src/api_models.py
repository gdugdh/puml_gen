from __future__ import annotations

from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    class _FieldSpec:
        def __init__(self, default: Any = None, default_factory: Any | None = None) -> None:
            self.default = default
            self.default_factory = default_factory

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self, "__annotations__", {})
            for key, value in annotations.items():
                if key in data:
                    setattr(self, key, data[key])
                elif hasattr(type(self), key):
                    default_value = getattr(type(self), key)
                    if isinstance(default_value, _FieldSpec):
                        if default_value.default_factory is not None:
                            default_value = default_value.default_factory()
                        else:
                            default_value = default_value.default
                    elif isinstance(default_value, list):
                        default_value = list(default_value)
                    elif isinstance(default_value, dict):
                        default_value = dict(default_value)
                    setattr(self, key, default_value)
                else:
                    setattr(self, key, None)

        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            payload = {
                key: getattr(self, key)
                for key in getattr(self, "__annotations__", {})
            }
            if exclude_none:
                return {key: value for key, value in payload.items() if value is not None}
            return payload

    def Field(*, default: Any = None, default_factory: Any | None = None, **_: Any) -> Any:
        return _FieldSpec(default=default, default_factory=default_factory)


PromptRole = Literal[
    "route-system",
    "route-user",
    "service-system",
    "service-user",
    "compress-system",
    "compress-user",
]

ModelName = Literal["openai/gpt-4o-mini", "local"]


class PromptMessage(BaseModel):
    role: PromptRole
    content: str


class LLMOptions(BaseModel):
    mirostat: int | None = None
    mirostat_eta: float | None = None
    mirostat_tau: float | None = None
    num_ctx: int | None = None
    num_gpu: int | None = None
    num_thread: int | None = None
    num_predict: int | None = None
    repeat_last_n: int | None = None
    repeat_penalty: float | None = None
    temperature: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    tfs_z: float | None = None
    top_k: int | None = None
    top_p: float | None = None


class GeneratePumlRequest(BaseModel):
    model: ModelName = "openai/gpt-4o-mini"
    messages: list[PromptMessage] = Field(default_factory=list)
    stream: bool = False
    options: LLMOptions = Field(default_factory=LLMOptions)
    input_path: str


class DiagramDocument(BaseModel):
    name: str
    puml: str


class GeneratePumlResponse(BaseModel):
    artifacts: list[DiagramDocument] = Field(default_factory=list)
    routes: list[DiagramDocument] = Field(default_factory=list)
