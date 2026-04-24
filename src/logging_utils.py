from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path("logs/puml_gen.log")


def log_event(event: str, payload: dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{timestamp}] {event}",
        _render_object(payload),
        "",
    ]
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def _render_object(value: object, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested in value.items():
            if isinstance(nested, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_render_object(nested, indent + 1))
            elif isinstance(nested, str):
                rendered = _render_string(nested, indent + 1)
                if "\n" in nested:
                    lines.append(f"{prefix}{key}: |")
                    lines.append(rendered)
                else:
                    lines.append(f"{prefix}{key}: {nested}")
            else:
                lines.append(f"{prefix}{key}: {nested}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_render_object(item, indent + 1))
            elif isinstance(item, str):
                if "\n" in item:
                    lines.append(f"{prefix}- |")
                    lines.append(_render_string(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines) if lines else f"{prefix}[]"
    if isinstance(value, str):
        if "\n" in value:
            return _render_string(value, indent)
        return f"{prefix}{value}"
    return f"{prefix}{value}"


def _render_string(value: str, indent: int) -> str:
    prefix = "  " * indent
    return "\n".join(f"{prefix}{line}" for line in value.splitlines()) or prefix
