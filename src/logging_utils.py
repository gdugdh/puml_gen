from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path("logs/puml_gen.log")


def log_event(event: str, payload: dict[str, object]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{timestamp}] {event}",
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        "",
    ]
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write("\n".join(lines))


def log_run_start(command: str, payload: dict[str, object]) -> None:
    log_event(
        "run_start",
        {
            "command": command,
            **payload,
        },
    )
