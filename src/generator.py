from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.api_models import DiagramDocument, GeneratePumlRequest, GeneratePumlResponse
from src.llm import build_llm_config
from src.prompts import build_prompt_overrides
from src.workflow import build_workflow


DEFAULT_OUTPUT_DIR = Path("output")


def gen_png_graph(app_obj: Any, name_photo: str = "graph.png") -> None:
    try:
        with open(name_photo, "wb") as file_obj:
            file_obj.write(app_obj.get_graph().draw_mermaid_png())
    except Exception:
        pass


def generate_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = "openai/gpt-4o-mini",
) -> list[Path]:
    request = GeneratePumlRequest(
        model=model,
        input_path=str(input_path),
    )
    response = generate_from_request(request, output_dir=output_dir)
    generated_files: list[Path] = []
    for document in [*response.routes, *response.artifacts]:
        name = document.name if hasattr(document, "name") else document["name"]
        generated_files.append(Path(output_dir) / name)
    return generated_files


def generate_from_request(
    request: GeneratePumlRequest,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> GeneratePumlResponse:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(request.input_path)
    if input_path.suffix.lower() != ".json":
        raise ValueError("input_path must point to a .json file")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))

    llm_config = build_llm_config(
        request.model,
        options=_model_dump(request.options, exclude_none=True),
        stream=request.stream,
    )
    prompt_overrides = build_prompt_overrides(
        [_model_dump(message) for message in request.messages]
    )
    functions_by_id = {
        function["function_id"]: function
        for function in data.get("functions", [])
    }
    workflow = build_workflow()

    gen_png_graph(workflow, "docs/graph.png")

    response_routes: list[dict[str, str]] = []
    response_artifacts: list[dict[str, str]] = []
    for route in data.get("routes", []):
        route_function = functions_by_id[route["handler_function_id"]]
        service_functions = _resolve_service_functions(route, functions_by_id)
        state = {
            "route": route,
            "route_function": route_function,
            "service_functions": service_functions,
            "llm_config": llm_config,
            "prompt_overrides": prompt_overrides,
            "max_retries": 3,
        }
        result = workflow.invoke(state)
        response_routes.extend(result.get("response_routes", []))
        response_artifacts.extend(result.get("response_artifacts", []))

    _write_documents(output_dir, response_routes)
    _write_documents(output_dir, response_artifacts)
    return GeneratePumlResponse(
        routes=[DiagramDocument(**document) for document in response_routes],
        artifacts=[DiagramDocument(**document) for document in response_artifacts],
    )


def _write_documents(output_dir: Path, documents: list[dict[str, str]]) -> None:
    for document in documents:
        (output_dir / document["name"]).write_text(document["puml"], encoding="utf-8")


def _model_dump(value: Any, *, exclude_none: bool = False) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=exclude_none)
    if isinstance(value, dict):
        if exclude_none:
            return {key: nested for key, nested in value.items() if nested is not None}
        return dict(value)
    raise TypeError(f"Unsupported model value: {type(value).__name__}")


def _resolve_service_functions(
    route: dict[str, object],
    functions_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    nested_service_ids = route.get("service_function_groups")
    service_ids: list[str] = []

    if isinstance(nested_service_ids, list):
        for group in nested_service_ids:
            if isinstance(group, list):
                service_ids.extend(function_id for function_id in group if isinstance(function_id, str))
            elif isinstance(group, str):
                service_ids.append(group)

    legacy_service_id = route.get("service_entry_function_id")
    if not service_ids and isinstance(legacy_service_id, str):
        service_ids.append(legacy_service_id)

    if not service_ids:
        raise ValueError(f"Route {route.get('route_id', '<unknown>')} does not reference service functions")

    service_functions: list[dict[str, object]] = []
    for function_id in service_ids:
        if function_id not in functions_by_id:
            raise KeyError(f"Unknown service function id '{function_id}' for route {route.get('route_id', '<unknown>')}")
        service_functions.append(functions_by_id[function_id])
    return service_functions
