from __future__ import annotations

import json
from pathlib import Path

from src.llm import load_config
from src.logging_utils import log_run_start
from src.workflow import DiagramMode
from src.workflow import build_workflow


def gen_png_graph(app_obj: Any, name_photo: str = "graph.png") -> None:
    """
    Генерирует PNG-изображение графа и сохраняет в файл.

    Args:
        app_obj: Скомпилированный объект графа
        name_photo: Имя файла для сохранения (по умолчанию "graph.png")
    """
    with open(name_photo, "wb") as f:
        f.write(app_obj.get_graph().draw_mermaid_png())


def generate_from_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    diagram_mode: DiagramMode = "route",
) -> list[Path]:
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_run_start(
        "synthetic_generator_with_ML.generate_from_file",
        {
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "diagram_mode": diagram_mode,
        },
    )

    llm_config = load_config()
    functions_by_id = {function["function_id"]: function for function in data.get("functions", [])}
    workflow = build_workflow()

    gen_png_graph(workflow, "docs/graph.png")

    generated_files: list[Path] = []
    for route in data.get("routes", []):
        route_function = functions_by_id[route["handler_function_id"]]
        service_functions = _resolve_service_functions(route, functions_by_id)
        state = {
            "route": route,
            "route_function": route_function,
            "service_functions": service_functions,
            "diagram_mode": diagram_mode,
            "llm_config": llm_config,
            "max_retries": 2,
        }
        result = workflow.invoke(state)
        out_path = output_dir / f"{_route_slug(str(route['route_id']))}.activity.puml"
        out_path.write_text(result["current_puml"], encoding="utf-8")
        generated_files.append(out_path)
    return generated_files


def _route_slug(route_id: str) -> str:
    route_id = route_id.replace(" /", "_").replace("/", "_").replace(" ", "_")
    route_id = route_id.replace("{", "").replace("}", "").replace("-", "_")
    return route_id.strip("_")


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
