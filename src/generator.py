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
        service_function = functions_by_id[route["service_entry_function_id"]]
        state = {
            "route": route,
            "route_function": route_function,
            "service_function": service_function,
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
