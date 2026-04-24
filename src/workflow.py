from __future__ import annotations

import difflib
import json
from typing import TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.llm import LLMConfig, chat_json


FUNCTION_BLOCK_PROMPT = PromptTemplate.from_template(
    """
Ты превращаешь IR одной route-функции в строгий JSON-блок.

Формат ответа:
{{
  "blocks": [
    {{"kind":"action","text":"..."}},
    {{"kind":"if","condition":"...","then":[{{"kind":"action","text":"..."}}],"else":[{{"kind":"action","text":"..."}}]}},
    {{"kind":"partition","title":"...","blocks":[{{"kind":"action","text":"..."}}]}}
  ]
}}

Жесткие правила:
1. Только JSON, без markdown.
2. kind только: action, if, partition.
3. Старайся описывать каждый блок кодом, если никак не получится передать смысл кода, то текстом описывай

route:
{route_json}

route_function:
{function_json}

feedback:
{feedback}
""".strip()
)

FUNCTION_SERVICE_BLOCK_PROMPT = PromptTemplate.from_template(
    """
Ты превращаешь IR одной функции в строгую последовательность блоков в JSON.

Формат ответа:
{{
  "blocks": [
    {{"kind":"action","text":"..."}},
    {{"kind":"if","condition":"...","then":[{{"kind":"action","text":"..."}}],"else":[{{"kind":"action","text":"..."}}]}},
    {{"kind":"partition","title":"...","blocks":[{{"kind":"action","text":"..."}}]}}
  ]
}}

Жесткие правила:
1. Только JSON, без markdown.
2. kind только: action, if, partition.
3. Старайся описывать каждый блок кодом, если никак не получится передать смысл кода, то текстом описывай

function:
{function_json}

feedback:
{feedback}
""".strip()
)

COMPRESS_BLOCK_PROMPT = PromptTemplate.from_template(
    """
Ты сжимаешь JSON-блок activity-диаграммы.
Ответ только JSON в том же формате:
{{"blocks":[...]}}

Правила компрессии:
{compression_rules}

Жесткие правила:
1. Только JSON.
2. Не меняй смысл веток if.
3. Не удаляй важные error branches, DB read/write, entity/model construction, внешние вызовы.
4. Удаляй или схлопывай мелкие технические шаги без бизнес-смысла.
5. Не добавляй новые блоки, которых не было в исходном JSON.

current_block_json:
{current_block_json}
""".strip()
)

VALIDATOR_PROMPT = PromptTemplate.from_template(
    """
Ты проверяешь уже собранную PlantUML activity-диаграмму.
Ответ только JSON:
{{"is_valid": true, "feedback": ""}}

Ошибки:
1. сломанный синтаксис;
2. route-level дубли внутри service-body;
3. helper-логика раскрыта второй раз;
4. важная ветка потеряна;
5. слишком много технического шума.

Не придирайся к мелкому стилю.

route:
{route_json}

route_function:
{route_function_json}

service_functions:
{service_functions_json}

diagram:
{current_puml}

deterministic_error:
{deterministic_error}
""".strip()
)


class ServiceArtifact(TypedDict):
    function_id: str
    blocks: list[dict[str, object]]
    current_puml: str


class DiagramState(TypedDict, total=False):
    route: dict[str, object]
    route_function: dict[str, object]
    service_functions: list[dict[str, object]]
    llm_config: LLMConfig
    header_fragment: str
    footer_fragment: str
    route_blocks: list[dict[str, object]]
    route_body_fragment: str
    raw_blocks: list[dict[str, object]]
    filtered_blocks: list[dict[str, object]]
    compressed_blocks: list[dict[str, object]]
    body_fragment: str
    service_artifacts: list[ServiceArtifact]
    current_puml: str
    current_diff: str
    validation_feedback: str
    retry_count: int
    max_retries: int
    validator_passed: bool


def build_workflow():
    graph = StateGraph(DiagramState)
    graph.add_node("build_shell", build_shell_node)
    graph.add_node("generate_route_blocks", generate_route_blocks_node)
    graph.add_node("generate_service_blocks", generate_service_blocks_node)
    graph.add_node("compress_blocks", compress_blocks_node)
    graph.add_node("render_puml", render_puml_node)
    graph.add_node("validate_puml", validate_puml_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "build_shell")
    graph.add_edge("build_shell", "generate_route_blocks")
    graph.add_edge("generate_route_blocks", "generate_service_blocks")
    graph.add_edge("generate_service_blocks", "compress_blocks")
    graph.add_edge("compress_blocks", "render_puml")
    graph.add_edge("render_puml", "validate_puml")
    graph.add_conditional_edges(
        "validate_puml",
        route_after_validation,
        {
            "retry": "generate_route_blocks",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def build_shell_node(state: DiagramState) -> DiagramState:
    header_fragment = _build_header(state["route"], state["route_function"])
    footer_fragment = _build_footer(state["route"], state["route_function"])
    current_puml = _assemble_puml(header_fragment, "", footer_fragment)
    return {
        "header_fragment": header_fragment,
        "footer_fragment": footer_fragment,
        "route_blocks": [],
        "route_body_fragment": "",
        "raw_blocks": [],
        "filtered_blocks": [],
        "compressed_blocks": [],
        "body_fragment": "",
        "service_artifacts": [],
        "current_puml": current_puml,
        "current_diff": _diff_text("", current_puml),
        "retry_count": 0,
        "validation_feedback": "",
        "validator_passed": False,
    }


def generate_route_blocks_node(state: DiagramState) -> DiagramState:
    prompt = FUNCTION_BLOCK_PROMPT.format(
        route_json=json.dumps(state["route"], ensure_ascii=False, indent=2),
        function_json=json.dumps(state["route_function"], ensure_ascii=False, indent=2),
        feedback=state.get("validation_feedback", ""),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты генерируешь только JSON-блоки для route-функции.",
        user_prompt=prompt,
        node_name="generate_route_blocks",
    )
    route_blocks = _filter_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        route_function=state["route_function"],
    )
    route_body_fragment = _render_blocks(route_blocks)
    return {
        "route_blocks": route_blocks,
        "route_body_fragment": route_body_fragment,
        "validator_passed": False,
    }


def generate_service_blocks_node(state: DiagramState) -> DiagramState:
    raw_blocks: list[dict[str, object]] = []
    filtered_service_blocks: list[dict[str, object]] = []
    service_artifacts: list[ServiceArtifact] = []

    for service_function in state["service_functions"]:
        prompt = FUNCTION_SERVICE_BLOCK_PROMPT.format(
            function_json=json.dumps(service_function, ensure_ascii=False, indent=2),
            feedback=state.get("validation_feedback", ""),
        )
        result = chat_json(
            state["llm_config"],
            system_prompt="Ты генерируешь только JSON-блоки для функции.",
            user_prompt=prompt,
            node_name=f"generate_service_blocks:{service_function.get('function_id', 'unknown')}",
        )
        next_raw_blocks = _ensure_dict_list(result.get("blocks"))
        next_filtered_blocks = _filter_generated_blocks(
            next_raw_blocks,
            route_function=state["route_function"],
        )
        function_id = str(service_function.get("function_id", "unknown"))
        raw_blocks.extend(next_raw_blocks)
        filtered_service_blocks.extend(next_filtered_blocks)
        service_artifacts.append(
            {
                "function_id": function_id,
                "blocks": next_filtered_blocks,
                "current_puml": _build_service_puml(function_id, next_filtered_blocks),
            }
        )

    filtered_blocks = list(state.get("route_blocks", [])) + filtered_service_blocks

    return {
        "raw_blocks": raw_blocks,
        "filtered_blocks": filtered_blocks,
        "service_artifacts": service_artifacts,
        "validator_passed": False,
    }


def compress_blocks_node(state: DiagramState) -> DiagramState:
    prompt = COMPRESS_BLOCK_PROMPT.format(
        compression_rules=_compression_rules(),
        current_block_json=json.dumps(state.get("filtered_blocks", []), ensure_ascii=False, indent=2),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты сжимаешь JSON-блоки и отвечаешь только JSON.",
        user_prompt=prompt,
        node_name="compress_blocks",
    )
    compressed_blocks = _ensure_dict_list(result.get("blocks"))
    compressed_blocks = _filter_generated_blocks(
        compressed_blocks,
        route_function=state["route_function"],
    )
    return {
        "compressed_blocks": compressed_blocks,
        "validator_passed": False,
    }


def render_puml_node(state: DiagramState) -> DiagramState:
    previous_puml = state.get("current_puml", "")
    body_fragment = _render_blocks(state.get("compressed_blocks", []))
    current_puml = _assemble_puml(state["header_fragment"], body_fragment, state["footer_fragment"])
    return {
        "body_fragment": body_fragment,
        "current_puml": current_puml,
        "current_diff": _diff_text(previous_puml, current_puml),
        "validator_passed": False,
    }


def validate_puml_node(state: DiagramState) -> DiagramState:
    deterministic_error = _deterministic_validate(state["current_puml"])
    prompt = VALIDATOR_PROMPT.format(
        route_json=json.dumps(state["route"], ensure_ascii=False, indent=2),
        route_function_json=json.dumps(state["route_function"], ensure_ascii=False, indent=2),
        service_functions_json=json.dumps(state["service_functions"], ensure_ascii=False, indent=2),
        current_puml=state["current_puml"],
        deterministic_error=deterministic_error or "none",
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты валидируешь activity-диаграмму и отвечаешь только JSON.",
        user_prompt=prompt,
        node_name="validate_puml",
    )
    feedback = str(result.get("feedback", "")).strip()
    llm_is_valid = result.get("is_valid")
    validator_passed = deterministic_error is None and llm_is_valid is True

    validation_parts: list[str] = []
    if deterministic_error:
        validation_parts.append(deterministic_error)
    if feedback:
        validation_parts.append(feedback)
    if llm_is_valid is not True:
        validation_parts.append(f"LLM validator returned is_valid={llm_is_valid!r}")

    next_state: DiagramState = {
        "validator_passed": validator_passed,
        "validation_feedback": "" if validator_passed else "\n".join(validation_parts),
    }
    next_state["retry_count"] = 0 if validator_passed else state.get("retry_count", 0) + 1
    return next_state


def finalize_node(state: DiagramState) -> DiagramState:
    final_error = _deterministic_validate(state["current_puml"])
    if final_error:
        raise RuntimeError(f"Final diagram is invalid: {final_error}")
    for service_artifact in state.get("service_artifacts", []):
        service_error = _deterministic_validate(service_artifact["current_puml"])
        if service_error:
            function_id = service_artifact.get("function_id", "unknown")
            raise RuntimeError(f"Service diagram {function_id} is invalid: {service_error}")
    return {
        "current_puml": state["current_puml"],
        "service_artifacts": state.get("service_artifacts", []),
    }


def route_after_validation(state: DiagramState) -> str:
    if state.get("validator_passed", False):
        return "finalize"
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry"
    raise RuntimeError(f"Validation failed: {state.get('validation_feedback', '')}")


def _build_header(
    route: dict[str, object],
    route_function: dict[str, object],
) -> str:
    lines = ["@startuml", f"title {route['route_id']}", "start", "", ":Request;", "note: R1"]
    for parameter in route_function.get("parameters", []):
        name = parameter["name"]
        location = parameter.get("location", "body")
        type_name = parameter.get("type")
        lines.append(f":**{name}** = Request.{_location_label(location)}.{name};")
        if type_name:
            lines.append(f"note: {type_name}")
    return "\n".join(lines).strip()


def _build_footer(
    route: dict[str, object],
    route_function: dict[str, object],
) -> str:
    lines = [""]
    response_model = route.get("response_model")
    if response_model:
        lines.append(f":Build response\\nwith **{_short_name(response_model)}**;")
    else:
        lines.append(":Build response;")
    lines.append("note: R2")
    lines.append(f":HTTP {_http_status_label(route_function.get('http', {}))};")
    lines.append("end")
    lines.append("@enduml")
    return "\n".join(lines).strip()


def _filter_generated_blocks(
    blocks: list[dict[str, object]],
    *,
    route_function: dict[str, object],
) -> list[dict[str, object]]:
    route_param_names = {parameter["name"] for parameter in route_function.get("parameters", [])}
    filtered: list[dict[str, object]] = []
    for block in blocks:
        next_block = _filter_single_block(block, route_param_names=route_param_names)
        if next_block is None:
            continue
        filtered.append(next_block)
    return filtered


def _filter_single_block(
    block: dict[str, object],
    *,
    route_param_names: set[str],
) -> dict[str, object] | None:
    kind = block.get("kind")
    if kind == "action":
        text = str(block.get("text", "")).strip()
        if _is_route_level_duplicate(text, route_param_names):
            return None
        return {"kind": "action", "text": text}

    if kind == "partition":
        title = str(block.get("title", "Block")).strip()
        inner = [
            item
            for item in (
                _filter_single_block(item, route_param_names=route_param_names)
                for item in _ensure_dict_list(block.get("blocks"))
            )
            if item is not None
        ]
        if not inner:
            return None
        return {"kind": "partition", "title": title, "blocks": inner}

    if kind == "if":
        condition = str(block.get("condition", "")).strip()
        then_blocks = [
            item
            for item in (
                _filter_single_block(item, route_param_names=route_param_names)
                for item in _ensure_dict_list(block.get("then"))
            )
            if item is not None
        ]
        else_blocks = [
            item
            for item in (
                _filter_single_block(item, route_param_names=route_param_names)
                for item in _ensure_dict_list(block.get("else"))
            )
            if item is not None
        ]
        if not then_blocks and not else_blocks:
            return None
        payload: dict[str, object] = {"kind": "if", "condition": condition, "then": then_blocks}
        if else_blocks:
            payload["else"] = else_blocks
        return payload

    return None


def _is_route_level_duplicate(text: str, route_param_names: set[str]) -> bool:
    lowered = text.lower()
    if "request." in lowered:
        return True
    if lowered in {"request", "receive request"}:
        return True
    if "request" in lowered and ("receive" in lowered or "extract" in lowered):
        return True
    if lowered.startswith("build response") or lowered.startswith("http "):
        return True
    if "response" in lowered and ("return success" in lowered or "send http" in lowered or "success response" in lowered):
        return True
    if lowered.startswith("return ") and "response" in lowered:
        return True
    if "r1" in lowered or "r2" in lowered:
        return True
    for name in route_param_names:
        if lowered == f"{name} = request.body.{name}".lower():
            return True
        if lowered == f"{name} = request.dependency.{name}".lower():
            return True
        if lowered == f"extract {name} from request".lower():
            return True
    return False


def _compression_rules() -> str:
    return (
        "- Удаляй мелкие технические шаги.\n"
        "- Схлопывай соседние preparation actions.\n"
        "- Не удаляй DB write/read, entity/model creation, ошибки."
    )


def _render_blocks(blocks: object) -> str:
    if not isinstance(blocks, list):
        return ""
    return "\n".join(_render_block_list(blocks))


def _render_block_list(blocks: list[object], indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = "    " * indent
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("kind")
        if kind == "action":
            text = _sanitize_action_text(str(block.get("text", "")))
            if text:
                lines.append(f"{prefix}:{text};")
        elif kind == "partition":
            title = str(block.get("title", "Block"))
            lines.append(f"{prefix}partition {title} {{")
            lines.extend(_render_block_list(_ensure_dict_list(block.get("blocks")), indent + 1))
            lines.append(f"{prefix}}}")
        elif kind == "if":
            condition = _sanitize_condition(str(block.get("condition", "condition")))
            lines.append(f"{prefix}if ({condition}) then (+)")
            lines.extend(_render_block_list(_ensure_dict_list(block.get("then")), indent + 1))
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                lines.append(f"{prefix}else (-)")
                lines.extend(_render_block_list(else_blocks, indent + 1))
            lines.append(f"{prefix}endif")
    return lines


def _deterministic_validate(text: str) -> str | None:
    if "@startuml" not in text:
        return "Missing @startuml"
    if "@enduml" not in text:
        return "Missing @enduml"
    if "\nstart\n" not in text:
        return "Missing start"
    if "\nend\n@enduml" not in text.replace("\r\n", "\n"):
        return "Missing end"
    if text.count("if (") != text.count("endif"):
        return "Unbalanced if/endif count"
    if text.count("@startuml") != 1 or text.count("@enduml") != 1:
        return "Diagram must contain exactly one @startuml and one @enduml"
    if text.count("(") != text.count(")"):
        return "Unbalanced parentheses count"
    lowered = text.lower()
    if "try" in lowered or "\nexcept" in lowered or "\ncatch" in lowered:
        return "Unsupported pseudo-syntax found"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("@startuml", "@enduml", "title ", "start", "end", "if ", "else", "endif", "partition ", "}", ":", "note")):
            continue
        return f"Unsupported line syntax: {line}"
    return None


def _assemble_puml(header_fragment: str, body_fragment: str, footer_fragment: str) -> str:
    parts = [header_fragment.rstrip()]
    if body_fragment.strip():
        parts.append("")
        parts.append(body_fragment.rstrip())
    parts.append("")
    parts.append(footer_fragment.rstrip())
    return "\n".join(parts).strip() + "\n"


def _location_label(location: str) -> str:
    mapping = {
        "body": "Body",
        "dependency": "Dependency",
        "path": "Path",
        "query": "Query",
        "header": "Header",
    }
    return mapping.get(location, "Body")


def _http_status_label(http_meta: dict[str, object]) -> str:
    status_code = http_meta.get("status_code")
    if status_code == 201:
        return "201 Created"
    return "200 OK"


def _short_name(value: object) -> str:
    if not isinstance(value, str):
        return "Response"
    return value.rsplit(".", maxsplit=1)[-1]


def _build_service_puml(function_id: str, blocks: list[dict[str, object]]) -> str:
    header_fragment = "\n".join(["@startuml", f"title {function_id}", "start"])
    footer_fragment = "\n".join([":Return;", "end", "@enduml"])
    return _assemble_puml(header_fragment, _render_blocks(blocks), footer_fragment)


def _sanitize_action_text(text: str) -> str:
    text = text.strip().removeprefix(":").removesuffix(";").strip()
    if text.lower().startswith("http "):
        return ""
    if text.lower().startswith("build response"):
        return ""
    if text.lower() == "request":
        return ""
    return text


def _sanitize_condition(text: str) -> str:
    text = text.strip()
    text = text.removeprefix("if").strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return text or "condition"


def _ensure_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _diff_text(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="before.puml",
            tofile="after.puml",
        )
    )
