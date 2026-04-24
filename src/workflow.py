from __future__ import annotations

import difflib
import json
from typing import Literal, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.llm import LLMConfig, chat_json


DiagramMode = Literal["route", "service"]

STYLE_SUMMARY = """
Стиль PlantUML activity:
- Только activity-диаграмма.
- Route mode: Request в начале, R1/R2, финальный HTTP.
- Service mode: без Request и без route-level HTTP.
- Действия только как :...;
- Ветвление только if (...) then (+) / else (-) / endif
- БД только текстом: find/create/update in DB `Table` ...
- Не использовать try/catch/except/end try.
""".strip()

FUNCTION_BLOCK_PROMPT = PromptTemplate.from_template(
    """
Ты превращаешь IR одной route-функции в строгий JSON-блок.
Ответ только JSON.

Формат:
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
3. Нельзя писать @startuml, @enduml, title, start, end, Request, R1, R2, Build response, HTTP.
4. Показывай только orchestration текущей route-функции.
5. Нельзя раскрывать service/helper functions.
6. Нельзя писать строки про Request.* или response-level блоки.
7. Нельзя использовать try/catch/except.
8. action.text без ":" в начале и без ";" в конце.
9. В route mode делай блок кратким и orchestration-only.
10. В service mode route-блок нужен только как контекст для service-функций, не добавляй лишние детали.

Стиль:
{style_summary}

diagram_mode:
{diagram_mode}

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
Ты превращаешь IR одной service-функции в строгий JSON-блок.
Ответ только JSON.

Формат:
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
3. Нельзя писать @startuml, @enduml, title, start, end, Request, R1, R2, Build response, HTTP.
4. Раскрывай только текущую service-функцию; nested helper functions раскрывать нельзя.
5. Не дублируй шаги, которые уже отражены в current_route_code.
6. Нельзя писать строки про Request.* или response-level блоки.
7. Нельзя использовать try/catch/except.
8. action.text без ":" в начале и без ";" в конце.
9. В route mode пиши только значимые бизнес-шаги service-функции.
10. В service mode можно чуть подробнее, но без технического шума.

Стиль:
{style_summary}

diagram_mode:
{diagram_mode}

route:
{route_json}

route_function:
{route_function_json}

current_route_code:
{current_route_code}

service_function:
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
6. В route mode делай блок заметно короче.
7. В service mode сохраняй больше деталей.

diagram_mode:
{diagram_mode}

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
5. слишком много технического шума для route mode.

Не придирайся к мелкому стилю.

diagram_mode:
{diagram_mode}

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


class DiagramState(TypedDict, total=False):
    route: dict[str, object]
    route_function: dict[str, object]
    service_functions: list[dict[str, object]]
    diagram_mode: DiagramMode
    llm_config: LLMConfig
    header_fragment: str
    footer_fragment: str
    route_blocks: list[dict[str, object]]
    route_body_fragment: str
    raw_blocks: list[dict[str, object]]
    filtered_blocks: list[dict[str, object]]
    compressed_blocks: list[dict[str, object]]
    body_fragment: str
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
    header_fragment = _build_header(state["diagram_mode"], state["route"], state["route_function"], state["service_functions"])
    footer_fragment = _build_footer(state["diagram_mode"], state["route"], state["route_function"], state["service_functions"])
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
        "current_puml": current_puml,
        "current_diff": _diff_text("", current_puml),
        "retry_count": 0,
        "validation_feedback": "",
        "validator_passed": False,
    }


def generate_route_blocks_node(state: DiagramState) -> DiagramState:
    prompt = FUNCTION_BLOCK_PROMPT.format(
        style_summary=STYLE_SUMMARY,
        diagram_mode=state["diagram_mode"],
        route_json=json.dumps(state["route"], ensure_ascii=False, indent=2),
        function_json=json.dumps(state["route_function"], ensure_ascii=False, indent=2),
        feedback=state.get("validation_feedback", ""),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты генерируешь только JSON-блоки для route activity-диаграммы.",
        user_prompt=prompt,
        node_name="generate_route_blocks",
    )
    route_blocks = _filter_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        diagram_mode=state["diagram_mode"],
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

    for service_function in state["service_functions"]:
        prompt = FUNCTION_SERVICE_BLOCK_PROMPT.format(
            style_summary=STYLE_SUMMARY,
            diagram_mode=state["diagram_mode"],
            route_json=json.dumps(state["route"], ensure_ascii=False, indent=2),
            route_function_json=json.dumps(state["route_function"], ensure_ascii=False, indent=2),
            current_route_code=state.get("route_body_fragment", ""),
            function_json=json.dumps(service_function, ensure_ascii=False, indent=2),
            feedback=state.get("validation_feedback", ""),
        )
        result = chat_json(
            state["llm_config"],
            system_prompt="Ты генерируешь только JSON-блоки для service activity-диаграммы.",
            user_prompt=prompt,
            node_name=f"generate_service_blocks:{service_function.get('function_id', 'unknown')}",
        )
        next_raw_blocks = _ensure_dict_list(result.get("blocks"))
        next_filtered_blocks = _filter_generated_blocks(
            next_raw_blocks,
            diagram_mode=state["diagram_mode"],
            route_function=state["route_function"],
        )
        raw_blocks.extend(next_raw_blocks)
        filtered_service_blocks.extend(next_filtered_blocks)

    filtered_blocks = list(state.get("route_blocks", [])) + filtered_service_blocks
    if state["diagram_mode"] == "service":
        filtered_blocks = filtered_service_blocks

    return {
        "raw_blocks": raw_blocks,
        "filtered_blocks": filtered_blocks,
        "validator_passed": False,
    }


def compress_blocks_node(state: DiagramState) -> DiagramState:
    prompt = COMPRESS_BLOCK_PROMPT.format(
        diagram_mode=state["diagram_mode"],
        compression_rules=_compression_rules(state["diagram_mode"]),
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
        diagram_mode=state["diagram_mode"],
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
        diagram_mode=state["diagram_mode"],
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
    validator_passed = deterministic_error is None

    next_state: DiagramState = {
        "validator_passed": validator_passed,
        "validation_feedback": "" if validator_passed else "\n".join(part for part in [deterministic_error, feedback] if part),
    }
    next_state["retry_count"] = 0 if validator_passed else state.get("retry_count", 0) + 1
    return next_state


def finalize_node(state: DiagramState) -> DiagramState:
    final_error = _deterministic_validate(state["current_puml"])
    if final_error:
        raise RuntimeError(f"Final diagram is invalid: {final_error}")
    return {"current_puml": state["current_puml"]}


def route_after_validation(state: DiagramState) -> str:
    if state.get("validator_passed", False):
        return "finalize"
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry"
    raise RuntimeError(f"Validation failed: {state.get('validation_feedback', '')}")


def _build_header(
    diagram_mode: DiagramMode,
    route: dict[str, object],
    route_function: dict[str, object],
    service_functions: list[dict[str, object]],
) -> str:
    title = route["route_id"] if diagram_mode == "route" else _service_diagram_title(route, service_functions)
    lines = ["@startuml", f"title {title}", "start"]
    if diagram_mode == "route":
        lines.extend(["", ":Request;", "note: R1"])
        for parameter in route_function.get("parameters", []):
            name = parameter["name"]
            location = parameter.get("location", "body")
            type_name = parameter.get("type")
            lines.append(f":**{name}** = Request.{_location_label(location)}.{name};")
            if type_name:
                lines.append(f"note: {type_name}")
    return "\n".join(lines).strip()


def _build_footer(
    diagram_mode: DiagramMode,
    route: dict[str, object],
    route_function: dict[str, object],
    service_functions: list[dict[str, object]],
) -> str:
    lines = [""]
    if diagram_mode == "route":
        response_model = route.get("response_model")
        if response_model:
            lines.append(f":Build response\\nwith **{_short_name(response_model)}**;")
        else:
            lines.append(":Build response;")
        lines.append("note: R2")
        lines.append(f":HTTP {_http_status_label(route_function.get('http', {}))};")
    else:
        lines.append(":Return;")
    lines.append("end")
    lines.append("@enduml")
    return "\n".join(lines).strip()


def _filter_generated_blocks(
    blocks: list[dict[str, object]],
    *,
    diagram_mode: DiagramMode,
    route_function: dict[str, object],
) -> list[dict[str, object]]:
    route_param_names = {parameter["name"] for parameter in route_function.get("parameters", [])}
    filtered: list[dict[str, object]] = []
    for block in blocks:
        next_block = _filter_single_block(block, diagram_mode=diagram_mode, route_param_names=route_param_names)
        if next_block is None:
            continue
        filtered.append(next_block)
    return filtered


def _filter_single_block(
    block: dict[str, object],
    *,
    diagram_mode: DiagramMode,
    route_param_names: set[str],
) -> dict[str, object] | None:
    kind = block.get("kind")
    if kind == "action":
        text = str(block.get("text", "")).strip()
        if _is_route_level_duplicate(text, route_param_names):
            return None
        if diagram_mode == "route" and _is_technical_noise(text):
            return None
        return {"kind": "action", "text": text}

    if kind == "partition":
        title = str(block.get("title", "Block")).strip()
        inner = [
            item
            for item in (
                _filter_single_block(item, diagram_mode=diagram_mode, route_param_names=route_param_names)
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
                _filter_single_block(item, diagram_mode=diagram_mode, route_param_names=route_param_names)
                for item in _ensure_dict_list(block.get("then"))
            )
            if item is not None
        ]
        else_blocks = [
            item
            for item in (
                _filter_single_block(item, diagram_mode=diagram_mode, route_param_names=route_param_names)
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


def _is_technical_noise(text: str) -> bool:
    lowered = text.lower()
    noise_markers = (
        "timestamp",
        "created_at",
        "updated_at",
        "current time",
        "get db session",
        "commit db",
        "commit changes",
        "prepare",
        "assign temporary",
    )
    return any(marker in lowered for marker in noise_markers)


def _compression_rules(diagram_mode: DiagramMode) -> str:
    if diagram_mode == "route":
        return (
            "- Удаляй мелкие технические шаги.\n"
            "- Схлопывай соседние preparation actions.\n"
            "- Оставляй только бизнес-значимые действия, DB write/read, entity/model creation, ошибки."
        )
    return (
        "- Сжимай умеренно.\n"
        "- Оставляй больше деталей функции.\n"
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


def _service_diagram_title(route: dict[str, object], service_functions: list[dict[str, object]]) -> str:
    if len(service_functions) == 1:
        return str(service_functions[0].get("function_id", route["route_id"]))
    return f"{route['route_id']} services"


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


def _is_hard_feedback(feedback: str) -> bool:
    lowered = feedback.lower()
    return any(
        marker in lowered
        for marker in (
            "сломанный синтаксис",
            "broken syntax",
            "duplicat",
            "helper",
            "смешаны",
            "missing",
            "unsupported",
            "неверный if",
        )
    )
