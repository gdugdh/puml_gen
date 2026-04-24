from __future__ import annotations

import difflib
import json
from typing import Literal, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from src.llm import LLMConfig, chat_json
from src.logging_utils import log_event


DiagramKind = Literal["route", "service", "merge"]


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
3. Старайся описывать каждый блок кодом, если никак не получится передать смысл кода, то текстом описывай.
4. Пиши текст на русском.
5. Если блок заканчивается raise, добавь сразу после него {{"kind":"action","text":"end"}}.
6. Если блок заканчивается return, добавь сразу после него {{"kind":"action","text":"stop"}}.
7. end и stop пиши только как action.text ровно "end" или "stop".

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
3. Старайся описывать каждый блок кодом, если никак не получится передать смысл кода, то текстом описывай.
4. Пиши текст на русском.
5. Если блок заканчивается raise, добавь сразу после него {{"kind":"action","text":"end"}}.
6. Если блок заканчивается return, добавь сразу после него {{"kind":"action","text":"stop"}}.
7. end и stop пиши только как action.text ровно "end" или "stop".
8. Старайся писать больше кода и минимум слов, вот примеры хороших результатов:
gameIds = множество ключей set<key> из donateHubGames по полю id
wataComissionRub = price / wataRateRubToUsdt * (terminalComissionInPercent / 100)
get_password_hash(user.password)

IR функции:
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
6. Не удаляй и не перемещай action с text "end" или "stop".

current_block_json:
{current_block_json}
""".strip()
)

VALIDATOR_PROMPT = PromptTemplate.from_template(
    """
Ты проверяешь уже собранную PlantUML activity-диаграмму.
Ответ только JSON:
{{"is_valid": true, "feedback": ""}}

Правила:
1. Если deterministic_error не "none", верни is_valid=false и feedback с этой ошибкой.
2. Если deterministic_error == "none", считай синтаксис PlantUML корректным.
3. Общая диаграмма ожидаемо содержит route-shell (Request/HTTP/Build response) и service actions вместе. Это не ошибка.
4. end в error-ветке и stop в успешном конце являются ожидаемыми PlantUML terminal blocks.
5. Не проверяй область видимости переменных: диаграмма не исполняемый код, поэтому {{e}} в тексте logging.error допустим.
6. Верни is_valid=false только если можешь указать конкретную потерянную важную ветку или конкретный дублирующий/мусорный фрагмент из diagram.
7. feedback всегда строка, не список.
8. Не копируй эти правила в feedback.

Не придирайся к мелкому стилю.

diagram_kind:
{diagram_kind}

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


class DiagramArtifact(TypedDict):
    function_id: str
    blocks: list[dict[str, object]]
    current_puml: str


class ServiceArtifact(DiagramArtifact):
    pass


class DiagramState(TypedDict, total=False):
    route: dict[str, object]
    route_function: dict[str, object]
    service_functions: list[dict[str, object]]
    llm_config: LLMConfig
    current_diagram_kind: DiagramKind
    current_service_function: dict[str, object] | None
    current_function_id: str
    current_blocks: list[dict[str, object]]
    route_blocks: list[dict[str, object]]
    raw_blocks: list[dict[str, object]]
    filtered_blocks: list[dict[str, object]]
    compressed_blocks: list[dict[str, object]]
    merged_blocks: list[dict[str, object]]
    body_fragment: str
    route_artifact: DiagramArtifact
    service_artifacts: list[ServiceArtifact]
    service_index: int
    current_puml: str
    current_diff: str
    validation_feedback: str
    retry_count: int
    max_retries: int
    validator_passed: bool


def build_workflow():
    graph = StateGraph(DiagramState)
    graph.add_node("init_workflow", init_workflow_node)
    graph.add_node("generate_route_blocks", generate_route_blocks_node)
    graph.add_node("check_service_functions", check_service_functions_node)
    graph.add_node("generate_service_blocks", generate_service_blocks_node)
    graph.add_node("compress_blocks", compress_blocks_node)
    graph.add_node("render_puml", render_puml_node)
    graph.add_node("validate_puml", validate_puml_node)
    graph.add_node("save_puml", save_puml_node)
    graph.add_node("merge_puml", merge_puml_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "init_workflow")
    graph.add_edge("init_workflow", "generate_route_blocks")
    graph.add_edge("generate_route_blocks", "render_puml")
    graph.add_edge("generate_service_blocks", "compress_blocks")
    graph.add_edge("compress_blocks", "render_puml")
    graph.add_edge("render_puml", "validate_puml")
    graph.add_edge("merge_puml", "validate_puml")
    graph.add_conditional_edges(
        "validate_puml",
        route_after_validation,
        {
            "save": "save_puml",
            "retry_route": "generate_route_blocks",
            "retry_service": "generate_service_blocks",
            "retry_merge": "merge_puml",
        },
    )
    graph.add_conditional_edges(
        "save_puml",
        route_after_save,
        {
            "check_services": "check_service_functions",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "check_service_functions",
        route_after_service_check,
        {
            "generate_service": "generate_service_blocks",
            "merge": "merge_puml",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def init_workflow_node(state: DiagramState) -> DiagramState:
    _log_workflow_node(
        "init_workflow",
        state,
        service_functions=len(state.get("service_functions", [])),
    )
    return {
        "current_diagram_kind": "route",
        "current_service_function": None,
        "current_function_id": str(state["route_function"].get("function_id", "route")),
        "current_blocks": [],
        "route_blocks": [],
        "raw_blocks": [],
        "filtered_blocks": [],
        "compressed_blocks": [],
        "merged_blocks": [],
        "body_fragment": "",
        "service_artifacts": [],
        "service_index": 0,
        "current_puml": "",
        "current_diff": "",
        "retry_count": 0,
        "validation_feedback": "",
        "validator_passed": False,
    }


def generate_route_blocks_node(state: DiagramState) -> DiagramState:
    _log_workflow_node(
        "generate_route_blocks:start",
        state,
        feedback_present=bool(state.get("validation_feedback")),
    )
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
    route_blocks = _prepare_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        route_function=state["route_function"],
        append_stop=False,
    )
    _log_workflow_node(
        "generate_route_blocks:end",
        state,
        route_blocks=len(route_blocks),
    )
    return {
        "current_diagram_kind": "route",
        "current_service_function": None,
        "current_function_id": str(state["route_function"].get("function_id", "route")),
        "current_blocks": route_blocks,
        "route_blocks": route_blocks,
        "filtered_blocks": route_blocks,
        "compressed_blocks": [],
        "validator_passed": False,
    }


def check_service_functions_node(state: DiagramState) -> DiagramState:
    service_functions = state.get("service_functions", [])
    service_index = state.get("service_index", 0)
    if service_index >= len(service_functions):
        _log_workflow_node(
            "check_service_functions",
            state,
            remaining=0,
            next_node="merge_puml",
        )
        return {
            "current_diagram_kind": "merge",
            "current_service_function": None,
            "current_function_id": "merged",
            "validation_feedback": "",
            "retry_count": 0,
        }

    service_function = service_functions[service_index]
    function_id = str(service_function.get("function_id", "unknown"))
    _log_workflow_node(
        "check_service_functions",
        state,
        remaining=len(service_functions) - service_index,
        function_id=function_id,
        next_node="generate_service_blocks",
    )
    return {
        "current_diagram_kind": "service",
        "current_service_function": service_function,
        "current_function_id": function_id,
        "current_blocks": [],
        "raw_blocks": [],
        "filtered_blocks": [],
        "compressed_blocks": [],
        "current_puml": "",
        "current_diff": "",
        "validation_feedback": "",
        "retry_count": 0,
        "validator_passed": False,
    }


def generate_service_blocks_node(state: DiagramState) -> DiagramState:
    service_function = state.get("current_service_function")
    if not isinstance(service_function, dict):
        raise RuntimeError("No current service function selected")

    function_id = str(service_function.get("function_id", "unknown"))
    _log_workflow_node(
        "generate_service_blocks:start",
        state,
        function_id=function_id,
        feedback_present=bool(state.get("validation_feedback")),
    )
    prompt = FUNCTION_SERVICE_BLOCK_PROMPT.format(
        function_json=json.dumps(service_function, ensure_ascii=False, indent=2),
        feedback=state.get("validation_feedback", ""),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты генерируешь только JSON-блоки для функции.",
        user_prompt=prompt,
        node_name=f"generate_service_blocks:{function_id}",
    )
    raw_blocks = _ensure_dict_list(result.get("blocks"))
    filtered_blocks = _prepare_generated_blocks(
        raw_blocks,
        route_function=state["route_function"],
        append_stop=False,
    )
    _log_workflow_node(
        "generate_service_blocks:end",
        state,
        function_id=function_id,
        raw_blocks=len(raw_blocks),
        filtered_blocks=len(filtered_blocks),
    )
    return {
        "current_diagram_kind": "service",
        "current_function_id": function_id,
        "current_blocks": filtered_blocks,
        "raw_blocks": raw_blocks,
        "filtered_blocks": filtered_blocks,
        "compressed_blocks": [],
        "validator_passed": False,
    }


def compress_blocks_node(state: DiagramState) -> DiagramState:
    input_blocks = state.get("current_blocks", [])
    _log_workflow_node(
        "compress_blocks:start",
        state,
        input_blocks=len(input_blocks),
        diagram_kind=state.get("current_diagram_kind"),
    )
    prompt = COMPRESS_BLOCK_PROMPT.format(
        compression_rules=_compression_rules(),
        current_block_json=json.dumps(input_blocks, ensure_ascii=False, indent=2),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты сжимаешь JSON-блоки и отвечаешь только JSON.",
        user_prompt=prompt,
        node_name="compress_blocks",
    )
    compressed_blocks = _prepare_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        route_function=state["route_function"],
        append_stop=True,
    )
    _log_workflow_node(
        "compress_blocks:end",
        state,
        compressed_blocks=len(compressed_blocks),
    )
    return {
        "compressed_blocks": compressed_blocks,
        "current_blocks": compressed_blocks,
        "validator_passed": False,
    }


def render_puml_node(state: DiagramState) -> DiagramState:
    previous_puml = state.get("current_puml", "")
    diagram_kind = state.get("current_diagram_kind", "route")
    blocks = state.get("current_blocks", [])

    if diagram_kind == "service":
        function_id = state.get("current_function_id", "unknown")
        current_puml = _build_service_puml(function_id, blocks)
    else:
        current_puml = _build_route_puml(state["route"], state["route_function"], blocks)

    body_fragment = _render_blocks(blocks)
    _log_workflow_node(
        "render_puml",
        state,
        diagram_kind=diagram_kind,
        body_lines=len(body_fragment.splitlines()) if body_fragment else 0,
        puml_lines=len(current_puml.splitlines()),
        diff_lines=len(_diff_text(previous_puml, current_puml).splitlines()),
    )
    return {
        "body_fragment": body_fragment,
        "current_puml": current_puml,
        "current_diff": _diff_text(previous_puml, current_puml),
        "validator_passed": False,
    }


def validate_puml_node(state: DiagramState) -> DiagramState:
    _log_workflow_node(
        "validate_puml:start",
        state,
        diagram_kind=state.get("current_diagram_kind"),
        puml_lines=len(state["current_puml"].splitlines()),
    )
    deterministic_error = _deterministic_validate(state["current_puml"])
    prompt = VALIDATOR_PROMPT.format(
        diagram_kind=state.get("current_diagram_kind", "route"),
        route_json=json.dumps(state["route"], ensure_ascii=False, indent=2),
        route_function_json=json.dumps(state["route_function"], ensure_ascii=False, indent=2),
        service_functions_json=json.dumps(state["service_functions"], ensure_ascii=False, indent=2),
        current_puml=state["current_puml"],
        deterministic_error=deterministic_error or "none",
    )
    result = chat_json(
        state["llm_config"],
        system_prompt="Ты валидируешь activity-диаграмму. Возвращай is_valid=false только с конкретной причиной из diagram.",
        user_prompt=prompt,
        node_name="validate_puml",
    )
    feedback = _feedback_to_text(result.get("feedback"))
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
    _log_workflow_node(
        "validate_puml:end",
        state,
        deterministic_error=deterministic_error,
        llm_is_valid=llm_is_valid,
        validator_passed=validator_passed,
        next_retry_count=next_state["retry_count"],
        feedback=next_state["validation_feedback"],
    )
    return next_state


def save_puml_node(state: DiagramState) -> DiagramState:
    diagram_kind = state.get("current_diagram_kind", "route")
    function_id = state.get("current_function_id", "unknown")
    artifact: DiagramArtifact = {
        "function_id": function_id,
        "blocks": state.get("current_blocks", []),
        "current_puml": state["current_puml"],
    }
    _log_workflow_node(
        "save_puml",
        state,
        diagram_kind=diagram_kind,
        function_id=function_id,
        puml_lines=len(state["current_puml"].splitlines()),
    )

    if diagram_kind == "route":
        return {
            "route_artifact": artifact,
            "validation_feedback": "",
            "retry_count": 0,
        }
    if diagram_kind == "service":
        service_artifacts = _upsert_service_artifact(
            state.get("service_artifacts", []),
            artifact,
        )
        return {
            "service_artifacts": service_artifacts,
            "service_index": state.get("service_index", 0) + 1,
            "validation_feedback": "",
            "retry_count": 0,
        }
    return {
        "current_puml": state["current_puml"],
        "validation_feedback": "",
        "retry_count": 0,
    }


def merge_puml_node(state: DiagramState) -> DiagramState:
    route_artifact = state.get("route_artifact")
    route_blocks = route_artifact["blocks"] if isinstance(route_artifact, dict) else state.get("route_blocks", [])
    service_artifacts = state.get("service_artifacts", [])
    merged_blocks = _merge_blocks(route_blocks, service_artifacts)
    current_puml = _build_route_puml(state["route"], state["route_function"], merged_blocks)
    _log_workflow_node(
        "merge_puml",
        state,
        route_blocks=len(route_blocks),
        service_artifacts=len(service_artifacts),
        merged_blocks=len(merged_blocks),
    )
    return {
        "current_diagram_kind": "merge",
        "current_service_function": None,
        "current_function_id": "merged",
        "current_blocks": merged_blocks,
        "merged_blocks": merged_blocks,
        "current_puml": current_puml,
        "current_diff": _diff_text(state.get("current_puml", ""), current_puml),
        "validator_passed": False,
    }


def finalize_node(state: DiagramState) -> DiagramState:
    _log_workflow_node(
        "finalize:start",
        state,
        service_artifacts=len(state.get("service_artifacts", [])),
    )
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
        "route_artifact": state.get("route_artifact"),
        "service_artifacts": state.get("service_artifacts", []),
    }


def route_after_validation(state: DiagramState) -> str:
    diagram_kind = state.get("current_diagram_kind", "route")
    if state.get("validator_passed", False):
        _log_workflow_node("route_after_validation", state, next_node="save_puml")
        return "save"
    if state.get("retry_count", 0) >= state.get("max_retries", 2):
        _log_workflow_node("route_after_validation", state, next_node="raise")
        raise RuntimeError(f"Validation failed: {state.get('validation_feedback', '')}")
    next_node = {
        "route": "retry_route",
        "service": "retry_service",
        "merge": "retry_merge",
    }.get(diagram_kind, "retry_route")
    _log_workflow_node("route_after_validation", state, next_node=next_node)
    return next_node


def route_after_save(state: DiagramState) -> str:
    if state.get("current_diagram_kind") == "merge":
        _log_workflow_node("route_after_save", state, next_node="finalize")
        return "finalize"
    _log_workflow_node("route_after_save", state, next_node="check_service_functions")
    return "check_services"


def route_after_service_check(state: DiagramState) -> str:
    if isinstance(state.get("current_service_function"), dict):
        return "generate_service"
    return "merge"


def _prepare_generated_blocks(
    blocks: list[dict[str, object]],
    *,
    route_function: dict[str, object],
    append_stop: bool,
) -> list[dict[str, object]]:
    filtered_blocks = _filter_generated_blocks(blocks, route_function=route_function)
    marked_blocks = _add_terminal_markers(filtered_blocks)
    if append_stop:
        marked_blocks = _ensure_final_stop(marked_blocks)
    return marked_blocks


def _build_route_puml(
    route: dict[str, object],
    route_function: dict[str, object],
    blocks: list[dict[str, object]],
) -> str:
    header_fragment = _build_route_header(route, route_function)
    body_blocks = _strip_terminal_blocks(blocks, strip_stop=True, strip_end=False)
    footer_fragment = _build_route_footer(route, route_function)
    return _assemble_puml(header_fragment, _render_blocks(body_blocks), footer_fragment)


def _build_route_header(
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


def _build_route_footer(
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
    lines.append("stop")
    lines.append("@enduml")
    return "\n".join(lines).strip()


def _build_service_puml(function_id: str, blocks: list[dict[str, object]]) -> str:
    header_fragment = "\n".join(["@startuml", f"title {function_id}", "start"])
    body_blocks = _ensure_final_stop(blocks)
    return _assemble_puml(header_fragment, _render_blocks(body_blocks), "@enduml")


def _filter_generated_blocks(
    blocks: list[dict[str, object]],
    *,
    route_function: dict[str, object],
) -> list[dict[str, object]]:
    route_param_names = {parameter["name"] for parameter in route_function.get("parameters", [])}
    return _filter_block_list(blocks, route_param_names=route_param_names)


def _filter_block_list(
    blocks: list[dict[str, object]],
    *,
    route_param_names: set[str],
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for block in blocks:
        filtered.extend(_filter_single_block(block, route_param_names=route_param_names))
    return _normalize_linear_exception_flow(filtered)


def _filter_single_block(
    block: dict[str, object],
    *,
    route_param_names: set[str],
) -> list[dict[str, object]]:
    kind = str(block.get("kind", "")).strip().lower()
    if kind == "stop":
        return [{"kind": "action", "text": "stop"}]
    if kind == "end":
        return [{"kind": "action", "text": "end"}]
    if kind in {"except", "catch"}:
        return [{"kind": "except_marker"}]
    if kind == "return":
        return [{"kind": "action", "text": _return_action_text(block)}]
    if kind == "raise":
        text = str(block.get("text") or block.get("source") or block.get("exception") or "raise").strip()
        return [{"kind": "action", "text": text}]

    if kind == "action":
        text = str(block.get("text", "")).strip()
        if not text:
            return []
        if _is_terminal_action(text):
            return [{"kind": "action", "text": _terminal_action(text)}]
        if _is_exception_marker(text):
            return [{"kind": "except_marker"}]
        if text.lower() in {"none", "null"}:
            return [{"kind": "action", "text": "return None"}]
        if _is_return_action(text):
            return [{"kind": "action", "text": text}]
        if _is_route_level_duplicate(text, route_param_names):
            return []
        if _is_control_flow_noise(text):
            return []
        return [{"kind": "action", "text": text}]

    if kind == "partition":
        title = str(block.get("title", "Block")).strip()
        inner = _filter_block_list(_ensure_dict_list(block.get("blocks")), route_param_names=route_param_names)
        if not inner:
            return []
        return [{"kind": "partition", "title": title, "blocks": inner}]

    if kind == "if":
        condition = str(block.get("condition", "")).strip()
        then_blocks = _filter_block_list(_ensure_dict_list(block.get("then")), route_param_names=route_param_names)
        else_blocks = _filter_block_list(_ensure_dict_list(block.get("else")), route_param_names=route_param_names)
        if not then_blocks and not else_blocks:
            return []
        if not then_blocks and else_blocks:
            return [{"kind": "if", "condition": f"not {condition}", "then": else_blocks}]
        payload: dict[str, object] = {"kind": "if", "condition": condition, "then": then_blocks}
        if else_blocks:
            payload["else"] = else_blocks
        return [payload]

    if kind == "try":
        success_blocks = _filter_block_list(
            _ensure_dict_list(block.get("then") or block.get("blocks")),
            route_param_names=route_param_names,
        )
        error_blocks = _filter_block_list(
            _ensure_dict_list(block.get("else") or block.get("except") or block.get("catch")),
            route_param_names=route_param_names,
        )
        if success_blocks and error_blocks:
            return [
                {
                    "kind": "if",
                    "condition": "ошибка при выполнении",
                    "then": error_blocks,
                    "else": success_blocks,
                }
            ]
        if error_blocks:
            return [
                {
                    "kind": "if",
                    "condition": "ошибка при выполнении",
                    "then": error_blocks,
                }
            ]
        return success_blocks

    nested_blocks = _filter_block_list(
        _ensure_dict_list(block.get("blocks") or block.get("then")),
        route_param_names=route_param_names,
    )
    if nested_blocks:
        return nested_blocks

    return []


def _normalize_linear_exception_flow(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = [_normalize_nested_exception_flow(block) for block in blocks]
    except_index = next(
        (index for index, block in enumerate(normalized) if block.get("kind") == "except_marker"),
        -1,
    )
    if except_index < 0:
        return [block for block in normalized if block.get("kind") != "except_marker"]

    success_blocks = [block for block in normalized[:except_index] if block.get("kind") != "except_marker"]
    error_blocks = [block for block in normalized[except_index + 1 :] if block.get("kind") != "except_marker"]
    if not error_blocks:
        return success_blocks
    if not success_blocks:
        return [{"kind": "if", "condition": "ошибка при выполнении", "then": error_blocks}]
    return [
        {
            "kind": "if",
            "condition": "ошибка при выполнении",
            "then": error_blocks,
            "else": success_blocks,
        }
    ]


def _normalize_nested_exception_flow(block: dict[str, object]) -> dict[str, object]:
    kind = block.get("kind")
    if kind == "if":
        next_block: dict[str, object] = {
            "kind": "if",
            "condition": block.get("condition", "condition"),
            "then": _normalize_linear_exception_flow(_ensure_dict_list(block.get("then"))),
        }
        else_blocks = _ensure_dict_list(block.get("else"))
        if else_blocks:
            next_block["else"] = _normalize_linear_exception_flow(else_blocks)
        return next_block
    if kind == "partition":
        return {
            "kind": "partition",
            "title": block.get("title", "Block"),
            "blocks": _normalize_linear_exception_flow(_ensure_dict_list(block.get("blocks"))),
        }
    return block


def _add_terminal_markers(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    marked_blocks: list[dict[str, object]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "action":
            marked_blocks.append(block)
            text = str(block.get("text", ""))
            if _is_raise_action(text) and not _last_block_is(marked_blocks, "end"):
                marked_blocks.append({"kind": "action", "text": "end"})
            elif _is_return_action(text) and not _last_block_is(marked_blocks, "stop"):
                marked_blocks.append({"kind": "action", "text": "stop"})
        elif kind == "if":
            next_block: dict[str, object] = {
                "kind": "if",
                "condition": block.get("condition", "condition"),
                "then": _add_terminal_markers(_ensure_dict_list(block.get("then"))),
            }
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                next_block["else"] = _add_terminal_markers(else_blocks)
            marked_blocks.append(next_block)
        elif kind == "partition":
            marked_blocks.append(
                {
                    "kind": "partition",
                    "title": block.get("title", "Block"),
                    "blocks": _add_terminal_markers(_ensure_dict_list(block.get("blocks"))),
                }
            )
    return _dedupe_adjacent_terminal_blocks(marked_blocks)


def _ensure_final_stop(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    if _block_list_has_terminal_tail(blocks):
        return blocks
    return [*blocks, {"kind": "action", "text": "stop"}]


def _strip_terminal_blocks(
    blocks: list[dict[str, object]],
    *,
    strip_stop: bool,
    strip_end: bool,
) -> list[dict[str, object]]:
    stripped: list[dict[str, object]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "action":
            terminal = _terminal_action(str(block.get("text", "")))
            if terminal == "stop" and strip_stop:
                continue
            if terminal == "end" and strip_end:
                continue
            stripped.append(block)
        elif kind == "if":
            next_block: dict[str, object] = {
                "kind": "if",
                "condition": block.get("condition", "condition"),
                "then": _strip_terminal_blocks(
                    _ensure_dict_list(block.get("then")),
                    strip_stop=strip_stop,
                    strip_end=strip_end,
                ),
            }
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                next_block["else"] = _strip_terminal_blocks(
                    else_blocks,
                    strip_stop=strip_stop,
                    strip_end=strip_end,
                )
            stripped.append(next_block)
        elif kind == "partition":
            stripped.append(
                {
                    "kind": "partition",
                    "title": block.get("title", "Block"),
                    "blocks": _strip_terminal_blocks(
                        _ensure_dict_list(block.get("blocks")),
                        strip_stop=strip_stop,
                        strip_end=strip_end,
                    ),
                }
            )
    return stripped


def _merge_blocks(
    route_blocks: list[dict[str, object]],
    service_artifacts: list[ServiceArtifact],
) -> list[dict[str, object]]:
    used_function_ids: set[str] = set()
    merged_blocks = _merge_block_list(route_blocks, service_artifacts, used_function_ids)
    for service_artifact in service_artifacts:
        function_id = service_artifact["function_id"]
        if function_id not in used_function_ids:
            merged_blocks.extend(_blocks_for_route_merge(service_artifact["blocks"]))
    return _strip_terminal_blocks(merged_blocks, strip_stop=True, strip_end=False)


def _merge_block_list(
    blocks: list[dict[str, object]],
    service_artifacts: list[ServiceArtifact],
    used_function_ids: set[str],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "action":
            merged.append(block)
            service_artifact = _find_called_service_artifact(str(block.get("text", "")), service_artifacts)
            if service_artifact is not None:
                used_function_ids.add(service_artifact["function_id"])
                merged.extend(_blocks_for_route_merge(service_artifact["blocks"]))
        elif kind == "if":
            next_block: dict[str, object] = {
                "kind": "if",
                "condition": block.get("condition", "condition"),
                "then": _merge_block_list(
                    _ensure_dict_list(block.get("then")),
                    service_artifacts,
                    used_function_ids,
                ),
            }
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                next_block["else"] = _merge_block_list(else_blocks, service_artifacts, used_function_ids)
            merged.append(next_block)
        elif kind == "partition":
            merged.append(
                {
                    "kind": "partition",
                    "title": block.get("title", "Block"),
                    "blocks": _merge_block_list(
                        _ensure_dict_list(block.get("blocks")),
                        service_artifacts,
                        used_function_ids,
                    ),
                }
            )
    return merged


def _blocks_for_route_merge(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    merged_blocks = _rewrite_service_returns_for_route_merge(blocks)
    return _strip_terminal_blocks(merged_blocks, strip_stop=True, strip_end=False)


def _rewrite_service_returns_for_route_merge(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    rewritten: list[dict[str, object]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "action":
            text = str(block.get("text", ""))
            if _is_return_action(text):
                rewritten.append({"kind": "action", "text": _route_merge_return_text(text)})
            else:
                rewritten.append(block)
        elif kind == "if":
            next_block: dict[str, object] = {
                "kind": "if",
                "condition": block.get("condition", "condition"),
                "then": _rewrite_service_returns_for_route_merge(_ensure_dict_list(block.get("then"))),
            }
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                next_block["else"] = _rewrite_service_returns_for_route_merge(else_blocks)
            rewritten.append(next_block)
        elif kind == "partition":
            rewritten.append(
                {
                    "kind": "partition",
                    "title": block.get("title", "Block"),
                    "blocks": _rewrite_service_returns_for_route_merge(
                        _ensure_dict_list(block.get("blocks"))
                    ),
                }
            )
    return rewritten


def _find_called_service_artifact(
    text: str,
    service_artifacts: list[ServiceArtifact],
) -> ServiceArtifact | None:
    lowered = text.lower()
    for service_artifact in service_artifacts:
        function_id = service_artifact["function_id"]
        short_name = function_id.rsplit(".", maxsplit=1)[-1]
        if function_id.lower() in lowered or short_name.lower() in lowered:
            return service_artifact
    if len(service_artifacts) == 1 and _looks_like_service_call(text):
        return service_artifacts[0]
    return None


def _upsert_service_artifact(
    service_artifacts: list[ServiceArtifact],
    artifact: DiagramArtifact,
) -> list[ServiceArtifact]:
    next_artifact: ServiceArtifact = {
        "function_id": artifact["function_id"],
        "blocks": artifact["blocks"],
        "current_puml": artifact["current_puml"],
    }
    result: list[ServiceArtifact] = []
    replaced = False
    for service_artifact in service_artifacts:
        if service_artifact["function_id"] == next_artifact["function_id"]:
            result.append(next_artifact)
            replaced = True
        else:
            result.append(service_artifact)
    if not replaced:
        result.append(next_artifact)
    return result


def _is_route_level_duplicate(text: str, route_param_names: set[str]) -> bool:
    lowered = text.lower()
    normalized = " ".join(lowered.strip().strip(".:;").split())
    if "request." in lowered:
        return True
    if normalized in {"request", "receive request", "запрос", "получить запрос"}:
        return True
    if "request" in lowered and ("receive" in lowered or "extract" in lowered):
        return True
    if "запрос" in normalized and (
        "получ" in normalized or "принят" in normalized or "принять" in normalized or "извлеч" in normalized
    ):
        return True
    if lowered.startswith("build response") or lowered.startswith("http "):
        return True
    if "response" in lowered and ("return success" in lowered or "send http" in lowered or "success response" in lowered):
        return True
    if lowered.startswith("return ") and "response" in lowered:
        return True
    if normalized.startswith(("вернуть ответ", "возвратить ответ", "сформировать ответ", "отправить ответ")):
        return True
    if "ответ" in normalized and ("код" in normalized or "status" in normalized or "http" in normalized):
        return True
    if normalized.startswith(("http ", "status code ", "код ответа", "статус ответа")):
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


def _is_control_flow_noise(text: str) -> bool:
    normalized = " ".join(text.strip().lower().strip(".:;").split())
    if "try block" in normalized or "блок try" in normalized:
        return True
    if "блок except" in normalized or "блок catch" in normalized:
        return True
    return normalized in {
        "try",
        "try block",
        "start try",
        "start try block",
        "begin try",
        "begin try block",
        "enter try",
        "enter try block",
        "except",
        "catch",
        "exception handler",
    } or normalized.startswith(
        (
            "start try block ",
            "begin try block ",
            "enter try block ",
            "start except",
            "begin except",
            "enter except",
            "except ",
            "catch ",
        )
    )


def _looks_like_service_call(text: str) -> bool:
    normalized = " ".join(text.strip().lower().strip(".:;").split())
    if not normalized:
        return False
    call_markers = ("call", "invoke", "run", "use", "вызва", "вызов", "обрат", "использ")
    service_markers = ("service", "сервис")
    return any(marker in normalized for marker in call_markers) and any(
        marker in normalized for marker in service_markers
    )


def _compression_rules() -> str:
    return (
        "- Удаляй мелкие технические шаги.\n"
        "- Схлопывай соседние preparation actions.\n"
        "- Не удаляй DB write/read, entity/model creation, ошибки.\n"
        "- Не удаляй terminal actions end/stop."
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
            terminal = _terminal_action(str(block.get("text", "")))
            if terminal:
                lines.append(f"{prefix}{terminal}")
                continue
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
    normalized_text = text.replace("\r\n", "\n")
    if "@startuml" not in normalized_text:
        return "Missing @startuml"
    if "@enduml" not in normalized_text:
        return "Missing @enduml"
    if "\nstart\n" not in normalized_text:
        return "Missing start"
    non_empty_lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    if len(non_empty_lines) < 2 or non_empty_lines[-1] != "@enduml":
        return "Missing @enduml"
    if non_empty_lines[-2] not in {"end", "stop"}:
        return "Missing terminal end/stop before @enduml"
    if normalized_text.count("if (") != normalized_text.count("endif"):
        return "Unbalanced if/endif count"
    if normalized_text.count("@startuml") != 1 or normalized_text.count("@enduml") != 1:
        return "Diagram must contain exactly one @startuml and one @enduml"
    if normalized_text.count("(") != normalized_text.count(")"):
        return "Unbalanced parentheses count"
    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.lower().strip(":;")
        if normalized in {"try", "try:", "except", "except:", "catch", "catch:", "end try", "endtry"}:
            return f"Unsupported pseudo-syntax found: {line}"
        if line.startswith(("@startuml", "@enduml", "title ", "start", "end", "stop", "if ", "else", "endif", "partition ", "}", ":", "note")):
            continue
        return f"Unsupported line syntax: {line}"
    return None


def _assemble_puml(header_fragment: str, body_fragment: str, footer_fragment: str) -> str:
    parts = [header_fragment.rstrip()]
    if body_fragment.strip():
        parts.append("")
        parts.append(body_fragment.rstrip())
    if footer_fragment.strip():
        if footer_fragment.strip() != "@enduml":
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
    text = text.replace("during the try block", "during execution")
    text = text.replace("during try block", "during execution")
    text = text.replace("in the try block", "during execution")
    return text or "condition"


def _ensure_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _feedback_to_text(value: object) -> str:
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _return_action_text(block: dict[str, object]) -> str:
    text = str(block.get("text") or block.get("source") or "").strip()
    if not text or text.lower() in {"none", "null"}:
        return "return None"
    if _is_return_action(text):
        return text
    return f"return {text}"


def _route_merge_return_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    lowered = normalized.lower()
    for prefix in ("return ", "вернуть "):
        if lowered.startswith(prefix):
            value = normalized[len(prefix) :].strip() or "None"
            return f"service_result = {value}"
    if lowered in {"return", "вернуть"}:
        return "service_result = None"
    return f"service_result = {normalized}"


def _is_exception_marker(text: str) -> bool:
    normalized = " ".join(text.strip().lower().strip(".:;").split())
    return (
        normalized.startswith("except")
        or normalized.startswith("catch")
        or "блок except" in normalized
        or "блок catch" in normalized
        or normalized in {"exception handler", "обработчик исключения"}
    )


def _is_raise_action(text: str) -> bool:
    normalized = " ".join(text.lower().strip(".:;").split())
    return (
        normalized.startswith("raise ")
        or " raise " in f" {normalized} "
        or "re-raise" in normalized
        or "выброс" in normalized
        or "выбрасы" in normalized
    )


def _is_return_action(text: str) -> bool:
    normalized = " ".join(text.lower().strip(".:;").split())
    return normalized.startswith("return ") or normalized.startswith("вернуть ") or normalized.startswith("возвращ")


def _is_terminal_action(text: str) -> bool:
    return _terminal_action(text) in {"end", "stop"}


def _terminal_action(text: str) -> str:
    normalized = text.strip().removeprefix(":").removesuffix(";").strip().lower()
    if normalized in {"end", "stop"}:
        return normalized
    return ""


def _last_block_is(blocks: list[dict[str, object]], terminal: str) -> bool:
    if not blocks:
        return False
    last_block = blocks[-1]
    return last_block.get("kind") == "action" and _terminal_action(str(last_block.get("text", ""))) == terminal


def _block_list_has_terminal_tail(blocks: list[dict[str, object]]) -> bool:
    return bool(blocks) and blocks[-1].get("kind") == "action" and _is_terminal_action(str(blocks[-1].get("text", "")))


def _dedupe_adjacent_terminal_blocks(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    for block in blocks:
        if (
            block.get("kind") == "action"
            and deduped
            and deduped[-1].get("kind") == "action"
            and _terminal_action(str(block.get("text", "")))
            and _terminal_action(str(block.get("text", ""))) == _terminal_action(str(deduped[-1].get("text", "")))
        ):
            continue
        deduped.append(block)
    return deduped


def _diff_text(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="before.puml",
            tofile="after.puml",
        )
    )


def _log_workflow_node(node_name: str, state: DiagramState, **payload: object) -> None:
    try:
        route = state.get("route", {})
        log_event(
            "workflow_node",
            {
                "node_name": node_name,
                "route_id": route.get("route_id") if isinstance(route, dict) else None,
                "retry_count": state.get("retry_count", 0),
                **payload,
            },
        )
    except Exception:
        pass
