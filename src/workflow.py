from __future__ import annotations

import difflib
import json
from typing import Literal, TypedDict

from src.llm import LLMConfig, chat_json
from src.prompts import DEFAULT_PROMPTS, render_prompt


DiagramKind = Literal["route", "service", "merge"]


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
    prompt_overrides: dict[str, str]
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
    response_routes: list[dict[str, str]]
    response_artifacts: list[dict[str, str]]


def build_workflow():
    from langgraph.graph import END, START, StateGraph

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
    return {
        "current_diagram_kind": "route",
        "current_service_function": None,
        "current_function_id": str(state["route_function"].get("function_id", "route")),
        "prompt_overrides": state.get("prompt_overrides", {}),
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
        "response_routes": [],
        "response_artifacts": [],
    }


def generate_route_blocks_node(state: DiagramState) -> DiagramState:
    feedback = state.get("validation_feedback", "")
    prompt = render_prompt(
        _prompt_template(state, "route-user"),
        **_route_prompt_render_context(state, feedback),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt=_prompt_template(state, "route-system"),
        user_prompt=prompt,
        node_name="generate_route_blocks",
    )
    route_blocks = _prepare_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        route_function=state["route_function"],
        append_stop=False,
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
        return {
            "current_diagram_kind": "merge",
            "current_service_function": None,
            "current_function_id": "merged",
            "validation_feedback": "",
            "retry_count": 0,
        }

    service_function = service_functions[service_index]
    function_id = str(service_function.get("function_id", "unknown"))
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
    feedback = state.get("validation_feedback", "")
    prompt = render_prompt(
        _prompt_template(state, "service-user"),
        **_service_prompt_render_context(state, service_function, feedback),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt=_prompt_template(state, "service-system"),
        user_prompt=prompt,
        node_name=f"generate_service_blocks:{function_id}",
    )
    raw_blocks = _ensure_dict_list(result.get("blocks"))
    filtered_blocks = _prepare_generated_blocks(
        raw_blocks,
        route_function=state["route_function"],
        append_stop=False,
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
    feedback = state.get("validation_feedback", "")
    prompt = render_prompt(
        _prompt_template(state, "compress-user"),
        **_compress_prompt_render_context(input_blocks, feedback),
    )
    result = chat_json(
        state["llm_config"],
        system_prompt=_prompt_template(state, "compress-system"),
        user_prompt=prompt,
        node_name="compress",
    )
    compressed_blocks = _prepare_generated_blocks(
        _ensure_dict_list(result.get("blocks")),
        route_function=state["route_function"],
        append_stop=True,
    )
    compressed_blocks = _cleanup_compressed_blocks(compressed_blocks)
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
    return {
        "body_fragment": body_fragment,
        "current_puml": current_puml,
        "current_diff": _diff_text(previous_puml, current_puml),
        "validator_passed": False,
    }


def validate_puml_node(state: DiagramState) -> DiagramState:
    deterministic_error = _deterministic_validate(state["current_puml"])
    validator_passed = deterministic_error is None
    validation_parts: list[str] = []
    if deterministic_error:
        validation_parts.append(deterministic_error)

    next_state: DiagramState = {
        "validator_passed": validator_passed,
        "validation_feedback": "" if validator_passed else "\n".join(validation_parts),
    }
    next_state["retry_count"] = 0 if validator_passed else state.get("retry_count", 0) + 1
    return next_state


def save_puml_node(state: DiagramState) -> DiagramState:
    diagram_kind = state.get("current_diagram_kind", "route")
    function_id = state.get("current_function_id", "unknown")
    artifact: DiagramArtifact = {
        "function_id": function_id,
        "blocks": state.get("current_blocks", []),
        "current_puml": state["current_puml"],
    }
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
    final_error = _deterministic_validate(state["current_puml"])
    if final_error:
        raise RuntimeError(f"Final diagram is invalid: {final_error}")
    for service_artifact in state.get("service_artifacts", []):
        service_error = _deterministic_validate(service_artifact["current_puml"])
        if service_error:
            function_id = service_artifact.get("function_id", "unknown")
            raise RuntimeError(f"Service diagram {function_id} is invalid: {service_error}")
    response_routes = [
        {
            "name": _route_filename(state["route"]),
            "puml": state["current_puml"],
        }
    ]
    response_artifacts = [
        {
            "name": _service_filename(state["route"], service_artifact["function_id"]),
            "puml": service_artifact["current_puml"],
        }
        for service_artifact in state.get("service_artifacts", [])
    ]
    return {
        "current_puml": state["current_puml"],
        "route_artifact": state.get("route_artifact"),
        "service_artifacts": state.get("service_artifacts", []),
        "response_routes": response_routes,
        "response_artifacts": response_artifacts,
    }


def route_after_validation(state: DiagramState) -> str:
    diagram_kind = state.get("current_diagram_kind", "route")
    if state.get("validator_passed", False):
        return "save"
    if state.get("retry_count", 0) > state.get("max_retries", 3):
        raise RuntimeError(f"Validation failed: {state.get('validation_feedback', '')}")
    next_node = {
        "route": "retry_route",
        "service": "retry_service",
        "merge": "retry_merge",
    }.get(diagram_kind, "retry_route")
    return next_node


def route_after_save(state: DiagramState) -> str:
    if state.get("current_diagram_kind") == "merge":
        return "finalize"
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
            action_text = str(block.get("text", ""))
            service_artifact = (
                _find_called_service_artifact(action_text, service_artifacts)
                if _should_inline_service_after_action(action_text)
                else None
            )
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


def _should_inline_service_after_action(text: str) -> bool:
    normalized = " ".join(text.strip().lower().strip(".:;").split())
    if not normalized:
        return False
    if _is_return_action(text):
        return False
    if "результат вызова" in normalized or "return result" in normalized:
        return False
    return True


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


def _cleanup_compressed_blocks(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    cleaned_nested: list[dict[str, object]] = []
    for block in blocks:
        kind = block.get("kind")
        if kind == "if":
            next_block: dict[str, object] = {
                "kind": "if",
                "condition": block.get("condition", "condition"),
                "then": _cleanup_compressed_blocks(_ensure_dict_list(block.get("then"))),
            }
            else_blocks = _ensure_dict_list(block.get("else"))
            if else_blocks:
                next_block["else"] = _cleanup_compressed_blocks(else_blocks)
            cleaned_nested.append(next_block)
        elif kind == "partition":
            inner = _cleanup_compressed_blocks(_ensure_dict_list(block.get("blocks")))
            if inner:
                cleaned_nested.append(
                    {
                        "kind": "partition",
                        "title": block.get("title", "Block"),
                        "blocks": inner,
                    }
                )
        else:
            cleaned_nested.append(block)

    collapsed = _collapse_technical_exception_if(cleaned_nested)
    if collapsed is not None:
        return _dedupe_adjacent_terminal_blocks(collapsed)

    result: list[dict[str, object]] = []
    for index, block in enumerate(cleaned_nested):
        if block.get("kind") != "action":
            result.append(block)
            continue
        text = str(block.get("text", ""))
        if _is_compress_removable_log_action(text):
            continue
        next_block = cleaned_nested[index + 1] if index + 1 < len(cleaned_nested) else None
        if _is_bare_raise_action(text) and isinstance(next_block, dict):
            if next_block.get("kind") == "action" and _terminal_action(str(next_block.get("text", ""))) == "end":
                continue
        result.append(block)
    return _dedupe_adjacent_terminal_blocks(result)


def _collapse_technical_exception_if(blocks: list[dict[str, object]]) -> list[dict[str, object]] | None:
    trailing_stop: list[dict[str, object]] = []
    candidate_blocks = blocks
    if len(blocks) >= 2:
        last_block = blocks[-1]
        if last_block.get("kind") == "action" and _terminal_action(str(last_block.get("text", ""))) == "stop":
            trailing_stop = [last_block]
            candidate_blocks = blocks[:-1]
    if len(candidate_blocks) != 1:
        return None
    block = candidate_blocks[0]
    if block.get("kind") != "if":
        return None
    condition = " ".join(str(block.get("condition", "")).strip().lower().split())
    if condition != "ошибка при выполнении":
        return None
    error_blocks = _ensure_dict_list(block.get("then"))
    success_blocks = _ensure_dict_list(block.get("else"))
    if not success_blocks or not _is_pure_technical_exception_branch(error_blocks):
        return None
    return [*success_blocks, *trailing_stop]


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


def _route_generation_instruction(feedback: str) -> str:
    if feedback.strip():
        return (
            "Ты исправляешь предыдущий невалидный JSON route-блоков. "
            "Используй feedback как список конкретных ошибок, сохрани смысл IR и выдай полный исправленный JSON заново."
        )
    return "Ты превращаешь IR route-функции в корректный JSON блоков activity-диаграммы."


def _service_generation_instruction(feedback: str) -> str:
    if feedback.strip():
        return (
            "Ты исправляешь предыдущий невалидный JSON service-блоков. "
            "Используй feedback как список конкретных ошибок, сохрани смысл IR функции и выдай полный исправленный JSON заново."
        )
    return "Ты превращаешь IR service-функции в корректный JSON блоков activity-диаграммы."


def _compression_instruction(feedback: str) -> str:
    if feedback.strip():
        return (
            "Ты исправляешь предыдущую неудачную компрессию. "
            "Строго учитывай feedback валидатора. Если в feedback сказано сохранить или явно показать шаг, ветку, service call, "
            "service return, route response, success completion, error handling, end или stop, не удаляй, не схлопывай и не маскируй их."
        )
    return "Ты аккуратно сжимаешь JSON-блок, не теряя важный control flow и terminal steps."


def _feedback_section(feedback: str) -> str:
    normalized = feedback.strip()
    if not normalized:
        return ""
    return "<feedback>\n" + normalized + "\n</feedback>"


def _prompt_template(state: DiagramState, role: str) -> str:
    overrides = state.get("prompt_overrides", {})
    return overrides.get(role, DEFAULT_PROMPTS[role])


def _route_prompt_render_context(state: DiagramState, feedback: str) -> dict[str, object]:
    route_context = _route_prompt_context(state["route"], state["route_function"])
    return {
        "instruction": _route_generation_instruction(feedback),
        "route_context_json": json.dumps(route_context, ensure_ascii=False, indent=2),
        "feedback_section": _feedback_section(feedback),
        "endpoint": _route_endpoint_label(state["route"], state["route_function"]),
        "code": _extract_function_code(state["route_function"]),
        "puml_gen": _joined_puml_context(state),
    }


def _service_prompt_render_context(
    state: DiagramState,
    service_function: dict[str, object],
    feedback: str,
) -> dict[str, object]:
    function_context = _service_prompt_context(service_function)
    return {
        "instruction": _service_generation_instruction(feedback),
        "function_context_json": json.dumps(function_context, ensure_ascii=False, indent=2),
        "feedback_section": _feedback_section(feedback),
        "endpoint": _route_endpoint_label(state["route"], state["route_function"]),
        "code": _extract_function_code(service_function),
        "puml_gen": _joined_puml_context(state),
        "function_id": service_function.get("function_id", "unknown"),
    }


def _compress_prompt_render_context(
    input_blocks: list[dict[str, object]],
    feedback: str,
) -> dict[str, object]:
    return {
        "instruction": _compression_instruction(feedback),
        "compression_rules": _compression_rules(),
        "feedback_section": _feedback_section(feedback),
        "current_block_json": json.dumps(input_blocks, ensure_ascii=False, indent=2),
    }


def _joined_puml_context(state: DiagramState) -> str:
    fragments: list[str] = []
    current_puml = state.get("current_puml", "").strip()
    if current_puml:
        fragments.append(current_puml)
    route_artifact = state.get("route_artifact")
    if isinstance(route_artifact, dict):
        route_puml = str(route_artifact.get("current_puml", "")).strip()
        if route_puml and route_puml not in fragments:
            fragments.append(route_puml)
    for service_artifact in state.get("service_artifacts", []):
        service_puml = str(service_artifact.get("current_puml", "")).strip()
        if service_puml and service_puml not in fragments:
            fragments.append(service_puml)
    return "\n\n".join(fragments)


def _extract_function_code(function: dict[str, object]) -> str:
    for key in ("code", "source", "body", "raw_code"):
        value = function.get(key)
        if isinstance(value, str) and value.strip():
            return value
    signature = function.get("signature")
    if isinstance(signature, str) and signature.strip():
        step_lines = []
        for step in _prompt_steps(function):
            source = step.get("source")
            if isinstance(source, str) and source.strip():
                step_lines.append(source)
        if step_lines:
            return signature + "\n" + "\n".join(step_lines)
        return signature
    return ""


def _route_endpoint_label(
    route: dict[str, object],
    route_function: dict[str, object],
) -> str:
    endpoint = route.get("endpoint") or route.get("route_id")
    if isinstance(endpoint, str) and endpoint.strip():
        return endpoint
    http_meta = route_function.get("http", {})
    if isinstance(http_meta, dict):
        method = str(http_meta.get("method", "")).strip()
        path = str(http_meta.get("path", "")).strip()
        return " ".join(part for part in (method, path) if part)
    return ""


def _route_filename(route: dict[str, object]) -> str:
    route_id = str(route.get("route_id", "route"))
    return f"{_slug(route_id)}.activity.puml"


def _service_filename(route: dict[str, object], function_id: str) -> str:
    route_id = str(route.get("route_id", "route"))
    return f"{_slug(route_id)}.{_slug(function_id)}.activity.puml"


def _slug(value: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-"} else "_" for char in value]
    return "_".join(part for part in "".join(chars).split("_") if part) or "unknown"


def _route_prompt_context(
    route: dict[str, object],
    route_function: dict[str, object],
) -> dict[str, object]:
    http_meta = route_function.get("http", {})
    return {
        "route_id": route.get("route_id"),
        "http": {
            "method": http_meta.get("method"),
            "path": http_meta.get("path"),
            "status_code": http_meta.get("status_code"),
        },
        "signature": route_function.get("signature"),
        "parameters": _prompt_parameters(route_function),
        "steps": _prompt_steps(route_function),
    }


def _service_prompt_context(function: dict[str, object]) -> dict[str, object]:
    return {
        "function_id": function.get("function_id"),
        "signature": function.get("signature"),
        "steps": _prompt_steps(function, strip_technical_exceptions=True),
    }


def _validator_prompt_context(
    state: DiagramState,
    *,
    diagram_kind: DiagramKind,
) -> dict[str, object]:
    if diagram_kind == "service":
        service_function = state.get("current_service_function")
        if isinstance(service_function, dict):
            return {
                "service_function": _service_prompt_context(service_function),
            }
        return {
            "service_function": {
                "function_id": state.get("current_function_id", "unknown"),
            }
        }
    if diagram_kind == "merge":
        return {
            "route": _route_prompt_context(state["route"], state["route_function"]),
            "service_functions": [
                _service_prompt_context(function)
                for function in state.get("service_functions", [])
            ],
        }
    return {
        "route": _route_prompt_context(state["route"], state["route_function"]),
    }


def _prompt_parameters(function: dict[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for parameter in function.get("parameters", []):
        if not isinstance(parameter, dict):
            continue
        result.append(
            {
                "name": parameter.get("name"),
                "location": parameter.get("location"),
                "type": parameter.get("type"),
            }
        )
    return result


def _prompt_steps(
    function: dict[str, object],
    *,
    strip_technical_exceptions: bool = False,
) -> list[dict[str, object]]:
    steps_source = function.get("steps", [])
    if strip_technical_exceptions:
        steps_source = _strip_technical_try_except_steps(steps_source)
    result: list[dict[str, object]] = []
    for step in steps_source:
        if not isinstance(step, dict):
            continue
        compact_step = {
            "kind": step.get("kind"),
            "source": step.get("source"),
        }
        if not compact_step["source"]:
            for key in ("condition", "target", "value", "result"):
                if step.get(key) is not None:
                    compact_step[key] = step.get(key)
        result.append({key: value for key, value in compact_step.items() if value is not None})
    return result


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


def _is_bare_raise_action(text: str) -> bool:
    normalized = " ".join(text.lower().strip(".:;").split())
    return normalized == "raise"


def _is_return_action(text: str) -> bool:
    normalized = " ".join(text.lower().strip(".:;").split())
    return normalized.startswith("return ") or normalized.startswith("вернуть ") or normalized.startswith("возвращ")


def _is_compress_removable_log_action(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    return normalized.startswith(("logging.", "logger.")) or normalized.startswith(("log ", "logging "))


def _is_pure_technical_exception_branch(blocks: list[dict[str, object]]) -> bool:
    if not blocks:
        return False
    for block in blocks:
        if block.get("kind") != "action":
            return False
        text = str(block.get("text", ""))
        terminal = _terminal_action(text)
        if terminal == "end":
            continue
        if _is_compress_removable_log_action(text):
            continue
        if _is_bare_raise_action(text):
            continue
        return False
    return True


def _strip_technical_try_except_steps(steps: object) -> list[dict[str, object]]:
    if not isinstance(steps, list):
        return []
    normalized_steps = [step for step in steps if isinstance(step, dict)]
    try_index = next((index for index, step in enumerate(normalized_steps) if step.get("kind") == "try"), -1)
    except_index = next((index for index, step in enumerate(normalized_steps) if step.get("kind") == "except"), -1)
    if try_index < 0 or except_index < 0 or except_index <= try_index:
        return normalized_steps
    except_step = normalized_steps[except_index]
    exception_name = str(except_step.get("exception") or "").strip()
    if exception_name not in {"Exception", "exception"}:
        return normalized_steps
    success_steps = normalized_steps[try_index + 1 : except_index]
    error_tail = normalized_steps[except_index + 1 :]
    if not success_steps or not _is_technical_exception_tail_steps(error_tail):
        return normalized_steps
    return success_steps


def _is_technical_exception_tail_steps(steps: list[dict[str, object]]) -> bool:
    if not steps:
        return False
    for step in steps:
        kind = str(step.get("kind", "")).strip().lower()
        source = str(step.get("source", "")).strip()
        if kind.startswith("log") or _is_compress_removable_log_action(source):
            continue
        if kind == "raise" and _is_bare_raise_action(source or "raise"):
            continue
        return False
    return True


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
