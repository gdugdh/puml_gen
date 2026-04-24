from __future__ import annotations

import json

from src.generator import generate_from_file
from src.llm import LLMConfig
from src.workflow import _deterministic_validate
from src.workflow import _filter_generated_blocks
from src.workflow import _merge_blocks
from src.workflow import _prepare_generated_blocks
from src.workflow import _render_blocks
from src.workflow import _route_prompt_context
from src.workflow import _service_prompt_context
from src.workflow import _validator_prompt_context
from src.workflow import compress_blocks_node
from src.workflow import generate_service_blocks_node
from src.workflow import route_after_validation
from src.workflow import validate_puml_node


def test_validate_puml_node_rejects_llm_is_valid_false(monkeypatch):
    def fake_chat_json(*args, **kwargs):
        return {"is_valid": False, "feedback": "important branch is missing"}

    monkeypatch.setattr("src.workflow.chat_json", fake_chat_json)

    state = {
        "route": {"route_id": "POST /auth"},
        "route_function": {"function_id": "handler", "parameters": []},
        "service_functions": [],
        "llm_config": LLMConfig(api_key="test", model="test", base_url="https://example.test"),
        "current_puml": "@startuml\ntitle POST /auth\nstart\n:Do work;\nend\n@enduml\n",
        "retry_count": 1,
    }

    result = validate_puml_node(state)

    assert result["validator_passed"] is False
    assert result["retry_count"] == 2
    assert "important branch is missing" in result["validation_feedback"]
    assert "is_valid=False" in result["validation_feedback"]


def test_validate_puml_node_includes_suggested_fix_and_resets_retry_count(monkeypatch):
    def fake_chat_json(*args, **kwargs):
        return {"is_valid": True, "feedback": "", "suggested_fix": ""}

    monkeypatch.setattr("src.workflow.chat_json", fake_chat_json)

    state = {
        "route": {"route_id": "POST /auth"},
        "route_function": {"function_id": "handler", "parameters": []},
        "service_functions": [],
        "llm_config": LLMConfig(api_key="test", model="test", base_url="https://example.test"),
        "current_puml": "@startuml\ntitle POST /auth\nstart\n:Do work;\nstop\n@enduml\n",
        "retry_count": 3,
    }

    result = validate_puml_node(state)

    assert result["validator_passed"] is True
    assert result["retry_count"] == 0
    assert result["validation_feedback"] == ""


def test_generate_from_file_writes_route_and_service_artifacts(tmp_path, monkeypatch):
    input_path = tmp_path / "input.json"
    output_dir = tmp_path / "output"
    input_path.write_text(json.dumps(_small_ir()), encoding="utf-8")

    def fake_chat_json(config, *, system_prompt, user_prompt, node_name):
        if node_name == "generate_route_blocks":
            return {"blocks": [{"kind": "action", "text": "Call register_user"}]}
        if node_name.startswith("generate_service_blocks:"):
            assert "current_route_code:" not in user_prompt
            assert "route_function:" not in user_prompt
            assert "\nroute:" not in user_prompt
            return {"blocks": [{"kind": "action", "text": "user = User from payload"}]}
        if node_name == "compress":
            return {"blocks": [{"kind": "action", "text": "Persist user"}]}
        if node_name == "validate_puml":
            return {"is_valid": True, "feedback": ""}
        raise AssertionError(f"unexpected node {node_name}")

    monkeypatch.setattr("src.generator.load_config", lambda: LLMConfig("test", "test", "https://example.test"))
    monkeypatch.setattr("src.generator.gen_png_graph", lambda app_obj, name_photo="graph.png": None)
    monkeypatch.setattr("src.workflow.chat_json", fake_chat_json)

    generated_files = generate_from_file(input_path, output_dir)

    assert [path.name for path in generated_files] == [
        "POST_auth.activity.puml",
        "POST_auth.src_auth_service_register_user.activity.puml",
    ]
    route_puml = (output_dir / "POST_auth.activity.puml").read_text(encoding="utf-8")
    service_puml = (output_dir / "POST_auth.src_auth_service_register_user.activity.puml").read_text(encoding="utf-8")

    assert "title POST /auth" in route_puml
    assert ":Persist user;" in route_puml
    assert route_puml.endswith("stop\n@enduml\n")
    assert service_puml.startswith("@startuml\n")
    assert "title src.auth.service.register_user" in service_puml
    assert "\nstart\n" in service_puml
    assert ":Persist user;" in service_puml
    assert ":Return;" not in service_puml
    assert service_puml.endswith("stop\n@enduml\n")


def test_control_flow_noise_does_not_break_puml_validation():
    blocks = _filter_generated_blocks(
        [
            {"kind": "action", "text": "Start try block for user registration."},
            {"kind": "action", "text": "Начать блок try для обработки ошибок."},
            {
                "kind": "if",
                "condition": "An exception occurs during the try block",
                "then": [{"kind": "action", "text": "Log error"}],
            },
        ],
        route_function={"parameters": []},
    )

    body = _render_blocks(blocks)
    puml = f"@startuml\ntitle Test\nstart\n{body}\nend\n@enduml\n"

    assert "Start try block" not in body
    assert "Начать блок try" not in body
    assert "during the try block" not in body
    assert "during execution" in body
    assert _deterministic_validate(puml) is None


def test_else_only_if_is_rendered_as_non_empty_then_branch():
    blocks = _filter_generated_blocks(
        [
            {
                "kind": "if",
                "condition": "User registration is successful",
                "then": [],
                "else": [{"kind": "action", "text": "Handle registration failure"}],
            }
        ],
        route_function={"parameters": []},
    )

    body = _render_blocks(blocks)

    assert "if (not User registration is successful) then (+)" in body
    assert ":Handle registration failure;" in body
    assert "else (-)" not in body


def test_raise_and_return_actions_render_as_terminal_blocks():
    blocks = _prepare_generated_blocks(
        [
            {"kind": "action", "text": "Raise AuthenticationError"},
            {"kind": "action", "text": "Return token_response"},
        ],
        route_function={"parameters": []},
        append_stop=False,
    )

    body = _render_blocks(blocks)

    assert ":Raise AuthenticationError;" in body
    assert "end" in body.splitlines()
    assert ":Return token_response;" in body
    assert "stop" in body.splitlines()


def test_unsupported_try_wrapper_is_flattened_without_losing_success_path():
    blocks = _prepare_generated_blocks(
        [
            {
                "kind": "try",
                "then": [
                    {"kind": "action", "text": "Create User entity"},
                    {"kind": "action", "text": "Commit user in DB"},
                    {"kind": "return", "text": "Возвращаем None после успешного завершения."},
                ],
                "else": [
                    {"kind": "action", "text": "Log registration error"},
                    {"kind": "raise", "text": "raise"},
                ],
            }
        ],
        route_function={"parameters": []},
        append_stop=True,
    )

    body = _render_blocks(blocks)

    assert ":Create User entity;" in body
    assert ":Commit user in DB;" in body
    assert ":Возвращаем None после успешного завершения.;" in body
    assert "if (ошибка при выполнении) then (+)" in body
    assert ":Log registration error;" in body
    assert "    end" in body.splitlines()
    assert "stop" in body.splitlines()


def test_linear_except_tail_is_rendered_as_error_branch():
    blocks = _prepare_generated_blocks(
        [
            {"kind": "action", "text": "generated_user_id = uuid.uuid4()"},
            {"kind": "action", "text": "db.commit()"},
            {"kind": "action", "text": "None"},
            {"kind": "action", "text": "stop"},
            {"kind": "action", "text": "except Exception as e:"},
            {"kind": "action", "text": "logging.error(f'Error: {e}')"},
            {"kind": "action", "text": "raise"},
            {"kind": "action", "text": "end"},
        ],
        route_function={"parameters": []},
        append_stop=True,
    )

    body = _render_blocks(blocks)

    assert "except Exception as e" not in body
    assert "if (ошибка при выполнении) then (+)" in body
    assert ":logging.error(f'Error: {e}');" in body
    assert ":raise;" in body
    assert "    end" in body.splitlines()
    assert "else (-)" in body
    assert ":generated_user_id = uuid.uuid4();" in body
    assert ":db.commit();" in body
    assert ":return None;" in body
    assert "    stop" in body.splitlines()


def test_service_return_is_rewritten_when_merged_into_route():
    merged = _merge_blocks(
        [{"kind": "action", "text": "token = auth_service.login(form_data)"}],
        [
            {
                "function_id": "src.auth.service.login",
                "blocks": [
                    {"kind": "action", "text": "token = create_access_token(user.username, user.id, token_expires)"},
                    {
                        "kind": "action",
                        "text": "return model.Token(access_token=token, token_type='bearer')",
                    },
                    {"kind": "action", "text": "stop"},
                ],
                "current_puml": "@startuml\n@enduml\n",
            }
        ],
    )

    body = _render_blocks(merged)

    assert ":token = auth_service.login(form_data);" in body
    assert ":token = create_access_token(user.username, user.id, token_expires);" in body
    assert ":service_result = model.Token(access_token=token, token_type='bearer');" in body
    assert "return model.Token" not in body
    assert "stop" not in body.splitlines()


def test_single_russian_service_call_is_inlined_during_merge():
    merged = _merge_blocks(
        [{"kind": "action", "text": "Вызвать сервис для регистрации пользователя с параметрами db и register_user_request."}],
        [
            {
                "function_id": "src.auth.service.register_user",
                "blocks": [
                    {"kind": "action", "text": "generated_user_id = uuid.uuid4()"},
                    {"kind": "action", "text": "return None"},
                    {"kind": "action", "text": "stop"},
                ],
                "current_puml": "@startuml\n@enduml\n",
            }
        ],
    )

    body = _render_blocks(merged)

    assert ":Вызвать сервис для регистрации пользователя с параметрами db и register_user_request.;" in body
    assert ":generated_user_id = uuid.uuid4();" in body
    assert ":service_result = None;" in body


def test_russian_route_shell_actions_are_filtered_from_route_blocks():
    blocks = _filter_generated_blocks(
        [
            {"kind": "action", "text": "Получить запрос на регистрацию пользователя."},
            {"kind": "action", "text": "Вызвать сервис для регистрации пользователя с параметрами db и register_user_request."},
            {
                "kind": "if",
                "condition": "если регистрация прошла успешно",
                "then": [
                    {"kind": "action", "text": "Вернуть ответ с кодом 201."},
                    {"kind": "action", "text": "stop"},
                ],
                "else": [
                    {"kind": "action", "text": "Обработать ошибку регистрации."},
                    {"kind": "action", "text": "end"},
                ],
            },
        ],
        route_function={"parameters": []},
    )

    body = _render_blocks(blocks)

    assert "Получить запрос" not in body
    assert "Вернуть ответ с кодом 201" not in body
    assert "Вызвать сервис для регистрации пользователя" in body
    assert "Обработать ошибку регистрации" in body


def test_route_and_service_prompt_contexts_are_compact():
    route = {"route_id": "POST /auth"}
    route_function = {
        "function_id": "src.auth.controller.register_user",
        "signature": "async def register_user(db, payload)",
        "http": {"method": "POST", "path": "/auth", "status_code": 201},
        "parameters": [
            {"name": "db", "location": "dependency", "type": "DbSession", "provider": "src.database.core.get_db"},
            {"name": "payload", "location": "body", "type": "RegisterUserRequest", "provider": None},
        ],
        "steps": [
            {"kind": "receive_request", "source": "register_user(db, payload)", "module": "src.auth.controller"},
            {"kind": "call", "source": "service.register_user(db, payload)", "target": "src.auth.service.register_user"},
        ],
        "file": "controller.py",
        "edges": [{"from": "a", "to": "b"}],
    }
    service_function = {
        "function_id": "src.auth.service.register_user",
        "signature": "def register_user(db, payload)",
        "steps": [
            {"kind": "call", "source": "db.add(user_model)", "module": "src.auth.service"},
            {"kind": "return", "source": "return None", "value": "None"},
        ],
        "package": "src.auth",
        "file": "service.py",
        "edges": [{"from": "a", "to": "b"}],
    }

    route_context = _route_prompt_context(route, route_function)
    service_context = _service_prompt_context(service_function)

    assert route_context == {
        "route_id": "POST /auth",
        "http": {"method": "POST", "path": "/auth", "status_code": 201},
        "signature": "async def register_user(db, payload)",
        "parameters": [
            {"name": "db", "location": "dependency", "type": "DbSession"},
            {"name": "payload", "location": "body", "type": "RegisterUserRequest"},
        ],
        "steps": [
            {"kind": "receive_request", "source": "register_user(db, payload)"},
            {"kind": "call", "source": "service.register_user(db, payload)"},
        ],
    }
    assert service_context == {
        "function_id": "src.auth.service.register_user",
        "signature": "def register_user(db, payload)",
        "steps": [
            {"kind": "call", "source": "db.add(user_model)"},
            {"kind": "return", "source": "return None"},
        ],
    }


def test_validator_prompt_context_depends_on_diagram_kind():
    state = {
        "route": {"route_id": "POST /auth"},
        "route_function": {
            "signature": "async def register_user(db, payload)",
            "http": {"method": "POST", "path": "/auth", "status_code": 201},
            "parameters": [],
            "steps": [{"kind": "call", "source": "service.register_user(db, payload)"}],
        },
        "service_functions": [
            {
                "function_id": "src.auth.service.register_user",
                "signature": "def register_user(db, payload)",
                "steps": [{"kind": "return", "source": "return None"}],
            }
        ],
        "current_service_function": {
            "function_id": "src.auth.service.register_user",
            "signature": "def register_user(db, payload)",
            "steps": [{"kind": "return", "source": "return None"}],
        },
    }

    route_context = _validator_prompt_context(state, diagram_kind="route")
    service_context = _validator_prompt_context(state, diagram_kind="service")
    merge_context = _validator_prompt_context(state, diagram_kind="merge")

    assert set(route_context) == {"route"}
    assert set(service_context) == {"service_function"}
    assert set(merge_context) == {"route", "service_functions"}


def test_generate_service_blocks_prompt_uses_feedback_as_instruction(monkeypatch):
    captured_prompt = {}

    def fake_chat_json(*args, **kwargs):
        captured_prompt["text"] = kwargs["user_prompt"]
        return {"blocks": [{"kind": "action", "text": "db.add(user_model)"}]}

    monkeypatch.setattr("src.workflow.chat_json", fake_chat_json)

    state = {
        "route_function": {"parameters": []},
        "current_service_function": {
            "function_id": "src.auth.service.register_user",
            "signature": "def register_user(db, payload)",
            "steps": [
                {"kind": "call", "source": "db.add(user_model)", "module": "src.auth.service"},
                {"kind": "return", "source": "return None"},
            ],
            "module": "src.auth.service",
            "file": "service.py",
        },
        "validation_feedback": "Missing success stop",
        "llm_config": LLMConfig(api_key="test", model="test", base_url="https://example.test"),
    }

    generate_service_blocks_node(state)

    assert "исправляешь предыдущий невалидный JSON service-блоков" in captured_prompt["text"]
    assert "Missing success stop" in captured_prompt["text"]
    assert '"function_id": "src.auth.service.register_user"' in captured_prompt["text"]
    assert '"signature": "def register_user(db, payload)"' in captured_prompt["text"]
    assert '"module"' not in captured_prompt["text"]
    assert '"file"' not in captured_prompt["text"]


def test_compress_blocks_prompt_uses_feedback_only_when_present(monkeypatch):
    prompts = []

    def fake_chat_json(*args, **kwargs):
        prompts.append(kwargs["user_prompt"])
        return {"blocks": [{"kind": "action", "text": "db.add(user_model)"}, {"kind": "action", "text": "stop"}]}

    monkeypatch.setattr("src.workflow.chat_json", fake_chat_json)

    base_state = {
        "route_function": {"parameters": []},
        "current_blocks": [
            {"kind": "action", "text": "db.add(user_model)"},
            {"kind": "action", "text": "return None"},
            {"kind": "action", "text": "stop"},
        ],
        "llm_config": LLMConfig(api_key="test", model="test", base_url="https://example.test"),
    }

    compress_blocks_node(base_state)
    compress_blocks_node({**base_state, "validation_feedback": "Добавьте явное завершение успешного потока."})

    assert "исправляешь предыдущую неудачную компрессию" not in prompts[0]
    assert "<feedback>" not in prompts[0]
    assert "исправляешь предыдущую неудачную компрессию" in prompts[1]
    assert "Добавьте явное завершение успешного потока." in prompts[1]
    assert "не удаляй, не схлопывай и не маскируй" in prompts[1]


def test_route_after_validation_allows_three_retries_and_raises_on_fourth():
    assert route_after_validation(
        {
            "current_diagram_kind": "service",
            "validator_passed": False,
            "retry_count": 3,
            "max_retries": 3,
        }
    ) == "retry_service"

    try:
        route_after_validation(
            {
                "current_diagram_kind": "service",
                "validator_passed": False,
                "retry_count": 4,
                "max_retries": 3,
                "validation_feedback": "still broken",
            }
        )
    except RuntimeError as error:
        assert "still broken" in str(error)
    else:
        raise AssertionError("expected RuntimeError on fourth failed validation")


def _small_ir() -> dict[str, object]:
    return {
        "functions": [
            {
                "function_id": "src.auth.controller.register_user",
                "parameters": [
                    {
                        "name": "payload",
                        "location": "body",
                        "type": "RegisterRequest",
                    }
                ],
                "http": {"status_code": 201},
            },
            {
                "function_id": "src.auth.service.register_user",
                "parameters": [
                    {
                        "name": "payload",
                        "type": "RegisterRequest",
                    }
                ],
            },
        ],
        "routes": [
            {
                "route_id": "POST /auth",
                "handler_function_id": "src.auth.controller.register_user",
                "response_model": "src.auth.schemas.UserResponse",
                "service_function_groups": [["src.auth.service.register_user"]],
            }
        ],
    }
