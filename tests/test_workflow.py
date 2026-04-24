from __future__ import annotations

import json
from pathlib import Path

from src.generator import generate_from_file
from src.llm import LLMConfig
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
        if node_name == "compress_blocks":
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
    assert service_puml.startswith("@startuml\n")
    assert "title src.auth.service.register_user" in service_puml
    assert "\nstart\n" in service_puml
    assert ":user = User from payload;" in service_puml
    assert ":Return;" in service_puml
    assert service_puml.endswith("end\n@enduml\n")


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
