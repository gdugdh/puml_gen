# AGENTS.md

## Scope

These instructions apply to the whole repository.

## Project Map

- `src/main.py`: CLI entrypoint. Reads `--input`, writes `.puml` files to `--outdir`.
- `src/generator.py`: loads IR, resolves route/service functions, runs the LangGraph workflow, writes artifacts.
- `src/workflow.py`: core pipeline for block generation, compression, merge, rendering, validation, retry handling.
- `src/llm.py`: OpenRouter JSON calls and config loading from `.env`.
- `src/logging_utils.py`: append-only structured logging to `logs/puml_gen.log`.
- `tests/test_workflow.py`: behavioral tests for normalization, rendering, validation, and file generation.

## Workflow Invariants

- Route and service generation both go through JSON `blocks`, not direct PUML generation.
- Only these block kinds are expected in normalized state: `action`, `if`, `partition`.
- Terminal markers are represented only as `{"kind":"action","text":"end"}` and `{"kind":"action","text":"stop"}`.
- Service diagrams may end with `stop`; merged route diagrams must keep route footer ownership of the final `stop`.
- When service blocks are inlined into a route, internal service `return ...` should be rewritten as a non-terminal action so the route flow can continue to `Build response` and `HTTP ...`.
- `_deterministic_validate()` is the hard guardrail for PUML syntax; LLM validation is a second pass for semantic loss or garbage.

## Working Rules

- Prefer fixing issues in normalization/merge helpers before weakening validator logic.
- Keep prompts and block-shaping logic aligned: if a new terminal behavior is introduced, update generation, compression, merge, and tests together.
- Preserve Russian output expectations in prompts and user-facing diagram text unless the existing code already emits code-like English fragments.
- Do not remove logging around workflow nodes and LLM requests/responses; they are the main debugging trail.

## Commands

- Run tests: `pytest`
- Run generator: `python -m src.main --input input/synthetic_data.json --outdir output`

## Notes For Future Changes

- Merge behavior is sensitive: route shell (`Request`, response build, HTTP status, final `stop`) and inlined service logic must remain visually consistent.
- If validation fails after `merge_puml`, inspect both `_merge_blocks()` and `_build_route_puml()` before touching prompts.
