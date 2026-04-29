from __future__ import annotations

from typing import Any


DEFAULT_PROMPTS: dict[str, str] = {
    "route-system": "Ты генерируешь только JSON-блоки для route-функции.",
    "route-user": """
Ты превращаешь компактный IR одной route-функции в строгий JSON-блок.

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
8. Если есть feedback, исправь именно описанную проблему и верни полный JSON заново, а не частичный diff.

# Инструкция
{instruction}

# Контекст
<route_context>
{route_context_json}
</route_context>

{feedback_section}
""".strip(),
    "service-system": "Ты генерируешь только JSON-блоки для функции.",
    "service-user": """
Ты превращаешь компактный IR одной service-функции в строгую последовательность блоков в JSON.

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
9. Если есть feedback, исправь именно описанную проблему и верни полный JSON заново, а не частичный diff.

# Инструкция
{instruction}

# Контекст
<service_function_context>
{function_context_json}
</service_function_context>

{feedback_section}
""".strip(),
    "compress-system": "Ты сжимаешь JSON-блоки и отвечаешь только JSON.",
    "compress-user": """
Ты сжимаешь JSON-блок activity-диаграммы.
Ответ только JSON в том же формате:
{{"blocks":[...]}}

# Инструкция
{instruction}

Правила компрессии:
{compression_rules}

Жесткие правила:
1. Только JSON.
2. Не меняй смысл веток if.
3. Не удаляй важные error branches, DB read/write, entity/model construction, внешние вызовы.
4. Удаляй или схлопывай мелкие технические шаги без бизнес-смысла.
5. Не добавляй новые блоки, которых не было в исходном JSON.
6. Не удаляй и не перемещай action с text "end" или "stop".
7. Если в feedback указано, что потерян service return, route response, success completion, error branch, service-call или terminal block, сохрани эти элементы в выходном JSON и не схлопывай их.
8. Если feedback требует явно показать завершение успешного потока, не удаляй return/stop и не заменяй их на более краткую формулировку.

{feedback_section}

current_block_json:
{current_block_json}
""".strip(),
}


def build_prompt_overrides(messages: list[dict[str, Any]]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str) and role in DEFAULT_PROMPTS:
            overrides[role] = content
    return overrides


def render_prompt(template: str, **context: object) -> str:
    normalized_context = {
        key: "" if value is None else value
        for key, value in context.items()
    }
    return template.format_map(_SafeFormatDict(normalized_context))


class _SafeFormatDict(dict[str, object]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
