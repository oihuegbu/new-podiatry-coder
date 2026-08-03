from types import SimpleNamespace
from unittest import mock

from app.core.llm_client import _openai_chat_completion


def test_openai_uses_current_completion_budget_parameter():
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content='{"ok":true}'))],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )
    client = mock.Mock()
    client.chat.completions.create.return_value = response

    with mock.patch("app.core.llm_client.get_openai_client",
                    return_value=client):
        content, usage = _openai_chat_completion(
            system_prompt="system", user_prompt="user", model="model",
            temperature=0.0, max_tokens=512, json_mode=True)

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert content == '{"ok":true}'
    assert usage["total_tokens"] == 12
