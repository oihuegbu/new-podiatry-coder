"""OpenAI chat-completion parameter self-adaptation (llm_client).

Newer OpenAI models renamed `max_tokens` -> `max_completion_tokens` and reject a
non-default `temperature`. The client learns each model's quirks from the API's own
400 and caches them; classic models (gpt-4o) are unaffected.
"""
import types
import pytest
from app.core import llm_client


def _resp(content="OK"):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            finish_reason="stop",
            message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))


class _Err(Exception):
    def __init__(self, param, code="unsupported_parameter"):
        super().__init__(f"Unsupported parameter: '{param}' is not supported with this model.")
        self.body = {"error": {"param": param, "code": code}}


class _FakeClient:
    def __init__(self, reject):
        self.reject = set(reject)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if "max_tokens" in kwargs and "max_tokens" in self.reject:
            raise _Err("max_tokens")
        if "temperature" in kwargs and "temperature" in self.reject:
            raise _Err("temperature")
        return _resp("OK")


def test_openai_adapts_max_tokens_and_temperature(monkeypatch):
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()
    fake = _FakeClient(reject={"max_tokens", "temperature"})
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: fake)
    out, _ = llm_client._openai_chat_completion(
        "sys", "user", model="gpt-newmodel-x", temperature=0.0, max_tokens=50, json_mode=False)
    assert out == "OK"
    final = fake.calls[-1]
    assert final["max_completion_tokens"] == 50 and "max_tokens" not in final
    assert "temperature" not in final
    # quirks are cached per model: a second call sends the adapted shape in ONE create()
    fake.calls.clear()
    llm_client._openai_chat_completion(
        "sys", "user", model="gpt-newmodel-x", temperature=0.0, max_tokens=50, json_mode=False)
    assert len(fake.calls) == 1 and "max_completion_tokens" in fake.calls[0]


def test_openai_classic_model_request_unchanged(monkeypatch):
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()
    fake = _FakeClient(reject=set())            # accepts classic params (gpt-4o-like)
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: fake)
    out, _ = llm_client._openai_chat_completion(
        "sys", "user", model="gpt-4o", temperature=0.0, max_tokens=50, json_mode=False)
    assert out == "OK" and len(fake.calls) == 1
    assert fake.calls[0]["max_tokens"] == 50 and fake.calls[0]["temperature"] == 0.0
    assert "max_completion_tokens" not in fake.calls[0]


def test_openai_unadaptable_400_reraises(monkeypatch):
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()

    class _C:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._c))

        def _c(self, **k):
            e = Exception("bad messages"); e.body = {"error": {"param": "messages", "code": None}}
            raise e

    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _C())
    with pytest.raises(Exception):
        llm_client._openai_chat_completion(
            "s", "u", model="m", temperature=0.0, max_tokens=10, json_mode=False)
