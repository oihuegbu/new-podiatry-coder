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
    # issue #6 F9-R11-C follow-up: the reasoning-token headroom floor (below)
    # pads the requested 50 up to the 16384 floor -- never the raw caller value.
    assert final["max_completion_tokens"] == 16384 and "max_tokens" not in final
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
    # padded to the same 16384 floor -- a non-reasoning model just never uses
    # the extra headroom, so this is harmless for gpt-4o-style models too.
    assert fake.calls[0]["max_tokens"] == 16384 and fake.calls[0]["temperature"] == 0.0
    assert "max_completion_tokens" not in fake.calls[0]


# ---- issue #6 F9-R11-C follow-up: reasoning-token headroom for the OpenAI path ----
# A GPT-5/o-series reasoning model spends hidden "thinking" tokens out of the SAME
# completion-token budget as the final answer -- the same problem
# `_claude_chat_completion` already pads its own budget for. Without an equivalent
# floor on the OpenAI path, a real structured-extraction call on a non-trivial note
# routinely burned its entire small default budget on reasoning before writing any
# JSON, truncating on the very first attempt (observed live: note processing after
# credits were restored) and recovering only through several slow doubling retries.
def test_openai_pads_small_max_tokens_to_the_reasoning_headroom_floor(monkeypatch):
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()
    fake = _FakeClient(reject=set())
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: fake)
    llm_client._openai_chat_completion(
        "sys", "user", model="gpt-5.6-sol", temperature=0.0, max_tokens=4096,
        json_mode=False)
    assert fake.calls[0]["max_tokens"] == 16384


def test_openai_scales_headroom_for_a_large_requested_budget(monkeypatch):
    """A caller that already asked for more than the floor gets 3x that request,
    not merely the floor -- the padding scales with the actual task, it does not
    cap it."""
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()
    fake = _FakeClient(reject=set())
    monkeypatch.setattr(llm_client, "get_openai_client", lambda: fake)
    llm_client._openai_chat_completion(
        "sys", "user", model="gpt-5.6-sol", temperature=0.0, max_tokens=10000,
        json_mode=False)
    assert fake.calls[0]["max_tokens"] == 30000


def test_openai_truncation_reports_the_effective_padded_budget(monkeypatch):
    """The truncation exception must name the budget actually sent (the padded
    one), not the caller's raw request -- otherwise the retry-doubling log
    would understate how much room the model actually had and already burned."""
    llm_client._OPENAI_UNSUPPORTED_PARAMS.clear()

    class _TruncatingClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(finish_reason="length",
                                               message=types.SimpleNamespace(content=""))],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                            total_tokens=2))

    monkeypatch.setattr(llm_client, "get_openai_client", lambda: _TruncatingClient())
    with pytest.raises(llm_client._TruncatedResponse, match="16384"):
        llm_client._openai_chat_completion(
            "sys", "user", model="gpt-5.6-sol", temperature=0.0, max_tokens=100,
            json_mode=False)


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
