import httpx
import pytest

from runtime.model_backend import OpenRouterBackend


class _FakeResponse:
    def __init__(self, status_code: int, text: str, payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
        self.request = httpx.Request("POST", "https://openrouter.test/chat/completions")

    def json(self):
        return self._payload


def test_retryable_400_then_success(monkeypatch):
    calls = {"count": 0}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return _FakeResponse(
                    400,
                    '{"error":{"message":"Developer instruction is not enabled for models/gemma-3-4b-it"}}',
                )
            return _FakeResponse(200, "", payload={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)

    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=2,
        initial_backoff_s=0,
        max_backoff_s=0,
    )
    result = backend.generate(
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "task"},
        ]
    )
    assert result.assistant_text == "ok"
    assert calls["count"] == 2


def test_non_retryable_400_fails_fast(monkeypatch):
    calls = {"count": 0}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            calls["count"] += 1
            return _FakeResponse(400, '{"error":{"message":"Invalid request body"}}')

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)

    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=5,
        initial_backoff_s=0,
        max_backoff_s=0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        backend.generate(
            messages=[
                {"role": "system", "content": "prompt"},
                {"role": "user", "content": "task"},
            ]
        )

    assert calls["count"] == 1


def test_backend_requires_model_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    with pytest.raises(ValueError, match="backend.model or OPENROUTER_MODEL is required"):
        OpenRouterBackend(api_key="test-key", model=None)


def test_generate_does_not_extract_tool_calls_from_assistant_text(monkeypatch):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _FakeResponse(
                200,
                "",
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": '<tool_call name="submit">{"final_artifact":"patch"}</tool_call>',
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=0,
    )
    result = backend.generate(
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "task"},
        ],
        tools=[{"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}],
    )
    assert result.assistant_text.startswith("<tool_call")
    assert result.tool_calls == []
