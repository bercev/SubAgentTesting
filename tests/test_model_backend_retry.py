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


def test_generate_emits_api_usage_event_when_usage_present(monkeypatch):
    emitted: list[str] = []

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
                    "id": "gen-123",
                    "provider": "StepFun",
                    "model": "stepfun/step-3.5-flash:free",
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "cost": 0.0042,
                        "is_byok": False,
                    },
                },
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)

    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=0,
        event_logger=emitted.append,
    )
    result = backend.generate(
        messages=[
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "task"},
        ]
    )

    assert result.assistant_text == "ok"
    usage_events = [line for line in emitted if line.startswith("api_usage ")]
    assert len(usage_events) == 1
    usage_line = usage_events[0]
    assert "response_id=gen-123" in usage_line
    assert "prompt_tokens=11" in usage_line
    assert "completion_tokens=7" in usage_line
    assert "total_tokens=18" in usage_line
    assert "cost_usd=0.0042" in usage_line


def test_backend_preview_logs_truncate_by_default(monkeypatch):
    emitted: list[str] = []
    request_tail = "REQTAIL"
    response_tail = "RESPTAIL"
    long_request = "r" * 2200 + request_tail
    long_response = "s" * 1300 + response_tail

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
                long_response,
                payload={"choices": [{"message": {"content": "ok"}}]},
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=0,
        event_logger=emitted.append,
    )
    backend.generate(messages=[{"role": "user", "content": long_request}])

    request_line = next(line for line in emitted if line.startswith("api_request "))
    response_line = next(line for line in emitted if line.startswith("api_response "))
    assert "payload_preview=" in request_line
    assert "body_preview=" in response_line
    assert "...[truncated]" in request_line
    assert "...[truncated]" in response_line
    assert request_tail not in request_line
    assert response_tail not in response_line


def test_backend_preview_logs_full_when_enabled(monkeypatch):
    emitted: list[str] = []
    request_tail = "REQTAIL"
    response_tail = "RESPTAIL"
    long_request = "r" * 2200 + request_tail
    long_response = "s" * 1300 + response_tail

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
                long_response,
                payload={"choices": [{"message": {"content": "ok"}}]},
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(
        api_key="test-key",
        model="openrouter/free",
        max_retries=0,
        event_logger=emitted.append,
        full_log_previews=True,
    )
    backend.generate(messages=[{"role": "user", "content": long_request}])

    request_line = next(line for line in emitted if line.startswith("api_request "))
    response_line = next(line for line in emitted if line.startswith("api_response "))
    assert "...[truncated]" not in request_line
    assert "...[truncated]" not in response_line
    assert request_tail in request_line
    assert response_tail in response_line


def test_generate_result_includes_finish_reason_and_usage_tokens(monkeypatch):
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
                            "finish_reason": "length",
                            "message": {"content": "partial"},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 4096,
                        "total_tokens": 4196,
                    },
                },
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(api_key="test-key", model="openrouter/free", max_retries=0)
    result = backend.generate(messages=[{"role": "user", "content": "task"}])

    assert result.assistant_text == "partial"
    assert result.finish_reason == "length"
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 4096
    assert result.total_tokens == 4196


def test_generate_result_defaults_usage_metadata_to_none_when_missing(monkeypatch):
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
                payload={"choices": [{"message": {"content": "ok"}}]},
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(api_key="test-key", model="openrouter/free", max_retries=0)
    result = backend.generate(messages=[{"role": "user", "content": "task"}])

    assert result.assistant_text == "ok"
    assert result.finish_reason is None
    assert result.prompt_tokens is None
    assert result.completion_tokens is None
    assert result.total_tokens is None
