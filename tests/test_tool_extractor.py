import httpx

from runtime.model_backend import OpenRouterBackend


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self.text = ""
        self._payload = payload
        self.request = httpx.Request("POST", "https://openrouter.test/chat/completions")

    def json(self):
        return self._payload


def test_generate_parses_structured_tool_calls(monkeypatch):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "workspace_list",
                                            "arguments": '{"path":"."}',
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(api_key="test-key", model="openrouter/free")
    result = backend.generate(messages=[{"role": "user", "content": "task"}])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "workspace_list"
    assert result.tool_calls[0].arguments == {"path": "."}


def test_generate_preserves_unparseable_tool_arguments_as_raw(monkeypatch):
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, *args, **kwargs):
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "submit",
                                            "arguments": '{"final_artifact": ',
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

    monkeypatch.setattr("runtime.model_backend.httpx.Client", _FakeClient)
    backend = OpenRouterBackend(api_key="test-key", model="openrouter/free")
    result = backend.generate(messages=[{"role": "user", "content": "task"}])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "submit"
    assert result.tool_calls[0].arguments["raw"] == '{"final_artifact": '
