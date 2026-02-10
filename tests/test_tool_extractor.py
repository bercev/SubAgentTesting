from runtime.model_backend import OpenRouterBackend


def test_extract_tool_calls_json_block():
    text = """
Please fix.
```json
{"name": "workspace_list", "arguments": {"path": "."}}
```
"""
    calls = OpenRouterBackend._extract_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "workspace_list"
    assert calls[0].arguments == {"path": "."}


def test_extract_tool_calls_xml_block():
    text = "<tool_call name=\"submit\">{\"final_artifact\": ""patch""}</tool_call>"
    calls = OpenRouterBackend._extract_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0].name == "submit"
    assert calls[0].arguments["final_artifact"] == "patch"
