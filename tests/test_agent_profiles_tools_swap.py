from pathlib import Path

import yaml


def _load_tools(profile_path: Path) -> list[str]:
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    tools = data.get("tools")
    assert isinstance(tools, list), f"`tools` must be a list in {profile_path}"
    return [str(item) for item in tools]


def test_strict_profile_excludes_bash():
    profile = Path("profiles/agents/gemini_2.5_flash_tools_mini-tools.yaml")
    tools = _load_tools(profile)
    required = {
        "submit",
        "workspace_list",
        "workspace_open",
        "workspace_search",
        "workspace_apply_patch",
        "workspace_write",
    }

    assert required.issubset(set(tools))
    assert "bash" not in tools


def test_fallback_profile_includes_bash():
    profile = Path("profiles/agents/gemini_2.5_flash_tools_mini-tools-bash.yaml")
    tools = _load_tools(profile)
    required = {
        "submit",
        "workspace_list",
        "workspace_open",
        "workspace_search",
        "workspace_apply_patch",
        "workspace_write",
        "bash",
    }

    assert required.issubset(set(tools))
