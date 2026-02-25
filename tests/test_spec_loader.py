from pathlib import Path

import pytest

from agents.spec_loader import AgentSpecLoader


def test_agent_spec_loads_inline_prompt_template(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi {skills}"
tools: []
skills: [s1]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, prompt, allowed = loader.load(agent_yaml)
    assert spec.name == "test"
    assert "Allowed Tools" in prompt
    assert allowed == {"submit"}


def test_agent_spec_loads_prompt_file(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n")

    prompt_file = tmp_path / "prompts" / "agent_prompt.txt"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text("Prompt from file {skills}\n", encoding="utf-8")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_file: prompts/agent_prompt.txt
tools: []
skills: [s1]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, prompt, allowed = loader.load(agent_yaml)
    assert spec.name == "test"
    assert "Prompt from file" in prompt
    assert "Allowed Tools" in prompt
    assert allowed == {"submit"}


def test_agent_spec_tools_allowlist_intersects_with_skill_tools(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n- workspace_open\n")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi {skills}"
tools: [bash, submit]
skills: [s1]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, _prompt, allowed = loader.load(agent_yaml)

    assert spec.tools == ["bash", "submit"]
    assert allowed == {"submit"}


def test_agent_spec_empty_tools_list_is_explicit_empty_allowlist(tmp_path: Path):
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi {skills}"
tools: []
skills: []
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, _prompt, allowed = loader.load(agent_yaml)

    assert spec.tools == []
    assert allowed == set()


def test_agent_spec_missing_tools_and_skills_keeps_unrestricted_allowlist(tmp_path: Path):
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, _prompt, allowed = loader.load(agent_yaml)

    assert spec.tools is None
    assert allowed is None


def test_agent_spec_skills_require_prompt_placeholder(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
tools: []
skills: [s1]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    with pytest.raises(ValueError, match="missing required `\\{skills\\}` placeholder"):
        loader.load(agent_yaml)


def test_tools_profile_rendered_prompt_includes_tool_protocol_rules():
    repo_root = Path(__file__).resolve().parents[1]
    loader = AgentSpecLoader(repo_root)

    _spec, prompt, _allowed = loader.load(repo_root / "profiles" / "agents" / "gemini_2.5_flash_tools.yaml")

    assert "submit(final_artifact)" in prompt
    assert "**/*.py" in prompt
    assert "workspace_search only accepts query and optional glob" in prompt
    assert "line-bounded workspace_open" in prompt
