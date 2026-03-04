from pathlib import Path

import pytest

from agent_architectures.constants import ARCHITECTURE_MINI_SWE_AGENT, ARCHITECTURE_NONE
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
    assert spec.agent_architecture == ARCHITECTURE_NONE
    assert spec.agent_architecture_config == {}


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


def test_agent_spec_skills_without_placeholder_allowed_in_patch_only(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
tools: [submit]
skills: [s1]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, prompt, allowed = loader.load(agent_yaml, runtime_mode="patch_only")

    assert spec.skills == ["s1"]
    assert prompt == "Hi"
    assert allowed == {"submit"}


def test_agent_spec_ignores_empty_skill_entries(tmp_path: Path):
    skills_dir = tmp_path / "skills" / "s1"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("Allowed Tools:\n- submit\n")

    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi {skills}"
tools: [submit]
skills: [s1, null, "  "]
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, prompt, allowed = loader.load(agent_yaml)

    assert spec.skills == ["s1"]
    assert "Allowed Tools" in prompt
    assert allowed == {"submit"}


def test_tools_profile_rendered_prompt_includes_tool_protocol_rules():
    repo_root = Path(__file__).resolve().parents[1]
    loader = AgentSpecLoader(repo_root)

    _spec, prompt, _allowed = loader.load(repo_root / "profiles" / "agents" / "gemini_2.5_flash_tools.yaml")

    assert "submit(final_artifact)" in prompt
    assert "**/*.py" in prompt
    assert "workspace_search only accepts query and optional glob" in prompt
    assert "line-bounded workspace_open" in prompt


def test_agent_spec_accepts_mini_architecture_config(tmp_path: Path):
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
agent_architecture: mini-swe-agent
agent_architecture_config:
  planner: constrained
  retries: 2
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    spec, _prompt, _allowed = loader.load(agent_yaml)

    assert spec.agent_architecture == ARCHITECTURE_MINI_SWE_AGENT
    assert spec.agent_architecture_config == {"planner": "constrained", "retries": 2}


def test_agent_spec_rejects_invalid_architecture_id(tmp_path: Path):
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
agent_architecture: unknown-arch
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    with pytest.raises(ValueError, match="Unsupported agent architecture"):
        loader.load(agent_yaml)


def test_agent_spec_rejects_non_object_architecture_config(tmp_path: Path):
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """
name: test
backend: {type: openrouter, model: x}
prompt_template: "Hi"
agent_architecture_config: "bad"
tool_to_skill_map: {}
termination: {tool: submit}
decoding_defaults: {}
"""
    )

    loader = AgentSpecLoader(tmp_path)
    with pytest.raises(ValueError, match="agent_architecture_config"):
        loader.load(agent_yaml)
