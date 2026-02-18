from pathlib import Path

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
