from pathlib import Path

import yaml


REQUIRED_PROMPT_LINES = [
    "Return only final artifact.",
    "No reasoning/tool-call/chatter/markdown fences.",
    "For patch output: must be a unified diff starting with `diff --git`.",
    "If unable to produce valid artifact, return empty output.",
]


def test_all_agent_prompts_include_artifact_contract():
    agents_dir = Path("agents")
    yaml_files = sorted(agents_dir.glob("*.yaml"))
    assert yaml_files, "No agent yaml files found under agents/."

    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt_template = data.get("prompt_template", "")
        missing = [line for line in REQUIRED_PROMPT_LINES if line not in prompt_template]
        assert not missing, f"{path} missing required prompt lines: {missing}"
