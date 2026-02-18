from pathlib import Path

import yaml


REQUIRED_PROMPT_LINES = [
    "Return only final artifact.",
    "If unable to produce valid artifact, return empty output.",
]
NO_REASONING_REQUIREMENTS = [
    "No reasoning/tool-call/chatter/markdown fences.",
    "No reasoning/chatter/markdown fences.",
]
PATCH_OUTPUT_REQUIREMENTS = [
    "For patch output: must be a unified diff starting with `diff --git`.",
    "For patch output, return one unified diff starting with `diff --git`.",
]


def _read_prompt_text(agent_path: Path, data: dict) -> str:
    prompt_template = data.get("prompt_template")
    prompt_file = data.get("prompt_file")
    if isinstance(prompt_template, str) and prompt_template.strip():
        return prompt_template
    if isinstance(prompt_file, str) and prompt_file.strip():
        raw = Path(prompt_file)
        candidates = [raw] if raw.is_absolute() else [agent_path.parent / raw, Path(".") / raw]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
    return ""


def test_all_agent_prompts_include_artifact_contract():
    agents_dir = Path("profiles/agents")
    yaml_files = sorted(agents_dir.glob("*.yaml"))
    assert yaml_files, "No agent yaml files found under profiles/agents/."

    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt_text = _read_prompt_text(path, data)
        missing = [line for line in REQUIRED_PROMPT_LINES if line not in prompt_text]
        assert not missing, f"{path} missing required prompt lines: {missing}"
        assert any(line in prompt_text for line in NO_REASONING_REQUIREMENTS), (
            f"{path} missing no-reasoning guard line"
        )
        assert any(line in prompt_text for line in PATCH_OUTPUT_REQUIREMENTS), (
            f"{path} missing patch-output contract line"
        )
