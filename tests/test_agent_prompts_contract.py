from pathlib import Path

import yaml


LEGACY_EMPTY_FAILURE_SNIPPETS = (
    "return empty output",
    'submit("")',
)
NON_EMPTY_FAILURE_SNIPPETS = (
    "cannot produce output {reason}",
    "do not submit an empty artifact",
    "do not submit an empty patch",
)


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


def test_all_agent_prompts_enforce_non_empty_failure_contract():
    agents_dir = Path("profiles/agents")
    yaml_files = sorted(agents_dir.glob("*.yaml"))
    assert yaml_files, "No agent yaml files found under profiles/agents/."

    for path in yaml_files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        prompt_text = _read_prompt_text(path, data)
        lowered = prompt_text.lower()

        assert "diff --git" in lowered, f"{path} missing diff output contract"
        assert any(snippet in lowered for snippet in NON_EMPTY_FAILURE_SNIPPETS), (
            f"{path} missing non-empty failure contract"
        )
        assert all(snippet not in lowered for snippet in LEGACY_EMPTY_FAILURE_SNIPPETS), (
            f"{path} contains legacy empty-output guidance"
        )
