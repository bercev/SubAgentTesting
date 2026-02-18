from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from skills.loader import load_skills
from runtime.schemas import BenchmarkTask


@dataclass
class AgentSpec:
    """Parsed agent profile schema used by runtime services."""

    name: str
    backend: Dict[str, Any]
    prompt_template: str
    tools: List[Dict[str, Any]]
    skills: List[str]
    tool_to_skill_map: Dict[str, List[str]]
    termination: Dict[str, Any]
    decoding_defaults: Dict[str, Any]


class AgentSpecLoader:
    """Loader for agent YAML profiles and resolved skill prompt text."""

    def __init__(self, base_dir: Path) -> None:
        """Keep repository base path for skill directory resolution."""

        self.base_dir = base_dir

    def load(self, path: Path) -> Tuple[AgentSpec, str, Set[str]]:
        """Load agent spec, render final system prompt, and derive allowed tools."""

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        prompt_template = self._resolve_prompt_template(data, path)
        spec = AgentSpec(
            name=data["name"],
            backend=data["backend"],
            prompt_template=prompt_template,
            tools=data.get("tools", []),
            skills=data.get("skills", []),
            tool_to_skill_map=data.get("tool_to_skill_map", {}),
            termination=data.get("termination", {}),
            decoding_defaults=data.get("decoding_defaults", {}),
        )
        skill_dirs = [self.base_dir / "skills" / s for s in spec.skills]
        skills_text, allowed_tools = load_skills(skill_dirs)
        prompt = self.render_prompt(spec, skills_text)
        return spec, prompt, allowed_tools

    def render_prompt(self, spec: AgentSpec, skills_text: str) -> str:
        """Render profile prompt template by injecting loaded skills text."""

        base = spec.prompt_template
        # Simple placeholder replacement
        base = base.replace("{skills}", skills_text)
        return base

    def _resolve_prompt_template(self, data: Dict[str, Any], agent_path: Path) -> str:
        """Resolve prompt text from inline template or external prompt file."""

        prompt_template = data.get("prompt_template")
        prompt_file = data.get("prompt_file")

        if prompt_template is not None and prompt_file is not None:
            raise ValueError("Agent spec must define only one of `prompt_template` or `prompt_file`")

        if prompt_file is not None:
            if not isinstance(prompt_file, str) or not prompt_file.strip():
                raise ValueError("`prompt_file` must be a non-empty string path")
            prompt_path = self._resolve_prompt_path(prompt_file.strip(), agent_path)
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            return prompt_path.read_text(encoding="utf-8")

        if prompt_template is None:
            raise ValueError("Missing required prompt definition: set `prompt_template` or `prompt_file`")
        if not isinstance(prompt_template, str):
            raise ValueError("`prompt_template` must be a string")
        return prompt_template

    def _resolve_prompt_path(self, prompt_file: str, agent_path: Path) -> Path:
        """Resolve prompt-file path relative to agent file first, then repo base."""

        raw = Path(prompt_file)
        if raw.is_absolute():
            return raw

        candidate_agent_dir = agent_path.parent / raw
        if candidate_agent_dir.exists():
            return candidate_agent_dir
        return self.base_dir / raw


def build_allowed_tools_from_skills(skill_names: List[str], base_dir: Path) -> Set[str]:
    """Resolve allowed tools set from configured skill directories."""

    skill_dirs = [base_dir / "skills" / s for s in skill_names]
    _, allowed = load_skills(skill_dirs)
    return allowed


def format_task_user_message(task: BenchmarkTask) -> str:
    """Return task instruction payload for user-role message content."""

    return task.instruction
