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
        spec = AgentSpec(
            name=data["name"],
            backend=data["backend"],
            prompt_template=data["prompt_template"],
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


def build_allowed_tools_from_skills(skill_names: List[str], base_dir: Path) -> Set[str]:
    """Resolve allowed tools set from configured skill directories."""

    skill_dirs = [base_dir / "skills" / s for s in skill_names]
    _, allowed = load_skills(skill_dirs)
    return allowed


def format_task_user_message(task: BenchmarkTask) -> str:
    """Return task instruction payload for user-role message content."""

    return task.instruction
