from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import yaml

from agent_architectures.constants import ARCHITECTURE_NONE, normalize_architecture_id
from skills.loader import load_skills


@dataclass
class AgentSpec:
    """Parsed agent profile schema used by runtime services."""

    name: str
    backend: Dict[str, Any]
    prompt_template: str
    tools: Optional[List[str]]
    skills: List[str]
    tool_to_skill_map: Dict[str, List[str]]
    termination: Dict[str, Any]
    decoding_defaults: Dict[str, Any]
    agent_architecture: str
    agent_architecture_config: Dict[str, Any]


class AgentSpecLoader:
    """Loader for agent YAML profiles and resolved skill prompt text."""

    def __init__(self, base_dir: Path) -> None:
        """Keep repository base path for skill directory resolution."""

        self.base_dir = base_dir

    def load(
        self,
        path: Path,
        *,
        runtime_mode: Optional[Literal["patch_only", "tools_enabled"]] = None,
    ) -> Tuple[AgentSpec, str, Optional[Set[str]]]:
        """Load agent spec, render final system prompt, and derive allowed tools."""

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        prompt_template = self._resolve_prompt_template(data, path)
        tools = self._normalize_tools_field(data["tools"]) if "tools" in data else None
        spec = AgentSpec(
            name=data["name"],
            backend=data["backend"],
            prompt_template=prompt_template,
            tools=tools,
            skills=self._normalize_skills_field(data.get("skills", [])),
            tool_to_skill_map=data.get("tool_to_skill_map", {}),
            termination=data.get("termination", {}),
            decoding_defaults=data.get("decoding_defaults", {}),
            agent_architecture=self._normalize_agent_architecture(data.get("agent_architecture")),
            agent_architecture_config=self._normalize_architecture_config(
                data.get("agent_architecture_config")
            ),
        )
        skill_dirs = [self.base_dir / "skills" / s for s in spec.skills]
        skills_text, skill_allowed_tools = load_skills(skill_dirs)
        allowed_tools = self._compute_effective_allowed_tools(
            explicit_tools=spec.tools,
            skill_allowed_tools=skill_allowed_tools,
            has_skills=bool(spec.skills),
        )
        prompt = self.render_prompt(spec, skills_text, runtime_mode=runtime_mode)
        return spec, prompt, allowed_tools

    def render_prompt(
        self,
        spec: AgentSpec,
        skills_text: str,
        *,
        runtime_mode: Optional[Literal["patch_only", "tools_enabled"]] = None,
    ) -> str:
        """Render profile prompt template by injecting loaded skills text."""

        base = spec.prompt_template
        if spec.skills and "{skills}" not in base and runtime_mode != "patch_only":
            raise ValueError(
                "Prompt template is missing required `{skills}` placeholder while skills are configured"
            )
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

    def _normalize_tools_field(self, raw_tools: Any) -> List[str]:
        """Normalize agent `tools` field to a deduplicated list of tool names."""

        if raw_tools is None:
            return []
        if not isinstance(raw_tools, list):
            raise ValueError("`tools` must be a list of tool names")

        names: List[str] = []
        for idx, item in enumerate(raw_tools):
            name: Optional[str] = None
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                raw_name = item.get("name")
                if isinstance(raw_name, str):
                    name = raw_name.strip()
            if not name:
                raise ValueError(
                    f"`tools[{idx}]` must be a non-empty string or object with non-empty `name`"
                )
            names.append(name)

        deduped: List[str] = []
        seen: Set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _normalize_skills_field(self, raw_skills: Any) -> List[str]:
        """Normalize agent `skills` field to a deduplicated list of skill names."""

        if raw_skills is None:
            return []
        if not isinstance(raw_skills, list):
            raise ValueError("`skills` must be a list of skill names")

        names: List[str] = []
        for idx, item in enumerate(raw_skills):
            if item is None:
                # Be forgiving for YAML entries like `-` that parse as null.
                continue
            if not isinstance(item, str):
                raise ValueError(f"`skills[{idx}]` must be a string skill name")
            name = item.strip()
            if not name:
                continue
            names.append(name)

        deduped: List[str] = []
        seen: Set[str] = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            deduped.append(name)
        return deduped

    def _compute_effective_allowed_tools(
        self,
        *,
        explicit_tools: Optional[List[str]],
        skill_allowed_tools: Set[str],
        has_skills: bool,
    ) -> Optional[Set[str]]:
        """Compute effective runtime allowlist from explicit tools and skills."""

        explicit = set(explicit_tools) if explicit_tools is not None else None
        if has_skills:
            if explicit is None:
                return set(skill_allowed_tools)
            if len(explicit) == 0:
                # Empty tools list with skills means "defer to skills allowlist".
                return set(skill_allowed_tools)
            return set(skill_allowed_tools).intersection(explicit)
        if explicit is None:
            # Backward-compatible tools_enabled behavior when neither skills nor tools constrain tools.
            return None
        return explicit

    def _normalize_agent_architecture(self, value: Any) -> str:
        """Normalize architecture id from optional profile field."""

        if value is None:
            return ARCHITECTURE_NONE
        return normalize_architecture_id(value)

    @staticmethod
    def _normalize_architecture_config(value: Any) -> Dict[str, Any]:
        """Normalize optional architecture config block to a dict payload."""

        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("`agent_architecture_config` must be an object when provided")
        return dict(value)

def build_allowed_tools_from_skills(skill_names: List[str], base_dir: Path) -> Set[str]:
    """Resolve allowed tools set from configured skill directories."""

    skill_dirs = [base_dir / "skills" / s for s in skill_names]
    _, allowed = load_skills(skill_dirs)
    return allowed
