from pathlib import Path

from skills.loader import load_skills


def test_allowed_tools_parsed(tmp_path: Path):
    skill = tmp_path / "s" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text("""Skill\n\nAllowed Tools:\n- a\n- b\n""")
    text, allowed = load_skills([skill.parent])
    assert "Skill" in text
    assert allowed == {"a", "b"}
