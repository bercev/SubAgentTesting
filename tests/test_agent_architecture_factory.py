from agent_architectures.constants import (
    ARCHITECTURE_MINI_SWE_AGENT,
    ARCHITECTURE_NONE,
)
from agent_architectures.factory import get_agent_architecture, resolve_agent_architecture
from agent_architectures.legacy import LegacyArchitecture
from agent_architectures.mini_swe_agent import MiniSweAgentArchitecture


def test_resolve_agent_architecture_precedence_cli_over_run_over_profile():
    resolved = resolve_agent_architecture(
        cli_override="mini-swe-agent",
        run_override="none",
        profile_architecture="none",
    )
    assert resolved == ARCHITECTURE_MINI_SWE_AGENT



def test_resolve_agent_architecture_uses_run_override_when_cli_missing():
    resolved = resolve_agent_architecture(
        cli_override=None,
        run_override="mini-swe-agent",
        profile_architecture="none",
    )
    assert resolved == ARCHITECTURE_MINI_SWE_AGENT



def test_resolve_agent_architecture_defaults_to_none_when_all_missing():
    resolved = resolve_agent_architecture(
        cli_override=None,
        run_override=None,
        profile_architecture=None,
    )
    assert resolved == ARCHITECTURE_NONE



def test_get_agent_architecture_returns_legacy_for_none():
    architecture = get_agent_architecture("none")
    assert isinstance(architecture, LegacyArchitecture)



def test_get_agent_architecture_returns_mini_for_mini_swe_agent():
    architecture = get_agent_architecture("mini-swe-agent")
    assert isinstance(architecture, MiniSweAgentArchitecture)
