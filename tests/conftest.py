from __future__ import annotations

import pytest

from config.club_config import ClubConfig
from models.agent_io import (
    AgentResponse,
    ManagerSynthesis,
    PlayerProfile,
    RecommendationVerdict,
    ScoutingReport,
)
from services.llm_client import LLMClient


def make_player_profile(**overrides) -> PlayerProfile:
    """Shared factory for a resolved PlayerProfile, used across specialist
    agent tests so each doesn't hand-build an identical one."""
    defaults = dict(
        queried_name="Sample Striker",
        resolved=True,
        full_name="Sample Striker",
        club="Manchester United",
        identity_as_of="2025-05-25",
        identity_source="sample_identities.csv",
        stats_evidence=[],
        injury_evidence=[],
        evidence_gaps=[],
    )
    defaults.update(overrides)
    return PlayerProfile(**defaults)


class FakeLLMClient(LLMClient):
    """Deterministic stand-in for AnthropicLLMClient.

    Returns a canned instance of whatever response_model is requested, so
    tests never make real network calls or depend on API keys.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, max_tokens=2048):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_model": response_model,
            }
        )
        if response_model is AgentResponse:
            return AgentResponse(
                summary="Fake specialist finding.",
                confidence=0.7,
                evidence_gaps=["No live data source connected in this test."],
                recommended_next_steps=["Connect a real data provider."],
            )
        if response_model is ManagerSynthesis:
            return ManagerSynthesis(
                executive_summary="Fake executive summary.",
                recommendation="Fake recommendation.",
                verdict=RecommendationVerdict.MONITOR,
                confidence=0.65,
                key_risks=["Fake risk."],
                next_steps=["Fake next step."],
                challenge_resolution="Fake resolution.",
            )
        if response_model is ScoutingReport:
            return ScoutingReport(
                executive_summary="Fake report executive summary.",
                verdict=RecommendationVerdict.MONITOR,
                confidence=0.5,
                narrative="Fake full narrative report.",
            )
        raise AssertionError(f"FakeLLMClient asked for unsupported model: {response_model}")


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def man_utd_config() -> ClubConfig:
    from config.club_config import load_club_config

    return load_club_config("manchester_united")
