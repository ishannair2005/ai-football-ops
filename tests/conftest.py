from __future__ import annotations

import pytest

from config.club_config import ClubConfig
from models.agent_io import AgentResponse, ManagerSynthesis
from services.llm_client import LLMClient


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
                assumptions=["Assumed fake data is representative."],
                uncertainties=["No live data source connected in this test."],
                recommended_next_steps=["Connect a real data provider."],
            )
        if response_model is ManagerSynthesis:
            return ManagerSynthesis(
                executive_summary="Fake executive summary.",
                recommendation="Fake recommendation.",
                confidence=0.65,
                key_risks=["Fake risk."],
                next_steps=["Fake next step."],
                challenge_resolution="Fake resolution.",
            )
        raise AssertionError(f"FakeLLMClient asked for unsupported model: {response_model}")


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def man_utd_config() -> ClubConfig:
    from config.club_config import load_club_config

    return load_club_config("manchester_united")
