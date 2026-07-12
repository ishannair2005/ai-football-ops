from __future__ import annotations

from agents.devils_advocate_agent import DevilsAdvocateAgent
from agents.manager_agent import GeneralManagerAgent
from agents.scout_agent import ScoutAgent
from models.agent_io import AgentRequest, AgentResponse, ManagerSynthesis, RecommendationVerdict
from services.llm_client import LLMClient


def test_manager_consults_specialists_and_synthesizes(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout])

    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.recommendation == "Fake recommendation."
    assert len(result.agent_responses) == 1
    assert result.agent_responses[0].agent_name == "Scout Agent"
    # One call for the scout, one for the manager's synthesis.
    assert len(fake_llm_client.calls) == 2


def test_manager_never_performs_specialist_analysis_itself(fake_llm_client, man_utd_config):
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [])
    request = AgentRequest(query="Any question.", club_id="manchester_united")

    result = manager.handle_query(request)

    assert result.agent_responses == []
    # Even with zero specialists, the manager still only calls the LLM
    # for synthesis, never fabricating a specialist-style finding.
    assert len(fake_llm_client.calls) == 1


def test_manager_runs_devils_advocate_challenge_when_configured(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    devils_advocate = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout], devils_advocate)

    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.devils_advocate_challenge is not None
    assert result.devils_advocate_challenge.agent_name == "Devil's Advocate"
    assert result.challenge_resolution == "Fake resolution."
    # scout + draft synthesis + challenge + resolution synthesis
    assert len(fake_llm_client.calls) == 4


class SequencedLLMClient(LLMClient):
    """Returns pre-scripted responses in call order, so a test can prove the
    Manager actually uses the resolution call's output rather than quietly
    keeping the draft — something the shared, canned FakeLLMClient can't
    demonstrate since it returns identical data every call."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, max_tokens=2048):
        self.calls.append({"response_model": response_model})
        return self._responses.pop(0)


def test_manager_uses_resolution_synthesis_not_draft(man_utd_config):
    specialist_response = AgentResponse(
        agent_name="Scout Agent", summary="Scout finding.", confidence=0.6
    )
    draft = ManagerSynthesis(
        executive_summary="Draft summary.",
        recommendation="Draft recommendation.",
        verdict=RecommendationVerdict.BUY,
        confidence=0.7,
        key_risks=["Draft risk."],
        next_steps=["Draft step."],
    )
    challenge_response = AgentResponse(
        agent_name="Devil's Advocate", summary="Strong opposing argument.", confidence=0.5
    )
    final_synthesis = ManagerSynthesis(
        executive_summary="Final summary.",
        recommendation="Final recommendation, revised.",
        verdict=RecommendationVerdict.MONITOR,
        confidence=0.55,
        key_risks=["Final risk."],
        next_steps=["Final step."],
        challenge_resolution="Partially accepted the challenge; lowered confidence and added a risk.",
    )
    client = SequencedLLMClient([specialist_response, draft, challenge_response, final_synthesis])
    scout = ScoutAgent(client, man_utd_config)
    devils_advocate = DevilsAdvocateAgent(client, man_utd_config)
    manager = GeneralManagerAgent(client, man_utd_config, [scout], devils_advocate)

    request = AgentRequest(query="Should United sign Sample Striker?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.recommendation == "Final recommendation, revised."
    assert result.recommendation != draft.recommendation
    assert result.verdict == RecommendationVerdict.MONITOR
    assert result.verdict != draft.verdict
    assert result.challenge_resolution == (
        "Partially accepted the challenge; lowered confidence and added a risk."
    )
    assert result.devils_advocate_challenge.summary == "Strong opposing argument."
    assert len(client.calls) == 4
