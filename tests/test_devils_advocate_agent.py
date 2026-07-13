from __future__ import annotations

from agents.devils_advocate_agent import DevilsAdvocateAgent
from models.agent_io import (
    AgentResponse,
    ChallengeRequest,
    Evidence,
    EvidenceSource,
    ManagerSynthesis,
    RecommendationVerdict,
)


def _challenge_request(**overrides) -> ChallengeRequest:
    draft = ManagerSynthesis(
        executive_summary="United should pursue the signing.",
        recommendation="Sign the player in the summer window.",
        verdict=RecommendationVerdict.BUY,
        confidence=0.75,
        key_risks=["Injury history unclear."],
        next_steps=["Arrange a medical."],
    )
    specialist_responses = [
        AgentResponse(
            agent_name="Scout Agent",
            summary="Strong technical profile for the position.",
            confidence=0.8,
        )
    ]
    defaults = dict(
        original_query="Should United sign Sample Striker?",
        club_id="manchester_united",
        draft_recommendation=draft,
        specialist_responses=specialist_responses,
    )
    defaults.update(overrides)
    return ChallengeRequest(**defaults)


def test_devils_advocate_system_prompt_includes_club_context(fake_llm_client, man_utd_config):
    agent = DevilsAdvocateAgent(fake_llm_client, man_utd_config)

    prompt = agent.system_prompt()

    assert "Manchester United" in prompt
    assert "devil's advocate" in prompt.lower()


def test_devils_advocate_user_prompt_includes_query_draft_and_findings(fake_llm_client, man_utd_config):
    agent = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    request = _challenge_request()

    prompt = agent.build_user_prompt(request)

    assert "Should United sign Sample Striker?" in prompt
    assert "Sign the player in the summer window." in prompt
    assert "Scout Agent" in prompt
    assert "Strong technical profile for the position." in prompt


def test_devils_advocate_analyze_returns_named_response(fake_llm_client, man_utd_config):
    agent = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    request = _challenge_request()

    response = agent.analyze(request)

    assert response.agent_name == "Devil's Advocate"
    assert len(fake_llm_client.calls) == 1


def test_devils_advocate_prompt_includes_news_when_given(fake_llm_client, man_utd_config):
    news_evidence = [
        Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description="[rumour] Sample Striker training ground bust-up reported — "
            "Unconfirmed reports of a dressing room disagreement.",
            as_of_date="2025-05-10",
        )
    ]
    agent = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    request = _challenge_request(news_evidence=news_evidence, news_subject="Sample Striker")

    prompt = agent.build_user_prompt(request)

    assert "Sample Striker training ground bust-up reported" in prompt
    assert "2025-05-10" in prompt
