from __future__ import annotations

from agents.performance_analytics_agent import PerformanceAnalyticsAgent
from models.agent_io import AgentRequest, Evidence, EvidenceSource
from tests.conftest import make_player_profile


def test_performance_analytics_agent_returns_named_response(fake_llm_client, man_utd_config):
    agent = PerformanceAnalyticsAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="How productive is Sample Striker?", club_id="manchester_united")

    response = agent.analyze(request)

    assert response.agent_name == "Performance Analytics Agent"
    assert len(fake_llm_client.calls) == 1


def test_performance_analytics_agent_system_prompt_flags_advanced_metric_gap(
    fake_llm_client, man_utd_config
):
    agent = PerformanceAnalyticsAgent(fake_llm_client, man_utd_config)

    prompt = agent.system_prompt()

    assert "xG" in prompt
    assert "do not" in prompt.lower()
    assert "genuinely unavailable" in prompt.lower()


def test_performance_analytics_agent_includes_fetched_evidence_when_available(
    fake_llm_client, man_utd_config
):
    evidence = [
        Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description="Sample Striker: Forward at Manchester United, age 23 (28 apps, 14g 5a)",
            as_of_date="2025-05-25",
        )
    ]
    profile = make_player_profile(stats_evidence=evidence)
    request = AgentRequest(
        query="How productive is Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
        player_profile=profile,
    )
    agent = PerformanceAnalyticsAgent(fake_llm_client, man_utd_config)

    prompt = agent.build_user_prompt(request)

    assert "Verified statistics (ground your assessment in this" in prompt
    assert "28 apps, 14g 5a" in prompt
