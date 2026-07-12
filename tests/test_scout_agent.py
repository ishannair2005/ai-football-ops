from __future__ import annotations

from agents.scout_agent import ScoutAgent
from models.agent_io import AgentRequest


def test_scout_agent_returns_named_response(fake_llm_client, man_utd_config):
    agent = ScoutAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="Scout Joao Neves.", club_id="manchester_united")

    response = agent.analyze(request)

    assert response.agent_name == "Scout Agent"
    assert response.confidence == 0.7
    assert len(fake_llm_client.calls) == 1


def test_scout_agent_system_prompt_includes_club_context(fake_llm_client, man_utd_config):
    agent = ScoutAgent(fake_llm_client, man_utd_config)
    prompt = agent.system_prompt()

    assert "Manchester United" in prompt
    assert "Ruben Amorim" in prompt
    assert "chief scout" in prompt.lower()
