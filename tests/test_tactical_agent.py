from __future__ import annotations

from agents.tactical_agent import TacticalAgent
from models.agent_io import AgentRequest


def test_tactical_agent_returns_named_response(fake_llm_client, man_utd_config):
    agent = TacticalAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="Where does Sample Striker fit tactically?", club_id="manchester_united")

    response = agent.analyze(request)

    assert response.agent_name == "Tactical Agent"
    assert len(fake_llm_client.calls) == 1


def test_tactical_agent_system_prompt_includes_club_formation_and_manager(fake_llm_client, man_utd_config):
    agent = TacticalAgent(fake_llm_client, man_utd_config)

    prompt = agent.system_prompt()

    assert "Ruben Amorim" in prompt
    assert "3-4-3" in prompt
    assert "tactical analyst" in prompt.lower()


def test_tactical_agent_user_prompt_includes_query_and_context(fake_llm_client, man_utd_config):
    agent = TacticalAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(
        query="Which striker best fits the system?",
        club_id="manchester_united",
        context={"candidate": "Sample Striker"},
    )

    prompt = agent.build_user_prompt(request)

    assert "Which striker best fits the system?" in prompt
    assert "candidate: Sample Striker" in prompt
