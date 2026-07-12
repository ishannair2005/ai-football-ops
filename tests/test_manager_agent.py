from __future__ import annotations

from agents.manager_agent import GeneralManagerAgent
from agents.scout_agent import ScoutAgent
from models.agent_io import AgentRequest


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
