from __future__ import annotations

from agents.scout_agent import ScoutAgent
from models.agent_io import AgentRequest
from models.domain import PlayerStatsRecord
from tools.data_gateway import PlayerDataGateway
from tools.mock_provider import MockPlayerDataProvider


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


def test_scout_agent_prompt_has_no_fetched_data_without_player_context(fake_llm_client, man_utd_config):
    agent = ScoutAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="Which positions should we strengthen?", club_id="manchester_united")

    prompt = agent.build_user_prompt(request)

    assert "Fetched data: none available" in prompt


def test_scout_agent_prompt_flags_missing_record_for_named_player(fake_llm_client, man_utd_config):
    gateway = PlayerDataGateway(providers=[MockPlayerDataProvider(records={})])
    agent = ScoutAgent(fake_llm_client, man_utd_config, gateway)
    request = AgentRequest(
        query="Should we sign him?",
        club_id="manchester_united",
        context={"player": "Nobody FC"},
    )

    prompt = agent.build_user_prompt(request)

    assert "no data-provider record found" in prompt


def test_scout_agent_prompt_includes_fetched_evidence_when_available(fake_llm_client, man_utd_config):
    record = PlayerStatsRecord(
        name="Sample Striker",
        position="Forward",
        club="Manchester United",
        age=23,
        as_of_date="2025-05-25",
        source="sample_players.csv",
    )
    gateway = PlayerDataGateway(
        providers=[MockPlayerDataProvider(records={"sample striker": record})]
    )
    agent = ScoutAgent(fake_llm_client, man_utd_config, gateway)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    prompt = agent.build_user_prompt(request)

    assert "Fetched data (ground your assessment in this" in prompt
    assert "Sample Striker" in prompt
    assert "2025-05-25" in prompt
