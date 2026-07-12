from __future__ import annotations

from agents.transfer_market_agent import TransferMarketAgent
from models.agent_io import AgentRequest
from models.domain import PlayerStatsRecord
from tools.data_gateway import PlayerDataGateway
from tools.mock_provider import MockPlayerDataProvider


def test_transfer_market_agent_returns_named_response(fake_llm_client, man_utd_config):
    agent = TransferMarketAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="What would Sample Striker cost?", club_id="manchester_united")

    response = agent.analyze(request)

    assert response.agent_name == "Transfer Market Agent"
    assert len(fake_llm_client.calls) == 1


def test_transfer_market_agent_flags_unset_club_budget(fake_llm_client, man_utd_config):
    # The bundled manchester_united.yaml leaves both budget fields null.
    agent = TransferMarketAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="Can we afford him?", club_id="manchester_united")

    prompt = agent.build_user_prompt(request)

    assert "no transfer or wage budget figures are configured" in prompt


def test_transfer_market_agent_reports_configured_budget(fake_llm_client, man_utd_config):
    club = man_utd_config.model_copy(
        update={"transfer_budget_gbp": 100_000_000.0, "wage_budget_gbp_per_week": 250_000.0}
    )
    agent = TransferMarketAgent(fake_llm_client, club)
    request = AgentRequest(query="Can we afford him?", club_id="manchester_united")

    prompt = agent.build_user_prompt(request)

    assert "£100000000.0" in prompt
    assert "£250000.0" in prompt


def test_transfer_market_agent_includes_fetched_data_when_gateway_and_player_given(
    fake_llm_client, man_utd_config
):
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
    agent = TransferMarketAgent(fake_llm_client, man_utd_config, gateway)
    request = AgentRequest(
        query="What would Sample Striker cost?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    prompt = agent.build_user_prompt(request)

    assert "Fetched data (ground your assessment in this" in prompt
    assert "2025-05-25" in prompt
