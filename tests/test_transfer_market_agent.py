from __future__ import annotations

from agents.transfer_market_agent import TransferMarketAgent
from models.agent_io import AgentRequest, Evidence, EvidenceSource
from tests.conftest import make_player_profile


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


def test_transfer_market_agent_includes_fetched_data_when_profile_given(fake_llm_client, man_utd_config):
    evidence = [
        Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description="Sample Striker stats",
            as_of_date="2025-05-25",
        )
    ]
    profile = make_player_profile(stats_evidence=evidence)
    request = AgentRequest(
        query="What would Sample Striker cost?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
        player_profile=profile,
    )
    agent = TransferMarketAgent(fake_llm_client, man_utd_config)

    prompt = agent.build_user_prompt(request)

    assert "Verified statistics (ground your assessment in this" in prompt
    assert "2025-05-25" in prompt
