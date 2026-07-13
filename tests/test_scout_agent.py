from __future__ import annotations

from agents.scout_agent import ScoutAgent
from models.agent_io import AgentRequest, Evidence, EvidenceSource
from tests.conftest import make_player_profile


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
    assert "Michael Carrick" in prompt
    assert "chief scout" in prompt.lower()


def test_scout_agent_prompt_has_no_fetched_data_without_player_profile(fake_llm_client, man_utd_config):
    agent = ScoutAgent(fake_llm_client, man_utd_config)
    request = AgentRequest(query="Which positions should we strengthen?", club_id="manchester_united")

    prompt = agent.build_user_prompt(request)

    assert "no player named" in prompt.lower()
    assert "none available" in prompt.lower()


def test_scout_agent_prompt_flags_gap_for_unresolved_player(fake_llm_client, man_utd_config):
    agent = ScoutAgent(fake_llm_client, man_utd_config)
    profile = make_player_profile(queried_name="Nobody FC", resolved=False, full_name=None, club=None)
    request = AgentRequest(
        query="Should we sign him?",
        club_id="manchester_united",
        context={"player": "Nobody FC"},
        player_profile=profile,
    )

    prompt = agent.build_user_prompt(request)

    assert "could not be verified" in prompt


def test_scout_agent_prompt_includes_fetched_evidence_when_available(fake_llm_client, man_utd_config):
    evidence = [
        Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description="Sample Striker: Forward at Manchester United, age 23",
            as_of_date="2025-05-25",
        )
    ]
    profile = make_player_profile(stats_evidence=evidence)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
        player_profile=profile,
    )
    agent = ScoutAgent(fake_llm_client, man_utd_config)

    prompt = agent.build_user_prompt(request)

    assert "Verified statistics (ground your assessment in this" in prompt
    assert "Sample Striker" in prompt
    assert "2025-05-25" in prompt


def test_scout_agent_prompt_includes_injury_evidence_when_available(fake_llm_client, man_utd_config):
    evidence = [
        Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description="Sample Winger: Doubtful (Hamstring strain, expected return 2025-06-10)",
            as_of_date="2025-05-25",
        )
    ]
    profile = make_player_profile(
        queried_name="Sample Winger", full_name="Sample Winger", injury_evidence=evidence
    )
    request = AgentRequest(
        query="Should we sign Sample Winger?",
        club_id="manchester_united",
        context={"player": "Sample Winger"},
        player_profile=profile,
    )
    agent = ScoutAgent(fake_llm_client, man_utd_config)

    prompt = agent.build_user_prompt(request)

    assert "Injury data (cite this" in prompt
    assert "Doubtful" in prompt
    assert "Hamstring strain" in prompt
