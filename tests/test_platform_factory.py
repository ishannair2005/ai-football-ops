from __future__ import annotations

from agents.manager_agent import GeneralManagerAgent
from models.agent_io import (
    AgentRequest,
    AgentResponse,
    ComparisonRecommendation,
    PlatformResult,
    RecommendationVerdict,
)
from services.platform_factory import build_general_manager, build_platform


def test_build_general_manager_assembles_all_specialists(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    result = manager.handle_query(request)

    names = {response.agent_name for response in result.agent_responses}
    assert names == {
        "Scout Agent",
        "Tactical Agent",
        "Transfer Market Agent",
        "Performance Analytics Agent",
    }
    assert result.devils_advocate_challenge is not None
    assert result.devils_advocate_challenge.agent_name == "Devil's Advocate"
    # 4 specialists + draft synthesis + Devil's Advocate challenge + final resolution.
    assert len(fake_llm_client.calls) == 7


def test_build_general_manager_uses_csv_backed_data_for_bundled_sample_player(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    manager.handle_query(request)

    scout_call = next(
        call
        for call in fake_llm_client.calls
        if "Scouting task" in call["user_prompt"]
    )
    assert "Sample Striker: Forward at Manchester United" in scout_call["user_prompt"]
    assert "Injury data" in scout_call["user_prompt"]


def test_build_general_manager_resolves_bundled_player_identity(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)

    profile = manager._resolve_player_profile("Sample Striker")

    assert profile.resolved is True
    assert profile.full_name == "Sample Striker"
    assert profile.club == "Manchester United"


def test_build_general_manager_reports_gap_for_unresolvable_player(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)

    profile = manager._resolve_player_profile("Nobody FC")

    assert profile.resolved is False
    assert any("could not be verified" in gap for gap in profile.evidence_gaps)


def test_build_platform_returns_recommendation_and_report(fake_llm_client):
    platform = build_platform("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    result = platform.handle_query(request)

    assert isinstance(result, PlatformResult)
    assert result.report.verdict == result.recommendation.verdict
    assert result.report.confidence == result.recommendation.confidence
    assert result.report.verdict == RecommendationVerdict.MONITOR


def test_build_platform_report_collects_sources_and_freshness(fake_llm_client):
    platform = build_platform("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    result = platform.handle_query(request)

    # The fake LLM's canned AgentResponse has no supporting_evidence, so the
    # report's programmatic aggregation should honestly report nothing found
    # rather than fabricating a source or date.
    assert result.report.sources_used == []
    assert result.report.data_freshness == {}


def test_build_platform_forwards_callbacks_and_adds_report_status(fake_llm_client):
    platform = build_platform("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    statuses: list[str] = []
    agent_names: list[str] = []

    platform.handle_query(
        request,
        on_status=statuses.append,
        on_agent_response=lambda response: agent_names.append(response.agent_name),
    )

    # The Manager's own statuses/agent-responses (4 specialists + draft +
    # challenge + resolution) plus one more status for the report step.
    assert statuses[-1] == "Writing the final report..."
    assert statuses.count("Writing the final report...") == 1
    assert agent_names == [
        "Scout Agent",
        "Tactical Agent",
        "Transfer Market Agent",
        "Performance Analytics Agent",
        "Devil's Advocate",
    ]


def test_build_general_manager_comparison_end_to_end(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(query="Compare three strikers", club_id="manchester_united")

    result = manager._handle_comparison(
        request, ["Sample Striker", "Sample Winger", "Sample Midfielder"]
    )

    assert isinstance(result, ComparisonRecommendation)
    assert result.player_names == ["Sample Striker", "Sample Winger", "Sample Midfielder"]
    assert len(result.players) == 3
    assert len(result.decision_matrix) == 4
    for criterion in result.decision_matrix:
        assert len(criterion.ratings) == 3


def test_platform_handle_query_skips_report_agent_for_comparisons(man_utd_config):
    from agents.report_agent import ReportAgent
    from agents.scout_agent import ScoutAgent
    from models.agent_io import ComparisonOutcome, ComparisonSynthesis, PlayerNameExtraction
    from services.platform_factory import FootballOperationsPlatform
    from tests.test_manager_agent import SequencedLLMClient

    # Only 4 responses queued: extraction + 2 specialist calls + comparison
    # synthesis. If the Report Agent were (wrongly) invoked afterwards, it
    # would try to pop a 5th response that doesn't exist and raise --
    # reaching the assertions below proves it was correctly skipped.
    client = SequencedLLMClient(
        [
            PlayerNameExtraction(players=["Sample Striker", "Sample Winger"]),
            AgentResponse(agent_name="Scout Agent", summary="a", confidence=0.6),
            AgentResponse(agent_name="Scout Agent", summary="b", confidence=0.5),
            ComparisonSynthesis(
                executive_summary="s",
                outcome=ComparisonOutcome.MULTIPLE_VIABLE,
                verdict_rationale="r",
                confidence=0.5,
            ),
        ]
    )
    scout = ScoutAgent(client, man_utd_config)
    manager = GeneralManagerAgent(client, man_utd_config, [scout])
    report_agent = ReportAgent(client, man_utd_config)
    platform = FootballOperationsPlatform(manager, report_agent)
    request = AgentRequest(
        query="Compare Sample Striker vs Sample Winger", club_id="manchester_united"
    )

    result = platform.handle_query(request)

    assert isinstance(result, ComparisonRecommendation)
    assert result.outcome == ComparisonOutcome.MULTIPLE_VIABLE
    assert len(client.calls) == 4
