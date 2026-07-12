from __future__ import annotations

from models.agent_io import AgentRequest, PlatformResult, RecommendationVerdict
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
    assert result.report.data_as_of is None
