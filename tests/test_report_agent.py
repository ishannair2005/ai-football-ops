from __future__ import annotations

from agents.report_agent import ReportAgent
from models.agent_io import (
    AgentResponse,
    Evidence,
    EvidenceDomain,
    EvidenceSource,
    FinalRecommendation,
    ReportRequest,
    RecommendationVerdict,
)


def _recommendation() -> FinalRecommendation:
    scout_response = AgentResponse(
        agent_name="Scout Agent",
        summary="Strong technical profile.",
        confidence=0.8,
        supporting_evidence=[
            Evidence(
                source=EvidenceSource.DATA_PROVIDER,
                description="Sample Striker stats",
                as_of_date="2025-05-25",
                domain=EvidenceDomain.PLAYER_STATS,
            ),
        ],
    )
    tactical_response = AgentResponse(
        agent_name="Tactical Agent",
        summary="Good fit for the front three.",
        confidence=0.7,
    )
    transfer_response = AgentResponse(
        agent_name="Transfer Market Agent",
        summary="Fee likely within budget.",
        confidence=0.6,
        supporting_evidence=[
            Evidence(
                source=EvidenceSource.LLM_KNOWLEDGE,
                description="General market estimate",
                as_of_date="2025-04-01",
                domain=EvidenceDomain.TRANSFER_MARKET,
            ),
        ],
    )
    challenge = AgentResponse(
        agent_name="Devil's Advocate",
        summary="Squad already has cover at this position.",
        confidence=0.4,
        supporting_evidence=[
            Evidence(
                source=EvidenceSource.DATA_PROVIDER,
                description="Sample Striker stats",
                as_of_date="2025-05-25",
                domain=EvidenceDomain.PLAYER_STATS,
            ),
        ],
    )
    return FinalRecommendation(
        executive_summary="Draft executive summary.",
        recommendation="Sign him in the summer window.",
        verdict=RecommendationVerdict.BUY,
        confidence=0.65,
        agent_responses=[scout_response, tactical_response, transfer_response],
        devils_advocate_challenge=challenge,
        challenge_resolution="Accepted the challenge partially; monitor squad depth.",
    )


def test_report_agent_prompt_includes_each_specialist_by_name(fake_llm_client, man_utd_config):
    agent = ReportAgent(fake_llm_client, man_utd_config)
    request = ReportRequest(
        original_query="Should United sign Sample Striker?",
        club_id="manchester_united",
        recommendation=_recommendation(),
    )

    prompt = agent.build_user_prompt(request)

    assert "Scout Agent" in prompt
    assert "Tactical Agent" in prompt
    assert "Transfer Market Agent" in prompt
    assert "Devil's Advocate" in prompt
    assert "Strong technical profile." in prompt


def test_collect_sources_dedupes_across_specialists_and_challenge():
    recommendation = _recommendation()

    sources = ReportAgent._collect_sources(recommendation)

    # Scout and the Devil's Advocate cite the same (source, date) pair —
    # it should appear once, not twice.
    assert sources == [
        "data_provider (as of 2025-05-25)",
        "llm_knowledge (as of 2025-04-01)",
    ]


def test_collect_data_freshness_groups_by_domain_and_picks_earliest():
    recommendation = _recommendation()

    freshness = ReportAgent._collect_data_freshness(recommendation)

    assert freshness == {
        "Player statistics": "2025-05-25",
        "Transfer market": "2025-04-01",
    }


def test_collect_data_freshness_skips_evidence_without_a_domain():
    recommendation = FinalRecommendation(
        executive_summary="x",
        recommendation="x",
        verdict=RecommendationVerdict.MONITOR,
        confidence=0.5,
        agent_responses=[
            AgentResponse(
                agent_name="Scout Agent",
                summary="x",
                confidence=0.5,
                supporting_evidence=[
                    Evidence(
                        source=EvidenceSource.DATA_PROVIDER,
                        description="no domain tagged",
                        as_of_date="2025-01-01",
                    )
                ],
            )
        ],
    )

    assert ReportAgent._collect_data_freshness(recommendation) == {}


def test_collect_data_freshness_is_empty_when_no_evidence_has_dates():
    recommendation = FinalRecommendation(
        executive_summary="x",
        recommendation="x",
        verdict=RecommendationVerdict.MONITOR,
        confidence=0.5,
        agent_responses=[AgentResponse(agent_name="Scout Agent", summary="x", confidence=0.5)],
    )

    assert ReportAgent._collect_data_freshness(recommendation) == {}
    assert ReportAgent._collect_sources(recommendation) == []


def test_analyze_overrides_verdict_confidence_sources_and_freshness(fake_llm_client, man_utd_config):
    agent = ReportAgent(fake_llm_client, man_utd_config)
    recommendation = _recommendation()
    request = ReportRequest(
        original_query="Should United sign Sample Striker?",
        club_id="manchester_united",
        recommendation=recommendation,
    )

    report = agent.analyze(request)

    # The fake LLM client's canned ScoutingReport says verdict=MONITOR,
    # confidence=0.5 — analyze() must override both from the real
    # recommendation (verdict=BUY, confidence=0.65) rather than trusting
    # the LLM's restatement.
    assert report.verdict == RecommendationVerdict.BUY
    assert report.confidence == 0.65
    assert report.sources_used == [
        "data_provider (as of 2025-05-25)",
        "llm_knowledge (as of 2025-04-01)",
    ]
    assert report.data_freshness == {
        "Player statistics": "2025-05-25",
        "Transfer market": "2025-04-01",
    }
