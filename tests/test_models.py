from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.agent_io import (
    AgentResponse,
    Evidence,
    EvidenceSource,
    FinalRecommendation,
    ManagerSynthesis,
    RecommendationVerdict,
)


def test_agent_response_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        AgentResponse(summary="x", confidence=1.5)


def test_agent_response_defaults():
    response = AgentResponse(summary="Solid understudy for the right wing-back role.", confidence=0.6)
    assert response.supporting_evidence == []
    assert response.evidence_gaps == []
    assert response.agent_name == ""


def test_agent_response_verified_facts_only_includes_data_provider_evidence():
    response = AgentResponse(
        summary="x",
        confidence=0.5,
        supporting_evidence=[
            Evidence(source=EvidenceSource.DATA_PROVIDER, description="Verified stat", as_of_date="2026-07-10"),
            Evidence(source=EvidenceSource.LLM_KNOWLEDGE, description="General knowledge guess"),
        ],
    )

    assert [e.description for e in response.verified_facts] == ["Verified stat"]


def test_evidence_requires_valid_source():
    with pytest.raises(ValidationError):
        Evidence(source="not_a_real_source", description="x")

    ev = Evidence(source=EvidenceSource.LLM_KNOWLEDGE, description="General knowledge estimate.")
    assert ev.confidence == 0.5


def test_final_recommendation_from_synthesis_attaches_agent_responses():
    synthesis = ManagerSynthesis(
        executive_summary="Summary.",
        recommendation="Sign him.",
        verdict=RecommendationVerdict.BUY,
        confidence=0.8,
    )
    responses = [AgentResponse(agent_name="Scout Agent", summary="Good technical profile.", confidence=0.7)]
    final = FinalRecommendation.from_synthesis(synthesis, responses)
    assert final.agent_responses == responses
    assert final.recommendation == "Sign him."
