from __future__ import annotations

import pytest
from pydantic import ValidationError

from models.agent_io import AgentResponse, Evidence, EvidenceSource, ManagerSynthesis, FinalRecommendation


def test_agent_response_confidence_must_be_in_range():
    with pytest.raises(ValidationError):
        AgentResponse(summary="x", confidence=1.5)


def test_agent_response_defaults():
    response = AgentResponse(summary="Solid understudy for the right wing-back role.", confidence=0.6)
    assert response.supporting_evidence == []
    assert response.assumptions == []
    assert response.agent_name == ""


def test_evidence_requires_valid_source():
    with pytest.raises(ValidationError):
        Evidence(source="not_a_real_source", description="x")

    ev = Evidence(source=EvidenceSource.LLM_KNOWLEDGE, description="General knowledge estimate.")
    assert ev.confidence == 0.5


def test_final_recommendation_from_synthesis_attaches_agent_responses():
    synthesis = ManagerSynthesis(
        executive_summary="Summary.",
        recommendation="Sign him.",
        confidence=0.8,
    )
    responses = [AgentResponse(agent_name="Scout Agent", summary="Good technical profile.", confidence=0.7)]
    final = FinalRecommendation.from_synthesis(synthesis, responses)
    assert final.agent_responses == responses
    assert final.recommendation == "Sign him."
