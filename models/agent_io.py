"""Structured I/O contracts shared by every agent.

Agents never exchange free text with each other or with the General
Manager. They exchange :class:`AgentResponse` objects, all of which share
the same evidence/assumption/uncertainty vocabulary so the General Manager
can compare and reconcile findings across specialists mechanically rather
than by re-reading prose.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class EvidenceSource(StrEnum):
    """Where a piece of evidence originated.

    LLM_KNOWLEDGE marks facts drawn from the model's training data rather
    than a live data provider, so downstream consumers can flag staleness.
    """

    DATA_PROVIDER = "data_provider"
    LLM_KNOWLEDGE = "llm_knowledge"
    USER_PROVIDED = "user_provided"
    AGENT_FINDING = "agent_finding"


class Evidence(BaseModel):
    source: EvidenceSource
    description: str = Field(..., description="What this evidence shows.")
    value: str | None = Field(default=None, description="The concrete data point, if any.")
    as_of_date: str | None = Field(
        default=None, description="Date the underlying data is known to be current as of."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentRequest(BaseModel):
    """The task handed to a specialist agent by the General Manager."""

    query: str = Field(..., description="The natural-language question or task.")
    club_id: str
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional structured context (e.g. player names, budget, opponent).",
    )


class AgentResponse(BaseModel):
    """The structured contract every specialist agent must return."""

    agent_name: str = ""
    summary: str = Field(..., description="The agent's core finding in a few sentences.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ManagerSynthesis(BaseModel):
    """What the LLM produces when the General Manager reconciles specialist
    findings. Deliberately excludes ``agent_responses`` — the manager
    attaches those programmatically rather than asking the LLM to restate
    them, which would waste tokens and risk transcription drift.

    Reused for both the draft synthesis (before any Devil's Advocate
    challenge exists) and the final, post-challenge synthesis — only the
    latter populates ``challenge_resolution``.
    """

    executive_summary: str
    recommendation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_risks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    challenge_resolution: str | None = Field(
        default=None,
        description=(
            "How the Devil's Advocate's challenge was addressed — accepted, "
            "partially accepted, or rejected, and why. Populated only on the "
            "post-challenge synthesis, never the draft."
        ),
    )


class ChallengeRequest(BaseModel):
    """What the Devil's Advocate receives: the Manager's draft recommendation
    and the specialist findings it's built on, instead of a raw query."""

    original_query: str
    club_id: str
    draft_recommendation: ManagerSynthesis
    specialist_responses: list[AgentResponse]


class FinalRecommendation(BaseModel):
    """The General Manager's synthesis of all specialist responses, after
    resolving any Devil's Advocate challenge."""

    executive_summary: str
    recommendation: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_risks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    challenge_resolution: str | None = None
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    devils_advocate_challenge: AgentResponse | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_synthesis(
        cls,
        synthesis: ManagerSynthesis,
        agent_responses: list[AgentResponse],
        devils_advocate_challenge: AgentResponse | None = None,
    ) -> "FinalRecommendation":
        return cls(
            executive_summary=synthesis.executive_summary,
            recommendation=synthesis.recommendation,
            confidence=synthesis.confidence,
            key_risks=synthesis.key_risks,
            next_steps=synthesis.next_steps,
            challenge_resolution=synthesis.challenge_resolution,
            agent_responses=agent_responses,
            devils_advocate_challenge=devils_advocate_challenge,
        )
