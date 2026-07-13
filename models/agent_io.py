"""Structured I/O contracts shared by every agent.

Agents never exchange free text with each other or with the General
Manager. They exchange :class:`AgentResponse` objects, all of which share
the same evidence/evidence-gap vocabulary so the General Manager can
compare and reconcile findings across specialists mechanically rather
than by re-reading prose.

The platform never fills a factual gap with an assumption: missing
evidence is recorded as an explicit gap and reduces confidence, not
papered over with reasoning. ``Evidence`` with ``source ==
EvidenceSource.DATA_PROVIDER`` is what "verified" means throughout this
module — everything else (LLM knowledge, agent-derived findings) is
useful context but is never presented as a verified fact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class RecommendationVerdict(StrEnum):
    """The club's categorical decision on a transfer question."""

    BUY = "Buy"
    MONITOR = "Monitor"
    DO_NOT_SIGN = "Do Not Sign"


class EvidenceSource(StrEnum):
    """Where a piece of evidence originated.

    LLM_KNOWLEDGE marks facts drawn from the model's training data rather
    than a live data provider, so downstream consumers can flag staleness.
    Only DATA_PROVIDER evidence counts as a "verified fact" anywhere in
    this platform.
    """

    DATA_PROVIDER = "data_provider"
    LLM_KNOWLEDGE = "llm_knowledge"
    USER_PROVIDED = "user_provided"
    AGENT_FINDING = "agent_finding"


class EvidenceDomain(StrEnum):
    """Which kind of data an :class:`Evidence` item came from.

    Values double as the human-readable label shown in the "Data
    Freshness" breakdown, so freshness can be reported per data-type
    rather than collapsed into one blended date.
    """

    PLAYER_STATS = "Player statistics"
    INJURY = "Injury data"
    NEWS = "News"
    TRANSFER_MARKET = "Transfer market"
    OTHER = "Other"


class Evidence(BaseModel):
    source: EvidenceSource
    description: str = Field(..., description="What this evidence shows.")
    value: str | None = Field(default=None, description="The concrete data point, if any.")
    as_of_date: str | None = Field(
        default=None, description="Date the underlying data is known to be current as of."
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    domain: EvidenceDomain | None = Field(
        default=None, description="Which data-freshness bucket this evidence belongs to."
    )


class ResolvedIdentity(BaseModel):
    """A player's canonical identity, resolved from a raw, possibly
    ambiguous or misspelled, user-typed name."""

    full_name: str
    club: str | None = None
    player_ids: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "External IDs (e.g. fbref/fotmob/transfermarkt) when a resolver can supply "
            "them. Empty today — we have no real ID-issuing provider, and fabricating "
            "plausible-looking IDs would be exactly the kind of invention this exists "
            "to prevent."
        ),
    )
    as_of_date: str
    source: str


class PlayerProfile(BaseModel):
    """The single, verified player profile the Manager builds once and
    threads to every specialist and the Devil's Advocate, so no agent
    independently resolves identity or re-fetches the same player's data.

    ``evidence_gaps`` records every step that came back empty (identity
    not resolved, no stats found, no injury record found) so agents state
    the gap explicitly instead of reasoning around a silent hole.
    """

    queried_name: str
    resolved: bool
    full_name: str | None = None
    club: str | None = None
    player_ids: dict[str, str] = Field(default_factory=dict)
    identity_as_of: str | None = None
    identity_source: str | None = None
    stats_evidence: list[Evidence] = Field(default_factory=list)
    injury_evidence: list[Evidence] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    """The task handed to a specialist agent by the General Manager."""

    query: str = Field(..., description="The natural-language question or task.")
    club_id: str
    context: dict[str, str] = Field(
        default_factory=dict,
        description="Additional structured context (e.g. player names, budget, opponent).",
    )
    player_profile: PlayerProfile | None = Field(
        default=None,
        description="The Manager's pre-resolved player profile, when the query names a player.",
    )


class AgentResponse(BaseModel):
    """The structured contract every specialist agent must return."""

    agent_name: str = ""
    summary: str = Field(..., description="The agent's core finding in a few sentences.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_evidence: list[Evidence] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Facts that could not be verified from configured data providers, stated "
            "explicitly (e.g. 'Current injury status unavailable') — never filled in "
            "with an assumption."
        ),
    )
    recommended_next_steps: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def verified_facts(self) -> list[Evidence]:
        """The subset of ``supporting_evidence`` that's an actual verified
        fact (sourced from a data provider), never a stored field of its
        own so it can never drift from what was really fetched."""
        return [e for e in self.supporting_evidence if e.source == EvidenceSource.DATA_PROVIDER]


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
    verdict: RecommendationVerdict = Field(
        ..., description="The categorical decision: Buy, Monitor, or Do Not Sign."
    )
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
    player_profile: PlayerProfile | None = Field(
        default=None,
        description="The same verified profile the specialists saw, so the challenge "
        "can be grounded in the same facts rather than re-inferring them.",
    )
    news_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Recent news fetched once by the Manager — player-scoped when a "
        "player was named, club-scoped otherwise.",
    )
    news_subject: str | None = Field(
        default=None, description="Display label for what the news_evidence is about."
    )


class FinalRecommendation(BaseModel):
    """The General Manager's synthesis of all specialist responses, after
    resolving any Devil's Advocate challenge."""

    executive_summary: str
    recommendation: str
    verdict: RecommendationVerdict
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
            verdict=synthesis.verdict,
            confidence=synthesis.confidence,
            key_risks=synthesis.key_risks,
            next_steps=synthesis.next_steps,
            challenge_resolution=synthesis.challenge_resolution,
            agent_responses=agent_responses,
            devils_advocate_challenge=devils_advocate_challenge,
        )


class ReportRequest(BaseModel):
    """What the Report Agent receives: the original question and the
    Manager's fully resolved recommendation."""

    original_query: str
    club_id: str
    recommendation: FinalRecommendation


class ScoutingReport(BaseModel):
    """The Report Agent's output: a polished, presentable write-up of the
    Manager's recommendation. ``verdict``, ``confidence``, ``sources_used``,
    and ``data_freshness`` are overwritten programmatically after generation
    from ``ReportRequest.recommendation`` — the Report Agent writes the
    prose, it doesn't get to redecide or restate the facts underneath it."""

    executive_summary: str
    verdict: RecommendationVerdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    narrative: str = Field(
        ..., description="The full, flowing report covering every section by name."
    )
    sources_used: list[str] = Field(default_factory=list)
    data_freshness: dict[str, str] = Field(
        default_factory=dict,
        description="Earliest as-of date per EvidenceDomain label, e.g. "
        "{'Player statistics': '2026-07-10', 'Injury data': '2026-07-08'}.",
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PlatformResult(BaseModel):
    """What a full end-to-end query returns: the Manager's structured
    recommendation and the Report Agent's polished write-up of it."""

    recommendation: FinalRecommendation
    report: ScoutingReport
