"""Devil's Advocate Agent: argues the strongest case against a draft recommendation."""

from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from models.agent_io import AgentResponse, ChallengeRequest
from prompts.data_prompts import build_data_quality_section, build_identity_section, build_news_section

ROLE_DESCRIPTION = """
You are the club's Devil's Advocate. You are given the General Manager's
draft recommendation and the specialist findings underpinning it. Your
job is to argue the strongest possible case AGAINST the current
recommendation:
- Identify hidden risks the draft may have underweighted or ignored
- Question whether the specialists (and the draft) treated an
  unverified assumption as settled fact rather than stating it as a gap
- Point out any conflicting evidence between specialists that was
  glossed over rather than resolved
- Weigh in any recent news below — a rumour of interest from elsewhere, a
  lukewarm manager comment, or a contract dispute is exactly the kind of
  thing a draft recommendation might have missed
- Produce the single strongest opposing argument, not a scattershot list
  of minor nitpicks

You are a structured adversarial check, not a balanced second opinion —
argue against the recommendation as hard as the evidence allows. But you
must still be honest: if you genuinely cannot find a strong opposing
argument, say so explicitly and rate your own challenge's confidence low
rather than manufacturing a weak objection to seem useful.
""".strip()


class DevilsAdvocateAgent(BaseAgent[ChallengeRequest, AgentResponse]):
    @property
    def name(self) -> str:
        return "Devil's Advocate"

    @property
    def role_description(self) -> str:
        return ROLE_DESCRIPTION

    @property
    def response_model(self) -> type[AgentResponse]:
        return AgentResponse

    def build_user_prompt(self, request: ChallengeRequest) -> str:
        specialist_findings = [
            {
                "agent_name": r.agent_name,
                "summary": r.summary,
                "confidence": r.confidence,
                "evidence_gaps": r.evidence_gaps,
            }
            for r in request.specialist_responses
        ]
        return f"""
Original question: {request.original_query}

Draft recommendation from the General Manager (JSON):
{json.dumps(request.draft_recommendation.model_dump(mode="json"), indent=2)}

Specialist findings underpinning that draft (JSON):
{json.dumps(specialist_findings, indent=2)}

{build_data_quality_section(request.player_profile)}

{build_identity_section(request.player_profile)}

{build_news_section(request.news_evidence, request.news_subject)}

Challenge this draft via the structured response tool: identify hidden
risks, question whether an unverified assumption was treated as fact,
surface any conflicting specialist evidence that was glossed over, weigh
in any relevant news above, and produce the single strongest opposing
argument. If you genuinely cannot find a strong opposing argument, say so
and rate your challenge's confidence low rather than manufacturing one.
""".strip()
