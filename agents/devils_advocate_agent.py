"""Devil's Advocate Agent: argues the strongest case against a draft recommendation."""

from __future__ import annotations

import json

from agents.base_agent import BaseAgent
from config.club_config import ClubConfig
from models.agent_io import AgentResponse, ChallengeRequest
from prompts.data_prompts import build_news_section
from services.llm_client import LLMClient
from tools.news_gateway import NewsGateway

ROLE_DESCRIPTION = """
You are the club's Devil's Advocate. You are given the General Manager's
draft recommendation and the specialist findings underpinning it. Your
job is to argue the strongest possible case AGAINST the current
recommendation:
- Identify hidden risks the draft may have underweighted or ignored
- Question the assumptions the specialists (and the draft) relied on
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
    def __init__(
        self,
        llm_client: LLMClient,
        club_config: ClubConfig,
        news_gateway: NewsGateway | None = None,
    ) -> None:
        super().__init__(llm_client, club_config)
        self._news_gateway = news_gateway

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
                "assumptions": r.assumptions,
                "uncertainties": r.uncertainties,
            }
            for r in request.specialist_responses
        ]
        subject = request.player or request.club_id
        return f"""
Original question: {request.original_query}

Draft recommendation from the General Manager (JSON):
{json.dumps(request.draft_recommendation.model_dump(mode="json"), indent=2)}

Specialist findings underpinning that draft (JSON):
{json.dumps(specialist_findings, indent=2)}

{build_news_section(self._news_gateway, subject)}

Challenge this draft via the structured response tool: identify hidden
risks, question assumptions, surface any conflicting specialist evidence
that was glossed over, weigh in any relevant news above, and produce the
single strongest opposing argument. If you genuinely cannot find a strong
opposing argument, say so and rate your challenge's confidence low rather
than manufacturing one.
""".strip()
