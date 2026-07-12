"""General Manager Agent: coordinates specialists, never analyzes itself.

Runs every registered specialist agent against the request, drafts a
synthesis, and — when a Devil's Advocate is configured — puts that draft
through one round of challenge and resolution before finalizing it. The
Manager never performs specialist analysis itself; it only weighs,
reconciles, and (when challenged) revises.
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from agents.devils_advocate_agent import DevilsAdvocateAgent
from config.club_config import ClubConfig
from models.agent_io import (
    AgentRequest,
    AgentResponse,
    ChallengeRequest,
    FinalRecommendation,
    ManagerSynthesis,
)
from prompts.manager_prompts import (
    build_manager_system_prompt,
    build_manager_user_prompt,
    build_resolution_user_prompt,
)
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class GeneralManagerAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        club_config: ClubConfig,
        specialist_agents: list[BaseAgent],
        devils_advocate: DevilsAdvocateAgent | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._club_config = club_config
        self._specialist_agents = specialist_agents
        self._devils_advocate = devils_advocate

    def handle_query(self, request: AgentRequest) -> FinalRecommendation:
        specialist_responses = self._consult_specialists(request)
        draft = self._synthesize(request, specialist_responses)

        if self._devils_advocate is None:
            return FinalRecommendation.from_synthesis(draft, specialist_responses)

        challenge = self._challenge(request, draft, specialist_responses)
        final = self._resolve(request, specialist_responses, draft, challenge)
        return FinalRecommendation.from_synthesis(final, specialist_responses, challenge)

    def _consult_specialists(self, request: AgentRequest) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        for agent in self._specialist_agents:
            logger.info("General Manager delegating to %s", agent.name)
            responses.append(agent.analyze(request))
        return responses

    def _synthesize(
        self, request: AgentRequest, specialist_responses: list[AgentResponse]
    ) -> ManagerSynthesis:
        return self._llm_client.generate_structured(
            system_prompt=build_manager_system_prompt(self._club_config),
            user_prompt=build_manager_user_prompt(request, specialist_responses),
            response_model=ManagerSynthesis,
        )

    def _challenge(
        self,
        request: AgentRequest,
        draft: ManagerSynthesis,
        specialist_responses: list[AgentResponse],
    ) -> AgentResponse:
        logger.info("General Manager requesting a Devil's Advocate challenge")
        challenge_request = ChallengeRequest(
            original_query=request.query,
            club_id=request.club_id,
            draft_recommendation=draft,
            specialist_responses=specialist_responses,
            player=request.context.get("player"),
        )
        return self._devils_advocate.analyze(challenge_request)

    def _resolve(
        self,
        request: AgentRequest,
        specialist_responses: list[AgentResponse],
        draft: ManagerSynthesis,
        challenge: AgentResponse,
    ) -> ManagerSynthesis:
        return self._llm_client.generate_structured(
            system_prompt=build_manager_system_prompt(self._club_config),
            user_prompt=build_resolution_user_prompt(
                request, specialist_responses, draft, challenge
            ),
            response_model=ManagerSynthesis,
        )
