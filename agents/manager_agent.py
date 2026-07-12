"""General Manager Agent: coordinates specialists, never analyzes itself.

Phase 1 scope: run every registered specialist agent against the request
and synthesize their findings into one recommendation. Later phases add
delegation logic (choosing which specialists to consult), multi-round
challenge/response between agents, and the Devil's Advocate pass.
"""

from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from config.club_config import ClubConfig
from models.agent_io import AgentRequest, AgentResponse, FinalRecommendation, ManagerSynthesis
from prompts.manager_prompts import build_manager_system_prompt, build_manager_user_prompt
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class GeneralManagerAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        club_config: ClubConfig,
        specialist_agents: list[BaseAgent],
    ) -> None:
        self._llm_client = llm_client
        self._club_config = club_config
        self._specialist_agents = specialist_agents

    def handle_query(self, request: AgentRequest) -> FinalRecommendation:
        specialist_responses = self._consult_specialists(request)
        synthesis = self._synthesize(request, specialist_responses)
        return FinalRecommendation.from_synthesis(synthesis, specialist_responses)

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
