"""Abstract base class every specialist agent must extend.

Owns everything that would otherwise be duplicated across agents: turning
an :class:`AgentRequest` into an LLM call, injecting club context and the
data-honesty rules, validating the structured response, and logging.
Subclasses only define who they are and how to phrase their part of the
task.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from config.club_config import ClubConfig
from models.agent_io import AgentRequest, AgentResponse
from prompts.base_prompts import build_base_system_prompt
from services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, llm_client: LLMClient, club_config: ClubConfig) -> None:
        self._llm_client = llm_client
        self._club_config = club_config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent name, e.g. 'Scout Agent'."""

    @property
    @abstractmethod
    def role_description(self) -> str:
        """What this agent evaluates. Used to build its system prompt."""

    @abstractmethod
    def build_user_prompt(self, request: AgentRequest) -> str:
        """Turn a request into the specialist instructions for this agent."""

    def system_prompt(self) -> str:
        return build_base_system_prompt(self._club_config, self.role_description)

    def analyze(self, request: AgentRequest) -> AgentResponse:
        logger.info("%s analyzing request: %s", self.name, request.query)
        response = self._llm_client.generate_structured(
            system_prompt=self.system_prompt(),
            user_prompt=self.build_user_prompt(request),
            response_model=AgentResponse,
        )
        response.agent_name = self.name
        logger.info(
            "%s finished with confidence %.2f", self.name, response.confidence
        )
        return response
