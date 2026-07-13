"""General Manager Agent: coordinates specialists, never analyzes itself.

Resolves player identity exactly once (if a player is named), fetches
stats/injury evidence for the resolved canonical name, and threads the
same verified profile to every specialist and the Devil's Advocate — no
agent independently infers or re-fetches player information. Then runs
every registered specialist agent, drafts a synthesis, and — when a
Devil's Advocate is configured — puts that draft through one round of
challenge and resolution before finalizing it. The Manager never performs
specialist analysis itself; it only resolves, weighs, reconciles, and
(when challenged) revises.
"""

from __future__ import annotations

import logging
from typing import Callable

from agents.base_agent import BaseAgent
from agents.devils_advocate_agent import DevilsAdvocateAgent
from config.club_config import ClubConfig
from models.agent_io import (
    AgentRequest,
    AgentResponse,
    ChallengeRequest,
    FinalRecommendation,
    ManagerSynthesis,
    PlayerNameExtraction,
    PlayerProfile,
)
from prompts.manager_prompts import (
    PLAYER_EXTRACTION_SYSTEM_PROMPT,
    build_manager_system_prompt,
    build_manager_user_prompt,
    build_player_extraction_user_prompt,
    build_resolution_user_prompt,
)
from services.llm_client import LLMClient
from tools.data_gateway import PlayerDataGateway
from tools.injury_gateway import InjuryGateway
from tools.news_gateway import NewsGateway
from tools.player_identity_gateway import PlayerIdentityGateway

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]
AgentResponseCallback = Callable[[AgentResponse], None]


def _notify(callback: StatusCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


class GeneralManagerAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        club_config: ClubConfig,
        specialist_agents: list[BaseAgent],
        devils_advocate: DevilsAdvocateAgent | None = None,
        identity_gateway: PlayerIdentityGateway | None = None,
        data_gateway: PlayerDataGateway | None = None,
        injury_gateway: InjuryGateway | None = None,
        news_gateway: NewsGateway | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._club_config = club_config
        self._specialist_agents = specialist_agents
        self._devils_advocate = devils_advocate
        self._identity_gateway = identity_gateway
        self._data_gateway = data_gateway
        self._injury_gateway = injury_gateway
        self._news_gateway = news_gateway

    def handle_query(
        self,
        request: AgentRequest,
        on_status: StatusCallback | None = None,
        on_agent_response: AgentResponseCallback | None = None,
    ) -> FinalRecommendation:
        profile = self._resolve_player_profile(request, on_status)
        request = request.model_copy(update={"player_profile": profile})

        specialist_responses = self._consult_specialists(request, on_status, on_agent_response)

        _notify(on_status, "Drafting initial recommendation...")
        draft = self._synthesize(request, specialist_responses)

        if self._devils_advocate is None:
            return FinalRecommendation.from_synthesis(draft, specialist_responses)

        _notify(on_status, "Devil's Advocate is challenging the recommendation...")
        challenge = self._challenge(request, draft, specialist_responses, profile)
        if on_agent_response is not None:
            on_agent_response(challenge)

        _notify(on_status, "Finalizing the recommendation...")
        final = self._resolve(request, specialist_responses, draft, challenge)
        return FinalRecommendation.from_synthesis(final, specialist_responses, challenge)

    def _extract_player_name(
        self, request: AgentRequest, on_status: StatusCallback | None = None
    ) -> str | None:
        """A player name supplied out-of-band (e.g. a UI field) always wins
        and costs no extra call. Otherwise, parse the query itself — the
        resolver must work off what the user actually asked, not require
        them to duplicate the name into a separate field."""
        explicit = request.context.get("player")
        if explicit:
            return explicit

        _notify(on_status, "Checking whether the query names a specific player...")
        extraction: PlayerNameExtraction = self._llm_client.generate_structured(
            system_prompt=PLAYER_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=build_player_extraction_user_prompt(request.query),
            response_model=PlayerNameExtraction,
            max_tokens=256,
        )
        return extraction.player_name

    def _resolve_player_profile(
        self, request: AgentRequest, on_status: StatusCallback | None = None
    ) -> PlayerProfile | None:
        """Resolve identity and fetch stats/injury evidence exactly once for
        the query's player, if any. Every step that comes back empty is
        recorded as an explicit evidence gap rather than left silent."""
        queried_name = self._extract_player_name(request, on_status)
        if not queried_name:
            return None

        _notify(on_status, "Resolving player identity...")

        identity = self._identity_gateway.resolve(queried_name) if self._identity_gateway else None
        gaps: list[str] = []

        if identity is None:
            gaps.append(
                f"Player identity for '{queried_name}' could not be verified from "
                "configured data providers."
            )
            canonical_name = queried_name
        else:
            canonical_name = identity.full_name

        stats_evidence = (
            self._data_gateway.fetch_player_evidence(canonical_name) if self._data_gateway else []
        )
        if not stats_evidence:
            gaps.append("Current statistics unavailable from configured data providers.")

        injury_evidence = (
            self._injury_gateway.fetch_injury_evidence(canonical_name)
            if self._injury_gateway
            else []
        )
        if not injury_evidence:
            gaps.append("Current injury/availability status unavailable from configured data providers.")

        return PlayerProfile(
            queried_name=queried_name,
            resolved=identity is not None,
            full_name=identity.full_name if identity else None,
            club=identity.club if identity else None,
            player_ids=identity.player_ids if identity else {},
            identity_as_of=identity.as_of_date if identity else None,
            identity_source=identity.source if identity else None,
            stats_evidence=stats_evidence,
            injury_evidence=injury_evidence,
            evidence_gaps=gaps,
        )

    def _consult_specialists(
        self,
        request: AgentRequest,
        on_status: StatusCallback | None = None,
        on_agent_response: AgentResponseCallback | None = None,
    ) -> list[AgentResponse]:
        responses: list[AgentResponse] = []
        for agent in self._specialist_agents:
            logger.info("General Manager delegating to %s", agent.name)
            _notify(on_status, f"{agent.name} is analyzing...")
            response = agent.analyze(request)
            responses.append(response)
            if on_agent_response is not None:
                on_agent_response(response)
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
        profile: PlayerProfile | None,
    ) -> AgentResponse:
        logger.info("General Manager requesting a Devil's Advocate challenge")
        subject = (profile.full_name if profile and profile.resolved else None) or request.club_id
        news_evidence = (
            self._news_gateway.fetch_news_evidence(subject) if self._news_gateway else []
        )
        challenge_request = ChallengeRequest(
            original_query=request.query,
            club_id=request.club_id,
            draft_recommendation=draft,
            specialist_responses=specialist_responses,
            player_profile=profile,
            news_evidence=news_evidence,
            news_subject=subject,
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
