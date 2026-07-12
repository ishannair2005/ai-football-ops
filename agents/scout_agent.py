"""Scout Agent: evaluates player ability, traits, and potential."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from config.club_config import ClubConfig
from models.agent_io import AgentRequest
from services.llm_client import LLMClient
from tools.data_gateway import PlayerDataGateway

ROLE_DESCRIPTION = """
You are the club's chief scout. For any player under discussion, evaluate:
- Technical ability (on-ball skill, finishing, passing range, first touch)
- Physical traits (pace, strength, aerial ability, stamina, injury-relevant build)
- Mentality (composure, work rate, leadership, big-game temperament)
- Strengths and weaknesses relative to their position
- Development potential / ceiling given their age and trajectory
- Comparable players (profile-similar players, for calibration)

When comparing two or more players, evaluate each on the same criteria
before drawing a comparison. When no specific player is named, scout the
squad or position group implied by the question.
""".strip()


class ScoutAgent(BaseAgent):
    def __init__(
        self,
        llm_client: LLMClient,
        club_config: ClubConfig,
        data_gateway: PlayerDataGateway | None = None,
    ) -> None:
        super().__init__(llm_client, club_config)
        self._data_gateway = data_gateway

    @property
    def name(self) -> str:
        return "Scout Agent"

    @property
    def role_description(self) -> str:
        return ROLE_DESCRIPTION

    def build_user_prompt(self, request: AgentRequest) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in request.context.items()) or "(none provided)"
        return f"""
Scouting task: {request.query}

Additional context:
{context_lines}

{self._fetched_data_section(request)}

Provide your scouting assessment via the structured response tool.
""".strip()

    def _fetched_data_section(self, request: AgentRequest) -> str:
        player = request.context.get("player")
        if not player or self._data_gateway is None:
            return (
                "Fetched data: none available for this request — base your assessment "
                "on general knowledge and say so explicitly in your uncertainties."
            )

        evidence = self._data_gateway.fetch_player_evidence(player)
        if not evidence:
            return (
                f"Fetched data: no data-provider record found for '{player}'. "
                "Base your assessment on general knowledge and flag this gap in your uncertainties."
            )

        lines = "\n".join(
            f"- [{item.source}, as of {item.as_of_date}] {item.description}" for item in evidence
        )
        return f"""
Fetched data (ground your assessment in this; if your own knowledge conflicts
with it, say so explicitly rather than silently picking one):
{lines}
""".strip()
