"""Performance Analytics Agent: evaluates advanced on-pitch output metrics."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from config.club_config import ClubConfig
from models.agent_io import AgentRequest, AgentResponse
from prompts.data_prompts import build_player_data_section
from services.llm_client import LLMClient
from tools.data_gateway import PlayerDataGateway

ROLE_DESCRIPTION = """
You are the club's performance analyst. For any player under discussion,
evaluate using the newest available statistics:
- Expected goals (xG), expected assists (xA), non-penalty xG (NPxG)
- Shot-creating actions and goal-creating actions
- Progressive passes and progressive carries
- Touches, pressures, and defensive actions
- Possession and passing metrics, ball recoveries
- Expected threat

You have basic season figures (appearances, goals, assists) from the
data-provider evidence below when available. You do NOT have access to a
live provider for the advanced tracking metrics above (xG, xA, NPxG,
shot/goal-creating actions, progressive actions, pressures, expected
threat) — treat any such figures as general knowledge estimates, mark
them with reduced confidence, and flag this gap explicitly in your
uncertainties rather than presenting precise-sounding numbers as fact.
""".strip()


class PerformanceAnalyticsAgent(BaseAgent[AgentRequest, AgentResponse]):
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
        return "Performance Analytics Agent"

    @property
    def role_description(self) -> str:
        return ROLE_DESCRIPTION

    @property
    def response_model(self) -> type[AgentResponse]:
        return AgentResponse

    def build_user_prompt(self, request: AgentRequest) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in request.context.items()) or "(none provided)"
        return f"""
Performance analytics task: {request.query}

Additional context:
{context_lines}

{build_player_data_section(self._data_gateway, request.context)}

Provide your performance analytics assessment via the structured response tool.
""".strip()
