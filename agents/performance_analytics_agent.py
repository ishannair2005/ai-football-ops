"""Performance Analytics Agent: evaluates advanced on-pitch output metrics."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.agent_io import AgentRequest, AgentResponse
from prompts.data_prompts import build_data_quality_section, build_identity_section, build_player_data_section

ROLE_DESCRIPTION = """
You are the club's performance analyst. For any player under discussion,
evaluate using the newest available statistics:
- Season output (appearances, minutes, goals, assists, cards)
- Passing accuracy and defensive actions (tackles, interceptions) when
  the verified statistics below include them
- Expected goals (xG), expected assists (xA), non-penalty xG (NPxG)
- Shot-creating actions and goal-creating actions
- Progressive passes and progressive carries
- Touches, pressures, and expected threat

Ground the metrics above in the verified statistics section when it
includes them. Anything the verified data doesn't cover (this project's
configured providers do not currently supply xG/xA/NPxG, shot/goal-
creating actions, progressive actions, pressures, or expected threat) is
genuinely unavailable — treat any such figure as a general-knowledge
estimate, mark it with reduced confidence, and state this explicitly as
an evidence gap rather than presenting a precise-sounding number as fact.
""".strip()


class PerformanceAnalyticsAgent(BaseAgent[AgentRequest, AgentResponse]):
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
        profile = request.player_profile
        return f"""
Performance analytics task: {request.query}

Additional context:
{context_lines}

{build_data_quality_section(profile)}

{build_identity_section(profile)}

{build_player_data_section(profile)}

Provide your performance analytics assessment via the structured response tool.
""".strip()
