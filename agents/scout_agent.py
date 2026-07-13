"""Scout Agent: evaluates player ability, traits, and potential."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.agent_io import AgentRequest, AgentResponse
from prompts.data_prompts import (
    build_data_quality_section,
    build_identity_section,
    build_injury_section,
    build_player_data_section,
)

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


class ScoutAgent(BaseAgent[AgentRequest, AgentResponse]):
    @property
    def name(self) -> str:
        return "Scout Agent"

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
Scouting task: {request.query}

Additional context:
{context_lines}

{build_data_quality_section(profile)}

{build_identity_section(profile)}

{build_player_data_section(profile)}

{build_injury_section(profile)}

Provide your scouting assessment via the structured response tool.
""".strip()
