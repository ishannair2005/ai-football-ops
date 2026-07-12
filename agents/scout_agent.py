"""Scout Agent: evaluates player ability, traits, and potential."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.agent_io import AgentRequest

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

Provide your scouting assessment via the structured response tool.
""".strip()
