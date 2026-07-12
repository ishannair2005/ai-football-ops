"""Tactical Agent: evaluates system fit, role, and manager compatibility."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.agent_io import AgentRequest, AgentResponse

ROLE_DESCRIPTION = """
You are the club's tactical analyst. For any player, position group, or
system question under discussion, evaluate:
- Tactical fit within the club's current formation and style of play
- Formation compatibility (which system(s) suit this player/role)
- Role clarity (what specific job they would do on the pitch)
- Chemistry with the existing squad's key players and patterns of play
- Pressing fit (trigger recognition, work rate, press resistance)
- Positional flexibility (secondary positions/roles they could cover)
- Compatibility with the current manager's known tactical identity and
  history of player usage

The club's current manager and formation are given in your system prompt —
ground your reasoning in those specifics rather than generic tactical
commentary. When no specific player is named (e.g. "which striker best
fits the system"), evaluate the position group or system question implied
by the query instead.
""".strip()


class TacticalAgent(BaseAgent[AgentRequest, AgentResponse]):
    @property
    def name(self) -> str:
        return "Tactical Agent"

    @property
    def role_description(self) -> str:
        return ROLE_DESCRIPTION

    @property
    def response_model(self) -> type[AgentResponse]:
        return AgentResponse

    def build_user_prompt(self, request: AgentRequest) -> str:
        context_lines = "\n".join(f"- {k}: {v}" for k, v in request.context.items()) or "(none provided)"
        return f"""
Tactical task: {request.query}

Additional context:
{context_lines}

Provide your tactical assessment via the structured response tool.
""".strip()
