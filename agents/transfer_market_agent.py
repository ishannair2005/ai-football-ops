"""Transfer Market Agent: evaluates fee, wages, resale value, and financial fit."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from models.agent_io import AgentRequest, AgentResponse
from prompts.data_prompts import (
    build_data_quality_section,
    build_identity_section,
    build_player_data_section,
    build_transfer_section,
)

ROLE_DESCRIPTION = """
You are the club's transfer market analyst. For any player under
discussion, evaluate:
- Transfer fee (verified figure when available, otherwise a clearly
  labeled estimate)
- Wage expectations relative to the club's likely pay structure
- Resale value / age curve (how their market value is likely to trend)
- Contract situation (time remaining, leverage this creates for either side)
- Financial efficiency (value delivered per pound of fee + wages, relative
  to alternatives at the same position)

A transfer/contract data section is provided below when a player is
named. Ground your fee, contract-expiry, and (when present) wage figures
in it rather than general knowledge; any field it doesn't list (most
often wages and release clauses) is genuinely unknown, not zero — mark
those as evidence gaps and reflect them with reduced confidence rather
than inventing a plausible-sounding figure.
""".strip()


class TransferMarketAgent(BaseAgent[AgentRequest, AgentResponse]):
    @property
    def name(self) -> str:
        return "Transfer Market Agent"

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
Transfer market task: {request.query}

Additional context:
{context_lines}

{self._club_budget_section()}

{build_data_quality_section(profile)}

{build_identity_section(profile)}

{build_player_data_section(profile)}

{build_transfer_section(profile)}

Provide your transfer market assessment via the structured response tool.
""".strip()

    def _club_budget_section(self) -> str:
        budget = self._club_config.transfer_budget_gbp
        wages = self._club_config.wage_budget_gbp_per_week
        if budget is None and wages is None:
            return (
                "Club budget: no transfer or wage budget figures are configured for "
                f"{self._club_config.name}. Do not assume a specific budget — state this "
                "as an evidence gap rather than inventing a figure."
            )
        return (
            f"Club budget: transfer budget £{budget or 'unset'}, "
            f"wage budget £{wages or 'unset'} per week."
        )
