"""Shared prompt scaffolding for agents that ground answers in player data.

Extracted from the Scout Agent once a second specialist (Transfer Market
Agent) needed the exact same fetch-and-cite behavior, so it lives in one
place instead of being copy-pasted per agent.
"""

from __future__ import annotations

from tools.data_gateway import PlayerDataGateway


def build_player_data_section(
    gateway: PlayerDataGateway | None, context: dict[str, str]
) -> str:
    """Render the "fetched data" block for an agent's user prompt.

    Looks up ``context["player"]`` via ``gateway`` and either cites the
    resulting evidence (with as-of dates) or, when no gateway/player/record
    is available, produces an explicit disclosure instructing the agent to
    fall back to general knowledge and flag the gap — never to fabricate.
    """
    player = context.get("player")
    if not player or gateway is None:
        return (
            "Fetched data: none available for this request — base your assessment "
            "on general knowledge and say so explicitly in your uncertainties."
        )

    evidence = gateway.fetch_player_evidence(player)
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
