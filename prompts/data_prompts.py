"""Shared prompt scaffolding for agents that ground answers in player data.

Extracted from the Scout Agent once a second specialist (Transfer Market
Agent) needed the exact same fetch-and-cite behavior, so it lives in one
place instead of being copy-pasted per agent.
"""

from __future__ import annotations

from tools.data_gateway import PlayerDataGateway
from tools.injury_gateway import InjuryGateway
from tools.news_gateway import NewsGateway


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


def build_injury_section(gateway: InjuryGateway | None, context: dict[str, str]) -> str:
    """Render the "injury data" block for an agent's user prompt.

    Same three-branch shape as :func:`build_player_data_section`: no
    gateway/player named, gateway present but nothing found, or a cited
    record with an as-of date.
    """
    player = context.get("player")
    if not player or gateway is None:
        return (
            "Injury data: none available for this request — do not assume the "
            "player is fully fit or injured; flag this gap in your uncertainties."
        )

    evidence = gateway.fetch_injury_evidence(player)
    if not evidence:
        return (
            f"Injury data: no injury-provider record found for '{player}'. "
            "Do not assume availability; flag this gap in your uncertainties."
        )

    lines = "\n".join(
        f"- [{item.source}, as of {item.as_of_date}] {item.description}" for item in evidence
    )
    return f"""
Injury data (cite this rather than guessing at fitness/availability):
{lines}
""".strip()


def build_news_section(gateway: NewsGateway | None, subject: str | None) -> str:
    """Render the "recent news" block for an agent's user prompt.

    ``subject`` is typically a player name, falling back to a club name for
    player-less queries. Every item is labeled confirmed/rumour/etc. via its
    ``category`` so agents don't treat speculation as settled fact.
    """
    if not subject or gateway is None:
        return (
            "Recent news: none available for this request — do not assume "
            "there is no relevant news; flag this gap in your uncertainties."
        )

    evidence = gateway.fetch_news_evidence(subject)
    if not evidence:
        return f"Recent news: no news-provider items found for '{subject}'."

    lines = "\n".join(
        f"- [{item.source}, as of {item.as_of_date}] {item.description}" for item in evidence
    )
    return f"""
Recent news on {subject} (distinguish confirmed reporting from rumour when
you use this):
{lines}
""".strip()
