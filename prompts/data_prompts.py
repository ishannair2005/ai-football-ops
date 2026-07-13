"""Shared prompt scaffolding for agents that ground answers in player data.

These section-builders read pre-fetched evidence off a
:class:`~models.agent_io.PlayerProfile` rather than calling a gateway
themselves — the Manager resolves identity and fetches stats/injury/news
exactly once and threads the same profile to every agent, so no
specialist independently infers or re-fetches player information.
"""

from __future__ import annotations

from models.agent_io import Evidence, PlayerProfile


def _evidence_lines(evidence: list[Evidence]) -> str:
    return "\n".join(
        f"- [{item.source}, as of {item.as_of_date}] {item.description}" for item in evidence
    )


def build_identity_section(profile: PlayerProfile | None) -> str:
    """Render what the Player Resolver established about identity."""
    if profile is None:
        return (
            "Player identity: no player named in this request — do not assume one; "
            "evaluate the position group or system question implied by the query instead."
        )
    if not profile.resolved:
        return (
            f"Player identity: '{profile.queried_name}' could not be verified from "
            "configured data providers. Do not assume a club, league, or identity for "
            "this name — state this gap explicitly in evidence_gaps and reflect it in "
            "a low confidence score."
        )
    return (
        f"Player identity (verified as of {profile.identity_as_of}, source "
        f"{profile.identity_source}): {profile.full_name}, current club "
        f"{profile.club or 'unknown — treat as an evidence gap'}."
    )


def build_player_data_section(profile: PlayerProfile | None) -> str:
    """Render the "verified facts" block for statistics.

    Same honesty shape throughout: no profile / unresolved identity / a
    resolved player with no stats on file, all produce an explicit
    disclosure instructing the agent to state the gap in evidence_gaps and
    lower confidence — never to fabricate or assume.
    """
    if profile is None or not profile.resolved:
        return (
            "Verified statistics: none available for this request — do not assume "
            "season figures; state this as an evidence gap and reflect it in confidence."
        )

    if not profile.stats_evidence:
        return (
            f"Verified statistics: no data-provider record found for "
            f"'{profile.full_name}'. State this as an evidence gap rather than "
            "estimating from general knowledge."
        )

    return f"""
Verified statistics (ground your assessment in this; if your own knowledge
conflicts with it, say so explicitly rather than silently picking one):
{_evidence_lines(profile.stats_evidence)}
""".strip()


def build_injury_section(profile: PlayerProfile | None) -> str:
    """Render the "verified facts" block for injury/availability."""
    if profile is None or not profile.resolved:
        return (
            "Injury data: none available for this request — do not assume the "
            "player is fully fit or injured; state this as an evidence gap."
        )

    if not profile.injury_evidence:
        return (
            f"Injury data: no injury-provider record found for '{profile.full_name}'. "
            "Do not assume availability; state this as an evidence gap."
        )

    return f"""
Injury data (cite this rather than guessing at fitness/availability):
{_evidence_lines(profile.injury_evidence)}
""".strip()


def build_news_section(evidence: list[Evidence], subject: str | None) -> str:
    """Render the "recent news" block for an agent's user prompt.

    ``subject`` is a display label only (player or club name) — the
    evidence itself is fetched once by the Manager. Every item is labeled
    confirmed/rumour/etc. via its category so agents don't treat
    speculation as settled fact.
    """
    if not subject:
        return "Recent news: none available for this request."

    if not evidence:
        return f"Recent news: no news-provider items found for '{subject}'."

    return f"""
Recent news on {subject} (distinguish confirmed reporting from rumour when
you use this):
{_evidence_lines(evidence)}
""".strip()
