"""Shared prompt scaffolding for agents that ground answers in player data.

These section-builders read pre-fetched evidence off a
:class:`~models.agent_io.PlayerProfile` rather than calling a gateway
themselves — the Manager resolves identity and fetches stats/injury/news
exactly once and threads the same profile to every agent, so no
specialist independently infers or re-fetches player information.
"""

from __future__ import annotations

from models.agent_io import DataQualityStatus, Evidence, PlayerProfile


def _evidence_lines(evidence: list[Evidence]) -> str:
    return "\n".join(
        f"- [{item.source}, as of {item.as_of_date}] {item.description}" for item in evidence
    )


def build_data_quality_section(profile: PlayerProfile | None) -> str:
    """Render the Data Quality summary once — the per-domain sections
    below stay terse about *why* a domain is thin precisely because this
    is where that's stated in full, so it isn't repeated per specialist."""
    if profile is None or not profile.data_quality:
        return "Data Quality: not applicable — no player named in this request."
    rows = "\n".join(f"- {entry.domain}: {entry.status.value}" for entry in profile.data_quality)
    return f"""
Data Quality summary (this is the authoritative account of what's known
about {profile.queried_name} — do not restate these gaps yourself in
evidence_gaps; only add gaps specific to your own analysis):
{rows}
""".strip()


def _unavailable_hint(profile: PlayerProfile | None, domain: str) -> str:
    """A short pointer back at the shared Data Quality summary instead of
    a full explanation repeated in every section that's empty."""
    status = next((e.status for e in (profile.data_quality if profile else []) if e.domain == domain), None)
    if status == DataQualityStatus.PROVIDER_ERROR:
        return "a live data provider errored — see Data Quality summary above"
    return "see Data Quality summary above"


def build_identity_section(profile: PlayerProfile | None) -> str:
    """Render what the Player Resolver established about identity."""
    if profile is None:
        return (
            "Player identity: no player named in this request — do not assume one; "
            "evaluate the position group or system question implied by the query instead."
        )
    if not profile.resolved:
        return (
            f"Player identity: '{profile.queried_name}' could not be verified "
            f"({_unavailable_hint(profile, 'Identity')}). Do not assume a club, league, "
            "or identity for this name; reflect this in a low confidence score."
        )
    details = ", ".join(
        f"{label}: {value}"
        for label, value in (
            ("position", profile.position),
            ("nationality", profile.nationality),
            ("age", profile.age),
            ("competition", profile.competition),
        )
        if value is not None
    )
    return (
        f"Player identity (verified as of {profile.identity_as_of}, source "
        f"{profile.identity_source}): {profile.full_name}, current club "
        f"{profile.club or 'unknown — treat as an evidence gap'}"
        f"{f' ({details})' if details else ''}."
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
            "season figures."
        )

    if not profile.stats_evidence:
        return (
            f"Verified statistics: no data-provider record found for "
            f"'{profile.full_name}' ({_unavailable_hint(profile, 'Statistics')}). "
            "Do not estimate season figures from general knowledge."
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
            "player is fully fit or injured."
        )

    if not profile.injury_evidence:
        return (
            f"Injury data: no injury-provider record found for '{profile.full_name}' "
            f"({_unavailable_hint(profile, 'Injuries')}). Do not assume availability."
        )

    return f"""
Injury data (cite this rather than guessing at fitness/availability):
{_evidence_lines(profile.injury_evidence)}
""".strip()


def build_transfer_section(profile: PlayerProfile | None) -> str:
    """Render the "verified facts" block for transfer fee, wages, release
    clause, and contract expiry."""
    if profile is None or not profile.resolved:
        return (
            "Transfer/contract data: none available for this request — do not assume "
            "a fee, wage, or contract length."
        )

    if not profile.transfer_evidence:
        return (
            f"Transfer/contract data: no transfer-provider record found for "
            f"'{profile.full_name}' ({_unavailable_hint(profile, 'Transfer')}). Do not "
            "estimate a fee, wage, release clause, or contract expiry from general knowledge."
        )

    return f"""
Transfer/contract data (cite this rather than guessing at a fee, wage, or
contract length; any field not listed below is genuinely unknown, not zero):
{_evidence_lines(profile.transfer_evidence)}
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
