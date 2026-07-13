"""Streamlit UI for the AI Football Operations Platform.

Wired to the real FootballOperationsPlatform (not mocked). Presentation is
intentionally minimal — the project's emphasis is the agent architecture,
not the UI — but every section the platform is required to surface
(executive summary, verdict, confidence, each specialist report, the
Devil's Advocate challenge, the Manager's final decision, sources, and
data freshness) is rendered explicitly below. A query makes up to 8 real,
sequential LLM calls, so the loading state surfaces live per-step status
and reveals each specialist's card as soon as it's ready rather than
leaving the page blank for the full 60-90+ seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

# On Streamlit Cloud, secrets live in st.secrets and aren't guaranteed to be
# mirrored into the process environment. config.settings reads its values
# via pydantic-settings, which only looks at os.environ/.env — so copy any
# configured secrets across before anything imports/calls get_settings().
# Locally (no secrets.toml configured, using .env instead) st.secrets is
# simply empty and this is a no-op.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from config.settings import get_settings
from models.agent_io import (
    AgentRequest,
    AgentResponse,
    ComparisonOutcome,
    ComparisonRecommendation,
    DataQualityStatus,
    PlatformResult,
    RecommendationVerdict,
)
from services.logging_config import configure_logging
from services.llm_client import LLMClientError
from services.platform_factory import build_platform

configure_logging()

st.set_page_config(page_title="AI Football Operations Platform", layout="wide")

settings = get_settings()

VERDICT_DISPLAY = {
    RecommendationVerdict.BUY: st.success,
    RecommendationVerdict.MONITOR: st.warning,
    RecommendationVerdict.DO_NOT_SIGN: st.error,
}

SPECIALIST_LABELS = {
    "Scout Agent": "Scout Report",
    "Performance Analytics Agent": "Performance Analytics",
    "Tactical Agent": "Tactical Fit",
    "Transfer Market Agent": "Transfer Cost Analysis",
    "Devil's Advocate": "Devil's Advocate Challenge",
}

SPECIALIST_ICONS = {
    "Scout Agent": "🔍",
    "Tactical Agent": "🎯",
    "Transfer Market Agent": "💰",
    "Performance Analytics Agent": "📊",
    "Devil's Advocate": "🥊",
}

DATA_QUALITY_ICONS = {
    DataQualityStatus.AVAILABLE: "🟢",
    DataQualityStatus.OUTDATED: "🟡",
    DataQualityStatus.UNAVAILABLE: "⚪",
    DataQualityStatus.PROVIDER_ERROR: "🔴",
}


def render_data_quality(entries: list) -> None:
    """Shown once, near the top of a report, so specialists below don't
    each have to restate the same "X is unavailable" disclosure."""
    st.subheader("Data Quality")
    if not entries:
        st.caption("No player named in this query — data quality summary not applicable.")
        return
    available = sum(1 for e in entries if e.status == DataQualityStatus.AVAILABLE)
    rows = [
        {"Domain": e.domain, "Status": f"{DATA_QUALITY_ICONS.get(e.status, '')} {e.status.value}"}
        for e in entries
    ]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    with col2:
        st.metric("Overall quality", f"{available}/{len(entries)} available")

st.title("AI Football Operations Platform")
st.caption(f"Active club: {settings.active_club.replace('_', ' ').title()}")

query = st.text_area(
    "Ask the football operations department a question",
    placeholder="Should Manchester United sign Sample Striker?",
    height=100,
)

player_name = st.text_input(
    "Player name (optional override)",
    placeholder="e.g. Sample Striker",
    help="The platform automatically detects a named player in your question "
    "above, so this is only needed to override that (e.g. a nickname it might "
    "not resolve). Try asking about 'Sample Striker' directly in the question box.",
)

submitted = st.button("Get Recommendation", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        context = {"player": player_name.strip()} if player_name.strip() else {}
        request = AgentRequest(query=query, club_id=settings.active_club, context=context)

        status = st.status("Starting analysis...", expanded=True)
        cards = st.container()

        def on_status(message: str) -> None:
            status.update(label=message)
            status.write(message)

        def on_agent_response(response: AgentResponse) -> None:
            # In comparison mode, agent_name is suffixed with the candidate's
            # name (e.g. "Scout Agent (Sample Striker)") for disambiguation —
            # strip it before looking up the label/icon, then keep it in the
            # displayed header.
            base_name, _, suffix = response.agent_name.partition(" (")
            suffix = f" ({suffix}" if suffix else ""
            label = SPECIALIST_LABELS.get(base_name, base_name)
            icon = SPECIALIST_ICONS.get(base_name, "🧠")
            with cards.expander(f"{icon} {label}{suffix} (confidence: {response.confidence:.0%})"):
                st.write(response.summary)
                if response.verified_facts:
                    st.markdown("**Verified Facts**")
                    for ev in response.verified_facts:
                        st.write(f"- {ev.description} (source: {ev.source}, as of {ev.as_of_date or 'unknown'})")
                if response.evidence_gaps:
                    st.markdown("**Evidence Gaps**")
                    for gap in response.evidence_gaps:
                        st.write(f"- {gap}")

        try:
            platform = build_platform(settings.active_club)
            result = platform.handle_query(
                request, on_status=on_status, on_agent_response=on_agent_response
            )
        except LLMClientError as exc:
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(str(exc))
        else:
            status.update(label="Analysis complete", state="complete", expanded=False)

            if isinstance(result, ComparisonRecommendation):
                comparison = result

                st.header("Executive Summary")
                st.write(comparison.executive_summary)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Comparison Outcome")
                    if comparison.outcome == ComparisonOutcome.CLEAR_PREFERENCE and comparison.preferred_player:
                        st.success(f"Preferred signing: {comparison.preferred_player}")
                    elif comparison.outcome == ComparisonOutcome.MULTIPLE_VIABLE:
                        st.warning("Multiple candidates are viable — no single standout.")
                    else:
                        st.error("None of the candidates are recommended.")
                with col2:
                    st.metric("Confidence", f"{comparison.confidence:.0%}")

                st.subheader("Rationale")
                st.write(comparison.verdict_rationale)

                if comparison.key_differentiators:
                    st.markdown("**Key differentiators**")
                    for item in comparison.key_differentiators:
                        st.write(f"- {item}")

                st.subheader("Decision Matrix")
                matrix_rows = []
                for criterion in comparison.decision_matrix:
                    row = {"Criterion": criterion.criterion_name}
                    for rating in criterion.ratings:
                        row[rating.player_name] = f"{rating.tier.value} ({rating.confidence:.0%})"
                    matrix_rows.append(row)
                if matrix_rows:
                    st.dataframe(matrix_rows, use_container_width=True, hide_index=True)
                    with st.expander("Full reasoning behind each matrix cell"):
                        for criterion in comparison.decision_matrix:
                            st.markdown(f"**{criterion.criterion_name}**")
                            for rating in criterion.ratings:
                                st.write(f"- *{rating.player_name}* ({rating.tier.value}): {rating.summary}")

                st.caption(
                    "Each candidate's full specialist reports appeared above as they "
                    "completed during analysis."
                )
            else:
                rec = result.recommendation
                report = result.report

                st.header("Executive Summary")
                st.write(report.executive_summary)

                render_data_quality(report.data_quality)

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Overall Recommendation")
                    VERDICT_DISPLAY.get(rec.verdict, st.info)(rec.verdict.value)
                with col2:
                    st.metric("Confidence", f"{rec.confidence:.0%}")

                if rec.devils_advocate_challenge:
                    st.subheader("Devil's Advocate Challenge")
                    st.warning(rec.devils_advocate_challenge.summary)

                st.subheader("Manager's Final Decision")
                st.write(rec.recommendation)
                if rec.challenge_resolution:
                    st.markdown("**How the Devil's Advocate challenge was resolved**")
                    st.write(rec.challenge_resolution)

                if rec.key_risks:
                    st.markdown("**Key risks**")
                    for risk in rec.key_risks:
                        st.write(f"- {risk}")

                if rec.next_steps:
                    st.markdown("**Next steps**")
                    for step in rec.next_steps:
                        st.write(f"- {step}")

                st.subheader("Sources Used")
                if report.sources_used:
                    for source in report.sources_used:
                        st.write(f"- {source}")
                else:
                    st.caption("No data-provider sources were cited for this query.")

                st.subheader("Data Freshness")
                if report.data_freshness:
                    for domain, as_of in sorted(report.data_freshness.items()):
                        st.write(f"- **{domain}:** {as_of}")
                else:
                    st.caption("No dated data-provider evidence was available for this query.")

                with st.expander("Full Narrative Report"):
                    st.write(report.narrative)
