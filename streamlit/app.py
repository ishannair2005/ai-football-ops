"""Streamlit UI for the AI Football Operations Platform.

Wired to the real FootballOperationsPlatform (not mocked). Presentation is
intentionally minimal — the project's emphasis is the agent architecture,
not the UI — but every section the platform is required to surface
(executive summary, verdict, confidence, each specialist report, the
Devil's Advocate challenge, the Manager's final decision, sources, and
data freshness) is rendered explicitly below.
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
from models.agent_io import AgentRequest, PlatformResult, RecommendationVerdict
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
}

st.title("AI Football Operations Platform")
st.caption(f"Active club: {settings.active_club.replace('_', ' ').title()}")

query = st.text_area(
    "Ask the football operations department a question",
    placeholder="Should Manchester United sign Sample Striker?",
    height=100,
)

player_name = st.text_input(
    "Player name (optional)",
    placeholder="e.g. Sample Striker",
    help="If the question is about a specific player, name them here so "
    "specialists can look up data-provider records instead of relying on "
    "general knowledge alone. Try 'Sample Striker' against the bundled demo data.",
)

submitted = st.button("Get Recommendation", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        try:
            context = {"player": player_name.strip()} if player_name.strip() else {}
            with st.spinner("Consulting specialist agents..."):
                platform = build_platform(settings.active_club)
                result: PlatformResult = platform.handle_query(
                    AgentRequest(query=query, club_id=settings.active_club, context=context)
                )
        except LLMClientError as exc:
            st.error(str(exc))
        else:
            rec = result.recommendation
            report = result.report

            st.header("Executive Summary")
            st.write(report.executive_summary)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Overall Recommendation")
                VERDICT_DISPLAY.get(rec.verdict, st.info)(rec.verdict.value)
            with col2:
                st.metric("Confidence", f"{rec.confidence:.0%}")

            st.subheader("Specialist Reports")
            for response in rec.agent_responses:
                label = SPECIALIST_LABELS.get(response.agent_name, response.agent_name)
                with st.expander(f"{label} (confidence: {response.confidence:.0%})"):
                    st.write(response.summary)
                    if response.supporting_evidence:
                        st.markdown("**Supporting evidence**")
                        for ev in response.supporting_evidence:
                            st.write(f"- [{ev.source}] {ev.description} — {ev.value or 'n/a'}")
                    if response.assumptions:
                        st.markdown("**Assumptions**")
                        for a in response.assumptions:
                            st.write(f"- {a}")
                    if response.uncertainties:
                        st.markdown("**Uncertainties**")
                        for u in response.uncertainties:
                            st.write(f"- {u}")

            if rec.devils_advocate_challenge:
                st.subheader("Devil's Advocate Challenge")
                st.warning(rec.devils_advocate_challenge.summary)
                with st.expander(
                    f"Challenge details (confidence: {rec.devils_advocate_challenge.confidence:.0%})"
                ):
                    if rec.devils_advocate_challenge.assumptions:
                        st.markdown("**Assumptions questioned**")
                        for a in rec.devils_advocate_challenge.assumptions:
                            st.write(f"- {a}")
                    if rec.devils_advocate_challenge.uncertainties:
                        st.markdown("**Hidden risks raised**")
                        for u in rec.devils_advocate_challenge.uncertainties:
                            st.write(f"- {u}")

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

            st.caption(
                f"Data as of: {report.data_as_of}"
                if report.data_as_of
                else "Data as of: no dated data-provider evidence was available for this query."
            )

            with st.expander("Full Narrative Report"):
                st.write(report.narrative)
