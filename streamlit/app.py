"""Phase 1 Streamlit skeleton for the AI Football Operations Platform.

Wired to the real GeneralManagerAgent (not mocked). Presentation is
intentionally minimal — the project's emphasis is the agent architecture,
not the UI. Later phases add charts, richer risk analysis, and full
agent-finding breakdowns.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config.settings import get_settings
from models.agent_io import AgentRequest, FinalRecommendation
from services.logging_config import configure_logging
from services.llm_client import LLMClientError
from services.platform_factory import build_general_manager

configure_logging()

st.set_page_config(page_title="AI Football Operations Platform", layout="wide")

settings = get_settings()

st.title("AI Football Operations Platform")
st.caption(f"Active club: {settings.active_club.replace('_', ' ').title()}")

query = st.text_area(
    "Ask the football operations department a question",
    placeholder="Should Manchester United sign Joao Neves?",
    height=100,
)

player_name = st.text_input(
    "Player name (optional)",
    placeholder="e.g. Sample Striker",
    help="If the question is about a specific player, name them here so the "
    "Scout Agent can look up data-provider records instead of relying on "
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
                manager = build_general_manager(settings.active_club)
                result: FinalRecommendation = manager.handle_query(
                    AgentRequest(query=query, club_id=settings.active_club, context=context)
                )
        except LLMClientError as exc:
            st.error(str(exc))
        else:
            st.header("Executive Summary")
            st.write(result.executive_summary)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Recommendation")
                st.write(result.recommendation)
            with col2:
                st.metric("Confidence", f"{result.confidence:.0%}")

            if result.key_risks:
                st.subheader("Key Risks")
                for risk in result.key_risks:
                    st.write(f"- {risk}")

            if result.next_steps:
                st.subheader("Next Steps")
                for step in result.next_steps:
                    st.write(f"- {step}")

            st.subheader("Agent Findings")
            for response in result.agent_responses:
                with st.expander(f"{response.agent_name} (confidence: {response.confidence:.0%})"):
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
