"""Wires up a GeneralManagerAgent for the active club.

The Streamlit app and any future entrypoint (CLI, API) should build the
platform through this one function rather than re-assembling agents by
hand, so the roster of specialist agents lives in exactly one place.
"""

from __future__ import annotations

from agents.manager_agent import GeneralManagerAgent
from agents.scout_agent import ScoutAgent
from config.club_config import ClubConfig, load_club_config
from config.settings import get_settings
from services.llm_client import AnthropicLLMClient, LLMClient


def build_general_manager(
    club_id: str | None = None, llm_client: LLMClient | None = None
) -> GeneralManagerAgent:
    settings = get_settings()
    club_config: ClubConfig = load_club_config(club_id or settings.active_club)
    client = llm_client or AnthropicLLMClient(settings)

    # Phase 1 roster: Scout Agent only. Additional specialists register
    # here as they're built in later phases.
    specialists = [
        ScoutAgent(client, club_config),
    ]
    return GeneralManagerAgent(client, club_config, specialists)
