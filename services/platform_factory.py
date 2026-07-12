"""Wires up a GeneralManagerAgent for the active club.

The Streamlit app and any future entrypoint (CLI, API) should build the
platform through this one function rather than re-assembling agents by
hand, so the roster of specialist agents lives in exactly one place.
"""

from __future__ import annotations

from agents.manager_agent import GeneralManagerAgent
from agents.scout_agent import ScoutAgent
from agents.tactical_agent import TacticalAgent
from agents.transfer_market_agent import TransferMarketAgent
from config.club_config import ClubConfig, load_club_config
from config.settings import DATA_DIR, get_settings
from services.llm_client import AnthropicLLMClient, LLMClient
from tools.csv_provider import CSVPlayerDataProvider
from tools.data_gateway import PlayerDataGateway
from tools.mock_provider import MockPlayerDataProvider


def _build_player_data_gateway() -> PlayerDataGateway:
    # CSV first (real, swappable export), Mock last (always has a demo
    # record so the pipeline is never entirely empty). Add further
    # providers here — agents never need to change when this list does.
    csv_path = DATA_DIR / "player_stats" / "sample_players.csv"
    return PlayerDataGateway(
        providers=[
            CSVPlayerDataProvider(csv_path),
            MockPlayerDataProvider(),
        ]
    )


def build_general_manager(
    club_id: str | None = None, llm_client: LLMClient | None = None
) -> GeneralManagerAgent:
    settings = get_settings()
    club_config: ClubConfig = load_club_config(club_id or settings.active_club)
    client = llm_client or AnthropicLLMClient(settings)
    data_gateway = _build_player_data_gateway()

    # Additional specialists register here as they're built in later phases.
    specialists = [
        ScoutAgent(client, club_config, data_gateway),
        TacticalAgent(client, club_config),
        TransferMarketAgent(client, club_config, data_gateway),
    ]
    return GeneralManagerAgent(client, club_config, specialists)
