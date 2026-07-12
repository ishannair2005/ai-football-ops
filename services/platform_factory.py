"""Wires up the football operations platform for the active club.

The Streamlit app and any future entrypoint (CLI, API) should build the
platform through these functions rather than re-assembling agents by
hand, so the roster of specialists and data providers lives in exactly
one place.
"""

from __future__ import annotations

from agents.devils_advocate_agent import DevilsAdvocateAgent
from agents.manager_agent import GeneralManagerAgent
from agents.performance_analytics_agent import PerformanceAnalyticsAgent
from agents.report_agent import ReportAgent
from agents.scout_agent import ScoutAgent
from agents.tactical_agent import TacticalAgent
from agents.transfer_market_agent import TransferMarketAgent
from config.club_config import ClubConfig, load_club_config
from config.settings import DATA_DIR, get_settings
from models.agent_io import AgentRequest, PlatformResult, ReportRequest
from services.llm_client import AnthropicLLMClient, LLMClient
from tools.csv_injury_provider import CSVInjuryProvider
from tools.csv_news_provider import CSVNewsProvider
from tools.csv_provider import CSVPlayerDataProvider
from tools.data_gateway import PlayerDataGateway
from tools.injury_gateway import InjuryGateway
from tools.mock_injury_provider import MockInjuryProvider
from tools.mock_news_provider import MockNewsProvider
from tools.mock_provider import MockPlayerDataProvider
from tools.news_gateway import NewsGateway


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


def _build_injury_gateway() -> InjuryGateway:
    csv_path = DATA_DIR / "injuries" / "sample_injuries.csv"
    return InjuryGateway(
        providers=[
            CSVInjuryProvider(csv_path),
            MockInjuryProvider(),
        ]
    )


def _build_news_gateway() -> NewsGateway:
    csv_path = DATA_DIR / "news" / "sample_news.csv"
    return NewsGateway(
        providers=[
            CSVNewsProvider(csv_path),
            MockNewsProvider(),
        ]
    )


def build_general_manager(
    club_id: str | None = None, llm_client: LLMClient | None = None
) -> GeneralManagerAgent:
    settings = get_settings()
    club_config: ClubConfig = load_club_config(club_id or settings.active_club)
    client = llm_client or AnthropicLLMClient(settings)
    data_gateway = _build_player_data_gateway()
    injury_gateway = _build_injury_gateway()
    news_gateway = _build_news_gateway()

    # Additional specialists register here as they're built in later phases.
    specialists = [
        ScoutAgent(client, club_config, data_gateway, injury_gateway),
        TacticalAgent(client, club_config),
        TransferMarketAgent(client, club_config, data_gateway),
        PerformanceAnalyticsAgent(client, club_config, data_gateway),
    ]
    devils_advocate = DevilsAdvocateAgent(client, club_config, news_gateway)
    return GeneralManagerAgent(client, club_config, specialists, devils_advocate)


class FootballOperationsPlatform:
    """Runs a query through the General Manager, then the Report Agent."""

    def __init__(self, manager: GeneralManagerAgent, report_agent: ReportAgent) -> None:
        self._manager = manager
        self._report_agent = report_agent

    def handle_query(self, request: AgentRequest) -> PlatformResult:
        recommendation = self._manager.handle_query(request)
        report_request = ReportRequest(
            original_query=request.query,
            club_id=request.club_id,
            recommendation=recommendation,
        )
        report = self._report_agent.analyze(report_request)
        return PlatformResult(recommendation=recommendation, report=report)


def build_platform(
    club_id: str | None = None, llm_client: LLMClient | None = None
) -> FootballOperationsPlatform:
    settings = get_settings()
    club_config: ClubConfig = load_club_config(club_id or settings.active_club)
    client = llm_client or AnthropicLLMClient(settings)

    manager = build_general_manager(club_id, client)
    report_agent = ReportAgent(client, club_config)
    return FootballOperationsPlatform(manager, report_agent)
