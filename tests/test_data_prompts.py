from __future__ import annotations

from models.domain import PlayerStatsRecord
from prompts.data_prompts import build_player_data_section
from tools.data_gateway import PlayerDataGateway
from tools.mock_provider import MockPlayerDataProvider


def test_no_gateway_produces_generic_disclosure():
    section = build_player_data_section(None, {"player": "Sample Striker"})

    assert "none available for this request" in section


def test_no_player_in_context_produces_generic_disclosure():
    gateway = PlayerDataGateway(providers=[MockPlayerDataProvider(records={})])

    section = build_player_data_section(gateway, {})

    assert "none available for this request" in section


def test_player_named_but_not_found_flags_the_gap():
    gateway = PlayerDataGateway(providers=[MockPlayerDataProvider(records={})])

    section = build_player_data_section(gateway, {"player": "Nobody FC"})

    assert "no data-provider record found for 'Nobody FC'" in section


def test_player_found_cites_evidence_with_as_of_date():
    record = PlayerStatsRecord(
        name="Sample Striker",
        position="Forward",
        club="Manchester United",
        age=23,
        as_of_date="2025-05-25",
        source="sample_players.csv",
    )
    gateway = PlayerDataGateway(
        providers=[MockPlayerDataProvider(records={"sample striker": record})]
    )

    section = build_player_data_section(gateway, {"player": "Sample Striker"})

    assert "Fetched data (ground your assessment in this" in section
    assert "Sample Striker" in section
    assert "2025-05-25" in section
