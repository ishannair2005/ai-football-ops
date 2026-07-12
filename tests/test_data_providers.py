from __future__ import annotations

from pathlib import Path

import pytest

from tools.csv_provider import CSVPlayerDataProvider
from tools.mock_provider import MockPlayerDataProvider


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "player_stats" / "sample_players.csv"


def test_csv_provider_returns_record_for_known_player(sample_csv_path):
    provider = CSVPlayerDataProvider(sample_csv_path)

    record = provider.fetch_player("Sample Striker")

    assert record is not None
    assert record.position == "Forward"
    assert record.club == "Manchester United"
    assert record.source == "sample_players.csv"


def test_csv_provider_lookup_is_case_insensitive(sample_csv_path):
    provider = CSVPlayerDataProvider(sample_csv_path)

    assert provider.fetch_player("sample striker") is not None
    assert provider.fetch_player("SAMPLE STRIKER") is not None


def test_csv_provider_returns_none_for_unknown_player(sample_csv_path):
    provider = CSVPlayerDataProvider(sample_csv_path)

    assert provider.fetch_player("Nobody FC") is None


def test_csv_provider_missing_file_returns_none_without_crashing(tmp_path):
    provider = CSVPlayerDataProvider(tmp_path / "does_not_exist.csv")

    assert provider.fetch_player("Anyone") is None


def test_csv_provider_skips_malformed_row_without_crashing(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "name,position,club,age,nationality,appearances,goals,assists,as_of_date\n"
        "Broken Row,Forward,Some FC,not-a-number,England,10,1,1,2025-01-01\n"
        "Good Row,Forward,Some FC,20,England,10,1,1,2025-01-01\n"
    )
    provider = CSVPlayerDataProvider(bad_csv)

    assert provider.fetch_player("Broken Row") is None
    assert provider.fetch_player("Good Row") is not None


def test_mock_provider_returns_default_record():
    provider = MockPlayerDataProvider()

    record = provider.fetch_player("Demo Player")

    assert record is not None
    assert record.club == "Demo FC"


def test_mock_provider_returns_none_for_unknown_player():
    provider = MockPlayerDataProvider()

    assert provider.fetch_player("Nobody") is None


def test_mock_provider_accepts_custom_records():
    from models.domain import PlayerStatsRecord

    custom = {
        "custom player": PlayerStatsRecord(
            name="Custom Player",
            position="Goalkeeper",
            as_of_date="2025-01-01",
            source="custom",
        )
    }
    provider = MockPlayerDataProvider(records=custom)

    assert provider.fetch_player("Custom Player").position == "Goalkeeper"
