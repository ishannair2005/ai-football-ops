from __future__ import annotations

from pathlib import Path

import pytest

from models.agent_io import EvidenceSource
from models.domain import InjuryRecord
from tools.csv_injury_provider import CSVInjuryProvider
from tools.injury_gateway import InjuryGateway
from tools.mock_injury_provider import MockInjuryProvider


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "injuries" / "sample_injuries.csv"


def test_csv_injury_provider_returns_record_for_known_player(sample_csv_path):
    provider = CSVInjuryProvider(sample_csv_path)

    record = provider.fetch_injury_record("Sample Winger")

    assert record is not None
    assert record.status == "Doubtful"
    assert record.injury_type == "Hamstring strain"
    assert record.source == "sample_injuries.csv"


def test_csv_injury_provider_lookup_is_case_insensitive(sample_csv_path):
    provider = CSVInjuryProvider(sample_csv_path)

    assert provider.fetch_injury_record("sample winger") is not None


def test_csv_injury_provider_returns_none_for_unknown_player(sample_csv_path):
    provider = CSVInjuryProvider(sample_csv_path)

    assert provider.fetch_injury_record("Nobody FC") is None


def test_csv_injury_provider_missing_file_returns_none_without_crashing(tmp_path):
    provider = CSVInjuryProvider(tmp_path / "does_not_exist.csv")

    assert provider.fetch_injury_record("Anyone") is None


def test_mock_injury_provider_returns_default_record():
    provider = MockInjuryProvider()

    record = provider.fetch_injury_record("Demo Player")

    assert record is not None
    assert record.status == "Available"


def test_mock_injury_provider_returns_none_for_unknown_player():
    provider = MockInjuryProvider()

    assert provider.fetch_injury_record("Nobody") is None


def _record(**overrides) -> InjuryRecord:
    defaults = dict(
        player="Test Player",
        status="Injured",
        injury_type="Knee",
        expected_return="2025-08-01",
        as_of_date="2025-06-01",
        source="test",
    )
    defaults.update(overrides)
    return InjuryRecord(**defaults)


def test_gateway_returns_empty_when_no_provider_has_data():
    gateway = InjuryGateway(providers=[MockInjuryProvider(records={}), MockInjuryProvider(records={})])

    assert gateway.fetch_injury_evidence("Test Player") == []


def test_gateway_returns_evidence_from_first_matching_provider():
    record = _record()
    gateway = InjuryGateway(
        providers=[
            MockInjuryProvider(records={}),
            MockInjuryProvider(records={"test player": record}),
        ]
    )

    evidence = gateway.fetch_injury_evidence("Test Player")

    assert len(evidence) == 1
    assert evidence[0].source == EvidenceSource.DATA_PROVIDER
    assert evidence[0].as_of_date == "2025-06-01"
    assert "Injured" in evidence[0].description
    assert "Knee" in evidence[0].description


def test_gateway_stops_at_first_provider_priority_order():
    first = _record(status="Available", injury_type=None)
    second = _record(status="Injured", injury_type="Hamstring")
    gateway = InjuryGateway(
        providers=[
            MockInjuryProvider(records={"test player": first}),
            MockInjuryProvider(records={"test player": second}),
        ]
    )

    evidence = gateway.fetch_injury_evidence("Test Player")

    assert len(evidence) == 1
    assert "Available" in evidence[0].description
