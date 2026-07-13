from __future__ import annotations

from pathlib import Path

import pytest

from models.agent_io import EvidenceSource
from models.domain import TransferRecord
from tools.csv_transfer_provider import CSVTransferProvider
from tools.mock_transfer_provider import MockTransferProvider
from tools.transfer_gateway import TransferGateway


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "transfers" / "sample_transfers.csv"


def test_csv_transfer_provider_returns_none_for_unknown_player(sample_csv_path):
    provider = CSVTransferProvider(sample_csv_path)

    assert provider.fetch_transfer_record("Nobody FC") is None


def test_csv_transfer_provider_lookup_is_case_insensitive(sample_csv_path):
    provider = CSVTransferProvider(sample_csv_path)

    assert provider.fetch_transfer_record("demo player") is not None


def test_csv_transfer_provider_missing_file_returns_none_without_crashing(tmp_path):
    provider = CSVTransferProvider(tmp_path / "does_not_exist.csv")

    assert provider.fetch_transfer_record("Anyone") is None


def test_csv_transfer_provider_parses_blank_fields_as_none(tmp_path):
    csv_path = tmp_path / "transfers.csv"
    csv_path.write_text(
        "player,transfer_fee,wages,release_clause,contract_expiry,as_of_date\n"
        "Test Player,,,,,2025-06-01\n"
    )
    provider = CSVTransferProvider(csv_path)

    record = provider.fetch_transfer_record("Test Player")

    assert record is not None
    assert record.transfer_fee is None
    assert record.wages is None


def test_mock_transfer_provider_returns_default_record():
    provider = MockTransferProvider()

    record = provider.fetch_transfer_record("Demo Player")

    assert record is not None
    assert record.transfer_fee is None


def test_mock_transfer_provider_returns_none_for_unknown_player():
    provider = MockTransferProvider()

    assert provider.fetch_transfer_record("Nobody") is None


def _record(**overrides) -> TransferRecord:
    defaults = dict(
        player="Test Player",
        transfer_fee="£50m",
        wages=None,
        release_clause=None,
        contract_expiry="2028-06-30",
        as_of_date="2025-06-01",
        source="test",
    )
    defaults.update(overrides)
    return TransferRecord(**defaults)


def test_gateway_returns_empty_when_no_provider_has_data():
    gateway = TransferGateway(
        providers=[MockTransferProvider(records={}), MockTransferProvider(records={})]
    )

    assert gateway.fetch_transfer_evidence("Test Player") == []


def test_gateway_returns_evidence_from_first_matching_provider():
    record = _record()
    gateway = TransferGateway(
        providers=[
            MockTransferProvider(records={}),
            MockTransferProvider(records={"test player": record}),
        ]
    )

    evidence = gateway.fetch_transfer_evidence("Test Player")

    assert len(evidence) == 1
    assert evidence[0].source == EvidenceSource.DATA_PROVIDER
    assert evidence[0].as_of_date == "2025-06-01"
    assert "£50m" in evidence[0].description


def test_gateway_describes_record_with_no_known_figures_honestly():
    record = _record(transfer_fee=None, contract_expiry=None)
    gateway = TransferGateway(providers=[MockTransferProvider(records={"test player": record})])

    evidence = gateway.fetch_transfer_evidence("Test Player")

    assert len(evidence) == 1
    assert "no fee, wage, release clause, or contract expiry on file" in evidence[0].description


def test_gateway_tracks_last_error_from_erroring_provider():
    class ErroringProvider:
        def __init__(self) -> None:
            self.last_error = "boom"

        def fetch_transfer_record(self, player: str) -> TransferRecord | None:
            return None

    gateway = TransferGateway(providers=[ErroringProvider()])

    evidence = gateway.fetch_transfer_evidence("Test Player")

    assert evidence == []
    assert gateway.last_error == "boom"
