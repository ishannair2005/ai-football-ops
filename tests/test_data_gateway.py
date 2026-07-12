from __future__ import annotations

from models.agent_io import EvidenceSource
from models.domain import PlayerStatsRecord
from tools.data_gateway import PlayerDataGateway
from tools.mock_provider import MockPlayerDataProvider


def _provider(record: PlayerStatsRecord | None) -> MockPlayerDataProvider:
    records = {record.name.strip().lower(): record} if record else {}
    return MockPlayerDataProvider(records=records)


def _record(**overrides) -> PlayerStatsRecord:
    defaults = dict(
        name="Test Player",
        position="Forward",
        club="Test FC",
        age=22,
        as_of_date="2025-06-01",
        source="test",
    )
    defaults.update(overrides)
    return PlayerStatsRecord(**defaults)


def test_gateway_returns_empty_when_no_provider_has_data():
    gateway = PlayerDataGateway(providers=[_provider(None), _provider(None)])

    assert gateway.fetch_player_evidence("Test Player") == []


def test_gateway_returns_evidence_from_single_provider():
    gateway = PlayerDataGateway(providers=[_provider(_record())])

    evidence = gateway.fetch_player_evidence("Test Player")

    assert len(evidence) == 1
    assert evidence[0].source == EvidenceSource.DATA_PROVIDER
    assert evidence[0].as_of_date == "2025-06-01"


def test_gateway_queries_all_providers_and_flags_agreement_without_conflict_note():
    same_record = _record()
    gateway = PlayerDataGateway(providers=[_provider(same_record), _provider(same_record)])

    evidence = gateway.fetch_player_evidence("Test Player")

    assert len(evidence) == 2
    assert all(e.source == EvidenceSource.DATA_PROVIDER for e in evidence)


def test_gateway_flags_disagreement_between_providers():
    provider_a = _provider(_record(club="Test FC"))
    provider_b = _provider(_record(club="Rival FC"))
    gateway = PlayerDataGateway(providers=[provider_a, provider_b])

    evidence = gateway.fetch_player_evidence("Test Player")

    assert len(evidence) == 3
    disagreement = evidence[-1]
    assert disagreement.source == EvidenceSource.AGENT_FINDING
    assert "disagree" in disagreement.description.lower()
    assert "club" in disagreement.description.lower()
