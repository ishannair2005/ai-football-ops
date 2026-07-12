"""Aggregates one or more :class:`InjuryProvider` sources into evidence.

Simpler than :class:`tools.data_gateway.PlayerDataGateway` by design: no
cross-source disagreement detection (a deliberate scope cut — one proof of
concept of that pattern already exists for player stats). Providers are
queried in priority order and the first hit wins.
"""

from __future__ import annotations

from models.agent_io import Evidence, EvidenceSource
from models.domain import InjuryRecord
from tools.injury_provider import InjuryProvider


class InjuryGateway:
    def __init__(self, providers: list[InjuryProvider]) -> None:
        self._providers = providers

    def fetch_injury_evidence(self, player: str) -> list[Evidence]:
        for provider in self._providers:
            record = provider.fetch_injury_record(player)
            if record is not None:
                return [self._record_to_evidence(record)]
        return []

    @staticmethod
    def _record_to_evidence(record: InjuryRecord) -> Evidence:
        detail = record.injury_type or "no reported injury"
        return_note = f", expected return {record.expected_return}" if record.expected_return else ""
        return Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description=f"{record.player}: {record.status} ({detail}{return_note})",
            value=f"source={record.source}",
            as_of_date=record.as_of_date,
            confidence=0.9,
        )
