"""Aggregates one or more :class:`PlayerDataProvider` sources into evidence.

This is the only thing agents talk to for player data — never a specific
adapter directly — so adding, removing, or reordering providers never
touches agent code. When providers disagree on a fact, that disagreement
is surfaced as its own piece of evidence instead of silently picking a
winner, per the platform's data-honesty rules.
"""

from __future__ import annotations

from models.agent_io import Evidence, EvidenceSource
from models.domain import PlayerStatsRecord
from tools.data_provider import PlayerDataProvider

_COMPARABLE_FIELDS = ("position", "club", "age")


class PlayerDataGateway:
    def __init__(self, providers: list[PlayerDataProvider]) -> None:
        self._providers = providers

    def fetch_player_evidence(self, name: str) -> list[Evidence]:
        """Query every provider for ``name`` and return evidence for the GM/agents.

        Queries all providers (not just the first hit) so disagreements
        between sources can be detected rather than masked by priority order.
        """
        records = [
            record
            for provider in self._providers
            if (record := provider.fetch_player(name)) is not None
        ]
        if not records:
            return []

        evidence = [self._record_to_evidence(record) for record in records]
        disagreement = self._detect_disagreement(name, records)
        if disagreement is not None:
            evidence.append(disagreement)
        return evidence

    @staticmethod
    def _record_to_evidence(record: PlayerStatsRecord) -> Evidence:
        stats = f"{record.appearances or 0} apps, {record.goals or 0}g {record.assists or 0}a"
        return Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description=(
                f"{record.name}: {record.position} at {record.club or 'unknown club'}, "
                f"age {record.age if record.age is not None else 'unknown'} ({stats})"
            ),
            value=f"source={record.source}",
            as_of_date=record.as_of_date,
            confidence=0.9,
        )

    @staticmethod
    def _detect_disagreement(name: str, records: list[PlayerStatsRecord]) -> Evidence | None:
        if len(records) < 2:
            return None

        conflicts: list[str] = []
        baseline = records[0]
        for field in _COMPARABLE_FIELDS:
            values = {getattr(r, field) for r in records if getattr(r, field) is not None}
            if len(values) > 1:
                conflicts.append(f"{field}: {sorted(str(v) for v in values)}")

        if not conflicts:
            return None

        sources = ", ".join(r.source for r in records)
        return Evidence(
            source=EvidenceSource.AGENT_FINDING,
            description=(
                f"Data providers disagree on {name} ({sources}): " + "; ".join(conflicts)
            ),
            value=None,
            as_of_date=baseline.as_of_date,
            confidence=0.3,
        )
