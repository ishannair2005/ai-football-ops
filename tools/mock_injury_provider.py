"""Deterministic in-memory injury provider, for tests and safe fallback."""

from __future__ import annotations

from models.domain import InjuryRecord

_DEFAULT_RECORDS: dict[str, InjuryRecord] = {
    "demo player": InjuryRecord(
        player="Demo Player",
        status="Available",
        injury_type=None,
        expected_return=None,
        as_of_date="2025-06-01",
        source="mock_injury_provider",
    )
}


class MockInjuryProvider:
    """Returns canned injury records from a fixed, in-memory lookup table."""

    def __init__(self, records: dict[str, InjuryRecord] | None = None) -> None:
        self._records = records if records is not None else _DEFAULT_RECORDS

    def fetch_injury_record(self, player: str) -> InjuryRecord | None:
        return self._records.get(player.strip().lower())
