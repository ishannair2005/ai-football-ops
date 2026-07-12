"""Deterministic in-memory player data provider.

Used by tests (no filesystem, no network) and as a safe fallback in the
gateway so the pipeline always has at least one demonstrable record even
if no CSV export is configured.
"""

from __future__ import annotations

from models.domain import PlayerStatsRecord

_DEFAULT_RECORDS: dict[str, PlayerStatsRecord] = {
    "demo player": PlayerStatsRecord(
        name="Demo Player",
        position="Forward",
        club="Demo FC",
        age=24,
        nationality="Demoland",
        appearances=30,
        goals=15,
        assists=6,
        as_of_date="2025-06-01",
        source="mock_provider",
    )
}


class MockPlayerDataProvider:
    """Returns canned records from a fixed, in-memory lookup table."""

    def __init__(self, records: dict[str, PlayerStatsRecord] | None = None) -> None:
        self._records = records if records is not None else _DEFAULT_RECORDS

    def fetch_player(self, name: str) -> PlayerStatsRecord | None:
        return self._records.get(name.strip().lower())
