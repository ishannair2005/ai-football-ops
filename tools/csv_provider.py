"""CSV-backed player data provider.

Deliberately not a scraper: this reads structured local data (a fixture
export today, a licensed/official feed export tomorrow) rather than
parsing a football website's HTML, which is fragile and often against a
site's terms of service. Any future provider that can hand back rows in
this shape can plug into the same interface without touching agents.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from models.domain import PlayerStatsRecord

logger = logging.getLogger(__name__)

_INT_FIELDS = ("age", "appearances", "goals", "assists")


class CSVPlayerDataProvider:
    """Loads player records from a CSV file, keyed by name (case-insensitive)."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._records: dict[str, PlayerStatsRecord] | None = None

    def _load(self) -> dict[str, PlayerStatsRecord]:
        if self._records is not None:
            return self._records

        records: dict[str, PlayerStatsRecord] = {}
        if not self._csv_path.exists():
            logger.warning("CSV player data file not found at %s", self._csv_path)
            self._records = records
            return records

        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    parsed = dict(row)
                    for field in _INT_FIELDS:
                        raw = parsed.get(field)
                        parsed[field] = int(raw) if raw not in (None, "") else None
                    record = PlayerStatsRecord(**parsed, source=self._csv_path.name)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed row %d in %s: %s", row_num, self._csv_path, exc
                    )
                    continue
                records[record.name.strip().lower()] = record

        self._records = records
        return records

    def fetch_player(self, name: str) -> PlayerStatsRecord | None:
        return self._load().get(name.strip().lower())
