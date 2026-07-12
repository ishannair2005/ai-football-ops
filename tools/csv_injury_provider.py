"""CSV-backed injury/availability provider.

Same rationale as :class:`tools.csv_provider.CSVPlayerDataProvider`:
structured local data, not a scraper — a licensed/official medical feed
export would plug in the same way.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from models.domain import InjuryRecord

logger = logging.getLogger(__name__)


class CSVInjuryProvider:
    """Loads injury records from a CSV file, keyed by player (case-insensitive)."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._records: dict[str, InjuryRecord] | None = None

    def _load(self) -> dict[str, InjuryRecord]:
        if self._records is not None:
            return self._records

        records: dict[str, InjuryRecord] = {}
        if not self._csv_path.exists():
            logger.warning("CSV injury data file not found at %s", self._csv_path)
            self._records = records
            return records

        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    record = InjuryRecord(**dict(row), source=self._csv_path.name)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed row %d in %s: %s", row_num, self._csv_path, exc
                    )
                    continue
                records[record.player.strip().lower()] = record

        self._records = records
        return records

    def fetch_injury_record(self, player: str) -> InjuryRecord | None:
        return self._load().get(player.strip().lower())
