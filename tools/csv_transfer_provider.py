"""CSV-backed transfer/contract provider.

Same rationale as :class:`tools.csv_injury_provider.CSVInjuryProvider`:
structured local data, not a scraper — a licensed transfer/contract feed
export would plug in the same way.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from models.domain import TransferRecord

logger = logging.getLogger(__name__)


class CSVTransferProvider:
    """Loads transfer records from a CSV file, keyed by player (case-insensitive)."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._records: dict[str, TransferRecord] | None = None

    def _load(self) -> dict[str, TransferRecord]:
        if self._records is not None:
            return self._records

        records: dict[str, TransferRecord] = {}
        if not self._csv_path.exists():
            logger.warning("CSV transfer data file not found at %s", self._csv_path)
            self._records = records
            return records

        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    parsed = {k: (v if v else None) for k, v in dict(row).items()}
                    record = TransferRecord(**parsed, source=self._csv_path.name)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed row %d in %s: %s", row_num, self._csv_path, exc
                    )
                    continue
                records[record.player.strip().lower()] = record

        self._records = records
        return records

    def fetch_transfer_record(self, player: str) -> TransferRecord | None:
        return self._load().get(player.strip().lower())
