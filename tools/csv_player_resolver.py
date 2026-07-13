"""CSV-backed player identity resolver.

Same rationale as :class:`tools.csv_provider.CSVPlayerDataProvider`:
structured local data, not a scraper. One row per alias (e.g. both
"Tielemans" and "Youri Tielemans" can point at the same identity) rather
than fuzzy matching, consistent with every other provider in this
codebase being exact-match-only.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from models.agent_io import ResolvedIdentity

logger = logging.getLogger(__name__)

_ID_COLUMNS = ("fbref_id", "fotmob_id", "transfermarkt_id")


class CSVPlayerResolver:
    """Loads identity records from a CSV file, keyed by query name (case-insensitive)."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._records: dict[str, ResolvedIdentity] | None = None

    def _load(self) -> dict[str, ResolvedIdentity]:
        if self._records is not None:
            return self._records

        records: dict[str, ResolvedIdentity] = {}
        if not self._csv_path.exists():
            logger.warning("CSV player identity file not found at %s", self._csv_path)
            self._records = records
            return records

        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    parsed = dict(row)
                    query_name = parsed.pop("query_name")
                    player_ids = {
                        col.removesuffix("_id"): parsed.pop(col)
                        for col in _ID_COLUMNS
                        if parsed.get(col)
                    }
                    for col in _ID_COLUMNS:
                        parsed.pop(col, None)
                    identity = ResolvedIdentity(
                        **parsed, player_ids=player_ids, source=self._csv_path.name
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed row %d in %s: %s", row_num, self._csv_path, exc
                    )
                    continue
                records[query_name.strip().lower()] = identity

        self._records = records
        return records

    def resolve(self, name: str) -> ResolvedIdentity | None:
        return self._load().get(name.strip().lower())
