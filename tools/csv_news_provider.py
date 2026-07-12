"""CSV-backed news provider.

Same rationale as :class:`tools.csv_provider.CSVPlayerDataProvider`:
structured local data, not a scraper — a licensed news API integration
would plug in behind this same interface.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from models.domain import NewsItem

logger = logging.getLogger(__name__)


class CSVNewsProvider:
    """Loads news items from a CSV file, grouped by subject (case-insensitive)."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._items: dict[str, list[NewsItem]] | None = None

    def _load(self) -> dict[str, list[NewsItem]]:
        if self._items is not None:
            return self._items

        items: dict[str, list[NewsItem]] = {}
        if not self._csv_path.exists():
            logger.warning("CSV news data file not found at %s", self._csv_path)
            self._items = items
            return items

        with self._csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                try:
                    item = NewsItem(**dict(row), source=self._csv_path.name)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed row %d in %s: %s", row_num, self._csv_path, exc
                    )
                    continue
                items.setdefault(item.subject.strip().lower(), []).append(item)

        self._items = items
        return items

    def fetch_news(self, subject: str) -> list[NewsItem]:
        return self._load().get(subject.strip().lower(), [])
