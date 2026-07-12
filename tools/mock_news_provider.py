"""Deterministic in-memory news provider, for tests and safe fallback."""

from __future__ import annotations

from models.domain import NewsItem

_DEFAULT_ITEMS: dict[str, list[NewsItem]] = {
    "demo player": [
        NewsItem(
            subject="Demo Player",
            headline="Demo Player linked with a move away from Demo FC",
            category="rumour",
            summary="Unconfirmed reports suggest interest from a rival club.",
            as_of_date="2025-06-01",
            source="mock_news_provider",
        )
    ]
}


class MockNewsProvider:
    """Returns canned news items from a fixed, in-memory lookup table."""

    def __init__(self, items: dict[str, list[NewsItem]] | None = None) -> None:
        self._items = items if items is not None else _DEFAULT_ITEMS

    def fetch_news(self, subject: str) -> list[NewsItem]:
        return self._items.get(subject.strip().lower(), [])
