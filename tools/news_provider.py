"""Interface every news adapter must implement.

Returns a list rather than a single record (unlike player stats/injury
providers) since one subject can plausibly have several news items.
"""

from __future__ import annotations

from typing import Protocol

from models.domain import NewsItem


class NewsProvider(Protocol):
    """A source that can look up recent news items for a subject (player or club)."""

    def fetch_news(self, subject: str) -> list[NewsItem]:
        """Return news items about ``subject``, or an empty list if none."""
