"""Aggregates one or more :class:`NewsProvider` sources into evidence.

Unlike :class:`tools.injury_gateway.InjuryGateway`, this concatenates
results from every provider rather than stopping at the first hit — news
items from different sources are genuinely different pieces of
information, not competing values for the same fact, so there's nothing
to pick a "winner" between.
"""

from __future__ import annotations

from models.agent_io import Evidence, EvidenceDomain, EvidenceSource
from models.domain import NewsItem
from tools.news_provider import NewsProvider


class NewsGateway:
    def __init__(self, providers: list[NewsProvider]) -> None:
        self._providers = providers

    def fetch_news_evidence(self, subject: str) -> list[Evidence]:
        items: list[NewsItem] = []
        for provider in self._providers:
            items.extend(provider.fetch_news(subject))
        return [self._item_to_evidence(item) for item in items]

    @staticmethod
    def _item_to_evidence(item: NewsItem) -> Evidence:
        return Evidence(
            source=EvidenceSource.DATA_PROVIDER,
            description=f"[{item.category}] {item.headline} — {item.summary}",
            value=f"source={item.source}",
            as_of_date=item.as_of_date,
            confidence=0.6 if item.category == "rumour" else 0.85,
            domain=EvidenceDomain.NEWS,
        )
