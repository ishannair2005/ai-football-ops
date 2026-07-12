from __future__ import annotations

from pathlib import Path

import pytest

from models.domain import NewsItem
from tools.csv_news_provider import CSVNewsProvider
from tools.mock_news_provider import MockNewsProvider
from tools.news_gateway import NewsGateway


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "news" / "sample_news.csv"


def test_csv_news_provider_returns_items_for_known_subject(sample_csv_path):
    provider = CSVNewsProvider(sample_csv_path)

    items = provider.fetch_news("Sample Striker")

    assert len(items) == 2
    assert all(item.source == "sample_news.csv" for item in items)


def test_csv_news_provider_lookup_is_case_insensitive(sample_csv_path):
    provider = CSVNewsProvider(sample_csv_path)

    assert len(provider.fetch_news("sample striker")) == 2


def test_csv_news_provider_returns_empty_for_unknown_subject(sample_csv_path):
    provider = CSVNewsProvider(sample_csv_path)

    assert provider.fetch_news("Nobody FC") == []


def test_csv_news_provider_missing_file_returns_empty_without_crashing(tmp_path):
    provider = CSVNewsProvider(tmp_path / "does_not_exist.csv")

    assert provider.fetch_news("Anyone") == []


def test_mock_news_provider_returns_default_item():
    provider = MockNewsProvider()

    items = provider.fetch_news("Demo Player")

    assert len(items) == 1
    assert items[0].category == "rumour"


def test_mock_news_provider_returns_empty_for_unknown_subject():
    provider = MockNewsProvider()

    assert provider.fetch_news("Nobody") == []


def _item(**overrides) -> NewsItem:
    defaults = dict(
        subject="Test Player",
        headline="Test headline",
        category="confirmed",
        summary="Test summary.",
        as_of_date="2025-06-01",
        source="test",
    )
    defaults.update(overrides)
    return NewsItem(**defaults)


def test_gateway_returns_empty_when_no_provider_has_data():
    gateway = NewsGateway(providers=[MockNewsProvider(items={}), MockNewsProvider(items={})])

    assert gateway.fetch_news_evidence("Test Player") == []


def test_gateway_aggregates_items_across_all_providers():
    item_a = _item(headline="Headline A")
    item_b = _item(headline="Headline B")
    gateway = NewsGateway(
        providers=[
            MockNewsProvider(items={"test player": [item_a]}),
            MockNewsProvider(items={"test player": [item_b]}),
        ]
    )

    evidence = gateway.fetch_news_evidence("Test Player")

    assert len(evidence) == 2
    descriptions = {e.description for e in evidence}
    assert any("Headline A" in d for d in descriptions)
    assert any("Headline B" in d for d in descriptions)


def test_gateway_rumour_gets_lower_confidence_than_confirmed():
    rumour = _item(category="rumour", headline="Rumour headline")
    confirmed = _item(category="confirmed", headline="Confirmed headline")
    gateway = NewsGateway(providers=[MockNewsProvider(items={"test player": [rumour, confirmed]})])

    evidence = gateway.fetch_news_evidence("Test Player")

    rumour_ev = next(e for e in evidence if "Rumour headline" in e.description)
    confirmed_ev = next(e for e in evidence if "Confirmed headline" in e.description)
    assert rumour_ev.confidence < confirmed_ev.confidence
