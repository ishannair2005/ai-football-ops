from __future__ import annotations

import requests
import pytest

import tools.sportsapipro_client as sportsapipro_client
from tools.sportsapipro_client import SportsAPIProClient
from tools.sportsapipro_identity_provider import SportsAPIProIdentityProvider
from tools.sportsapipro_stats_provider import SportsAPIProStatsProvider
from tools.sportsapipro_transfer_provider import SportsAPIProTransferProvider


@pytest.fixture(autouse=True)
def _reset_shared_player_id_cache():
    """search_player_id's cache is process-wide by design (that's the
    point — it's what stops identity/stats/transfer resolution from each
    re-searching the same name). Reset it between tests so one test's
    cached id can't silently mask another test's HTTP mocking."""
    sportsapipro_client._player_id_cache.clear()
    yield
    sportsapipro_client._player_id_cache.clear()


class _FakeResponse:
    def __init__(self, payload: dict, status_ok: bool = True) -> None:
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise requests.HTTPError("boom")

    def json(self) -> dict:
        return self._payload


_SEARCH_RESPONSE = {
    "data": {
        "results": [
            {"type": "player", "score": 5.0, "entity": {"id": 111, "name": "Test Player"}},
            {"type": "team", "score": 99.0, "entity": {"id": 222, "name": "Not A Player"}},
        ]
    }
}

_EMPTY_SEARCH_RESPONSE = {"data": {"results": []}}

_PROFILE_RESPONSE = {
    "data": {
        "player": {
            "name": "Test Player",
            "position": "F",
            "dateOfBirth": "2000-01-01T00:00:00+00:00",
            "country": {"name": "Testland"},
            "team": {
                "name": "Test FC",
                "uniqueTournament": {"name": "Test League"},
            },
            "contractUntilTimestamp": 1893456000,  # 2030-01-01
            "proposedMarketValue": 10_000_000,
        }
    }
}

_STATISTICS_RESPONSE = {
    "data": {
        "seasons": [
            {
                "year": "24/25",
                "startYear": 2024,
                "team": {"name": "Test FC"},
                "uniqueTournament": {"name": "Test League"},
                "statistics": {
                    "appearances": 20,
                    "goals": 5,
                    "assists": 3,
                    "minutesPlayed": 1500,
                    "yellowCards": 2,
                    "redCards": 0,
                    "accuratePassesPercentage": 88.5,
                    "tackles": 10,
                    "interceptions": 4,
                },
            },
            {
                "year": "24/25",
                "startYear": 2024,
                "team": {"name": "Other National Team"},
                "uniqueTournament": {"name": "Friendly"},
                "statistics": {"appearances": 2, "goals": 0, "assists": 0, "minutesPlayed": 90},
            },
        ]
    }
}

_TRANSFER_HISTORY_RESPONSE = {
    "data": {
        "transferHistory": [
            {
                "fromTeamName": "Old Club",
                "toTeamName": "Test FC",
                "transferFeeDescription": "10M €",
                "transferDateTimestamp": 1580256000,
            }
        ]
    }
}


def _patch_get(monkeypatch, responses_by_path_prefix: dict[str, dict]) -> None:
    def fake_get(url, headers=None, params=None, timeout=None):
        for prefix, payload in responses_by_path_prefix.items():
            if prefix in url:
                return _FakeResponse(payload)
        raise AssertionError(f"Unexpected URL requested: {url}")

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)


def test_client_returns_none_and_sets_last_error_on_http_failure(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        return _FakeResponse({}, status_ok=False)

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)
    client = SportsAPIProClient("key")

    result = client.get("/api/whatever")

    assert result is None
    assert client.last_error is not None


def test_client_returns_none_and_sets_last_error_on_network_exception(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)
    client = SportsAPIProClient("key")

    result = client.get("/api/whatever")

    assert result is None
    assert "no network" in client.last_error


def test_identity_provider_resolves_and_maps_fields(monkeypatch):
    _patch_get(
        monkeypatch,
        {"/api/search": _SEARCH_RESPONSE, "/api/players/111": _PROFILE_RESPONSE},
    )
    provider = SportsAPIProIdentityProvider("key")

    identity = provider.resolve("Test Player")

    assert identity is not None
    assert identity.full_name == "Test Player"
    assert identity.club == "Test FC"
    assert identity.position == "Forward"
    assert identity.nationality == "Testland"
    assert identity.competition == "Test League"
    assert identity.player_ids == {"sportsapipro": "111"}
    assert identity.source == "sportsapipro.com"
    assert provider.last_error is None


def test_identity_provider_returns_none_when_no_player_found(monkeypatch):
    _patch_get(monkeypatch, {"/api/search": _EMPTY_SEARCH_RESPONSE})
    provider = SportsAPIProIdentityProvider("key")

    assert provider.resolve("Nobody FC") is None
    assert provider.last_error is None


def test_identity_provider_sets_last_error_on_failure(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None):
        raise requests.Timeout("slow")

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)
    provider = SportsAPIProIdentityProvider("key")

    assert provider.resolve("Test Player") is None
    assert provider.last_error is not None


def test_identity_provider_caches_result_per_name(monkeypatch):
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        if "/api/search" in url:
            return _FakeResponse(_SEARCH_RESPONSE)
        return _FakeResponse(_PROFILE_RESPONSE)

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)
    provider = SportsAPIProIdentityProvider("key")

    provider.resolve("Test Player")
    call_count_after_first = len(calls)
    provider.resolve("Test Player")

    assert len(calls) == call_count_after_first


def test_stats_provider_selects_club_season_and_maps_fields(monkeypatch):
    _patch_get(
        monkeypatch,
        {
            "/api/search": _SEARCH_RESPONSE,
            "/api/players/111/statistics": _STATISTICS_RESPONSE,
            "/api/players/111": _PROFILE_RESPONSE,
        },
    )
    provider = SportsAPIProStatsProvider("key")

    record = provider.fetch_player("Test Player")

    assert record is not None
    assert record.club == "Test FC"
    assert record.appearances == 20
    assert record.minutes == 1500
    assert record.pass_accuracy == 88.5
    assert record.tackles == 10
    assert record.interceptions == 4
    assert record.progressive_carries is None


def test_transfer_provider_combines_fee_and_contract_expiry(monkeypatch):
    _patch_get(
        monkeypatch,
        {
            "/api/search": _SEARCH_RESPONSE,
            "/api/players/111/transfer-history": _TRANSFER_HISTORY_RESPONSE,
            "/api/players/111": _PROFILE_RESPONSE,
        },
    )
    provider = SportsAPIProTransferProvider("key")

    record = provider.fetch_transfer_record("Test Player")

    assert record is not None
    assert "10M €" in record.transfer_fee
    assert "Old Club" in record.transfer_fee
    assert record.contract_expiry == "2030-01-01"
    assert record.wages is None
    assert record.release_clause is None


def test_search_player_id_is_shared_across_independent_provider_instances(monkeypatch):
    """The identity, statistics, and transfer providers are three separate
    objects (each independently swappable) but must not each re-search the
    same name — that's what made a single query burn through the rate
    limit in practice. They share one cache via search_player_id."""
    search_calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        if "/api/search" in url:
            search_calls.append(url)
            return _FakeResponse(_SEARCH_RESPONSE)
        if "/statistics" in url:
            return _FakeResponse(_STATISTICS_RESPONSE)
        if "/transfer-history" in url:
            return _FakeResponse(_TRANSFER_HISTORY_RESPONSE)
        return _FakeResponse(_PROFILE_RESPONSE)

    monkeypatch.setattr("tools.sportsapipro_client.requests.get", fake_get)

    identity_provider = SportsAPIProIdentityProvider("key")
    stats_provider = SportsAPIProStatsProvider("key")
    transfer_provider = SportsAPIProTransferProvider("key")

    identity_provider.resolve("Test Player")
    stats_provider.fetch_player("Test Player")
    transfer_provider.fetch_transfer_record("Test Player")

    assert len(search_calls) == 1
