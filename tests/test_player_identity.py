from __future__ import annotations

from pathlib import Path

import pytest

from models.agent_io import ResolvedIdentity
from tools.csv_player_resolver import CSVPlayerResolver
from tools.mock_player_resolver import MockPlayerResolver
from tools.player_identity_gateway import PlayerIdentityGateway


@pytest.fixture
def sample_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "player_identity" / "sample_identities.csv"


def test_csv_player_resolver_resolves_fictional_demo_player(sample_csv_path):
    resolver = CSVPlayerResolver(sample_csv_path)

    identity = resolver.resolve("Sample Striker")

    assert identity is not None
    assert identity.full_name == "Sample Striker"
    assert identity.club == "Manchester United"
    assert identity.source == "sample_identities.csv"


def test_csv_player_resolver_resolves_both_aliases_for_a_real_player(sample_csv_path):
    resolver = CSVPlayerResolver(sample_csv_path)

    by_surname = resolver.resolve("Tielemans")
    by_full_name = resolver.resolve("Youri Tielemans")

    assert by_surname is not None
    assert by_surname.full_name == "Youri Tielemans"
    assert by_surname.club == "Aston Villa"
    assert by_full_name.full_name == "Youri Tielemans"


def test_csv_player_resolver_lookup_is_case_insensitive(sample_csv_path):
    resolver = CSVPlayerResolver(sample_csv_path)

    assert resolver.resolve("tielemans") is not None
    assert resolver.resolve("TIELEMANS") is not None


def test_csv_player_resolver_returns_none_for_unknown_player(sample_csv_path):
    resolver = CSVPlayerResolver(sample_csv_path)

    assert resolver.resolve("Some Totally Unknown Player") is None


def test_csv_player_resolver_never_fabricates_ids(sample_csv_path):
    resolver = CSVPlayerResolver(sample_csv_path)

    identity = resolver.resolve("Tielemans")

    # We have no real FBref/FotMob/Transfermarkt integration, so player_ids
    # must stay empty rather than a plausible-looking invented value.
    assert identity.player_ids == {}


def test_csv_player_resolver_missing_file_returns_none_without_crashing(tmp_path):
    resolver = CSVPlayerResolver(tmp_path / "does_not_exist.csv")

    assert resolver.resolve("Anyone") is None


def test_mock_player_resolver_returns_default_record():
    resolver = MockPlayerResolver()

    identity = resolver.resolve("Demo Player")

    assert identity is not None
    assert identity.club == "Demo FC"


def test_mock_player_resolver_returns_none_for_unknown_player():
    resolver = MockPlayerResolver()

    assert resolver.resolve("Nobody") is None


def _identity(**overrides) -> ResolvedIdentity:
    defaults = dict(full_name="Test Player", club="Test FC", as_of_date="2026-01-01", source="test")
    defaults.update(overrides)
    return ResolvedIdentity(**defaults)


def test_gateway_returns_none_when_no_provider_has_data():
    gateway = PlayerIdentityGateway(
        providers=[MockPlayerResolver(records={}), MockPlayerResolver(records={})]
    )

    assert gateway.resolve("Test Player") is None


def test_gateway_stops_at_first_provider_priority_order():
    first = _identity(club="First FC")
    second = _identity(club="Second FC")
    gateway = PlayerIdentityGateway(
        providers=[
            MockPlayerResolver(records={"test player": first}),
            MockPlayerResolver(records={"test player": second}),
        ]
    )

    identity = gateway.resolve("Test Player")

    assert identity.club == "First FC"


def test_gateway_falls_through_to_second_provider_when_first_has_nothing():
    second = _identity(club="Second FC")
    gateway = PlayerIdentityGateway(
        providers=[
            MockPlayerResolver(records={}),
            MockPlayerResolver(records={"test player": second}),
        ]
    )

    identity = gateway.resolve("Test Player")

    assert identity is not None
    assert identity.club == "Second FC"
