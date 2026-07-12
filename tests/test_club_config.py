from __future__ import annotations

import pytest

from config.club_config import ClubConfigError, load_club_config


def test_loads_manchester_united():
    club = load_club_config("manchester_united")
    assert club.name == "Manchester United"
    assert club.manager == "Michael Carrick"


def test_loads_second_club_without_code_changes():
    club = load_club_config("arsenal")
    assert club.name == "Arsenal"
    assert club.manager == "Mikel Arteta"


def test_missing_club_raises_clear_error():
    with pytest.raises(ClubConfigError):
        load_club_config("nonexistent_club_xyz")
