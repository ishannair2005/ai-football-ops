"""Live season-statistics provider backed by sportsapipro.com.

Requires the player's numeric id, resolved independently of identity
resolution (so this works correctly even if identity comes from a
different provider, e.g. CSV/mock, for a given name) via the shared,
process-wide :func:`tools.sportsapipro_client.search_player_id` cache —
so running identity, statistics, and transfer resolution back-to-back for
the same name only searches once, not three times.

`/api/players/{id}/statistics` returns full career history across every
club and competition the player has appeared for. This provider selects
the player's current-club, most recent season entry (falling back to the
entry with the most minutes played when a club fielded them across
several competitions in the same season) — the same "one representative
season row" shape every other stats provider already returns. Advanced
metrics the API doesn't expose (progressive carries/passes) are left
unset rather than estimated.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from models.domain import PlayerStatsRecord
from tools.sportsapipro_client import SportsAPIProClient, search_player_id

_POSITION_LABELS = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}


def _age_from_date_of_birth(date_of_birth: str | None) -> int | None:
    if not date_of_birth:
        return None
    try:
        dob = datetime.fromisoformat(date_of_birth.replace("Z", "+00:00")).date()
    except ValueError:
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


class SportsAPIProStatsProvider:
    """Implements :class:`tools.data_provider.PlayerDataProvider`."""

    def __init__(self, api_key: str) -> None:
        self._client = SportsAPIProClient(api_key)
        self.last_error: str | None = None

    def fetch_player(self, name: str) -> PlayerStatsRecord | None:
        self.last_error = None
        player_id = search_player_id(self._client, name)
        self.last_error = self._client.last_error
        if player_id is None:
            return None
        return self._fetch_current_season(player_id)

    def _fetch_current_season(self, player_id: int) -> PlayerStatsRecord | None:
        profile_data = self._client.get(f"/api/players/{player_id}")
        self.last_error = self.last_error or self._client.last_error
        player = (profile_data or {}).get("data", {}).get("player")
        if not player:
            return None
        club = (player.get("team") or {}).get("name")

        stats_data = self._client.get(f"/api/players/{player_id}/statistics")
        self.last_error = self.last_error or self._client.last_error
        seasons = (stats_data or {}).get("data", {}).get("seasons", [])
        club_seasons = [s for s in seasons if (s.get("team") or {}).get("name") == club]
        if not club_seasons:
            return None

        current = max(
            club_seasons,
            key=lambda s: (s.get("startYear") or 0, s["statistics"].get("minutesPlayed") or 0),
        )
        stats = current["statistics"]

        return PlayerStatsRecord(
            name=player.get("name", ""),
            position=_POSITION_LABELS.get(player.get("position"), player.get("position") or ""),
            club=club,
            age=_age_from_date_of_birth(player.get("dateOfBirth")),
            nationality=(player.get("country") or {}).get("name"),
            appearances=stats.get("appearances"),
            goals=stats.get("goals"),
            assists=stats.get("assists"),
            minutes=stats.get("minutesPlayed"),
            yellow_cards=stats.get("yellowCards"),
            red_cards=stats.get("redCards"),
            pass_accuracy=stats.get("accuratePassesPercentage"),
            tackles=stats.get("tackles"),
            interceptions=stats.get("interceptions"),
            as_of_date=datetime.now(timezone.utc).date().isoformat(),
            source="sportsapipro.com",
        )
