"""Aggregates one or more :class:`PlayerResolver` sources.

Simpler than :class:`tools.data_gateway.PlayerDataGateway` by design, same
scope cut as :class:`tools.injury_gateway.InjuryGateway`: no cross-source
disagreement detection, just first-hit-wins across providers in priority
order.
"""

from __future__ import annotations

from models.agent_io import ResolvedIdentity
from tools.player_resolver import PlayerResolver


class PlayerIdentityGateway:
    def __init__(self, providers: list[PlayerResolver]) -> None:
        self._providers = providers
        #: Set by resolve() when the most recent lookup came back empty
        #: because a live provider actually errored (network/timeout/HTTP
        #: failure) rather than simply having no record for the name.
        #: Providers with no such distinction (CSV/mock) never set this.
        self.last_error: str | None = None

    def resolve(self, name: str) -> ResolvedIdentity | None:
        self.last_error = None
        for provider in self._providers:
            identity = provider.resolve(name)
            if identity is not None:
                return identity
            error = getattr(provider, "last_error", None)
            if error:
                self.last_error = error
        return None
