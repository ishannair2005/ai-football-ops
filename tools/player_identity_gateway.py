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

    def resolve(self, name: str) -> ResolvedIdentity | None:
        for provider in self._providers:
            identity = provider.resolve(name)
            if identity is not None:
                return identity
        return None
