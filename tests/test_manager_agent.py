from __future__ import annotations

from agents.devils_advocate_agent import DevilsAdvocateAgent
from agents.manager_agent import GeneralManagerAgent, _data_quality_entry, _is_outdated
from agents.scout_agent import ScoutAgent
from agents.tactical_agent import TacticalAgent
from models.agent_io import (
    AgentRequest,
    AgentResponse,
    ComparisonOutcome,
    ComparisonRatingTier,
    ComparisonRecommendation,
    ComparisonSynthesis,
    DataQualityStatus,
    Evidence,
    ManagerSynthesis,
    PlayerNameExtraction,
    RecommendationVerdict,
    ResolvedIdentity,
    rating_tier_for_confidence,
)
from models.domain import InjuryRecord, PlayerStatsRecord, TransferRecord
from services.llm_client import LLMClient
from tools.data_gateway import PlayerDataGateway
from tools.injury_gateway import InjuryGateway
from tools.mock_injury_provider import MockInjuryProvider
from tools.mock_player_resolver import MockPlayerResolver
from tools.mock_provider import MockPlayerDataProvider
from tools.mock_transfer_provider import MockTransferProvider
from tools.player_identity_gateway import PlayerIdentityGateway
from tools.transfer_gateway import TransferGateway


def test_manager_consults_specialists_and_synthesizes(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout])

    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.recommendation == "Fake recommendation."
    assert len(result.agent_responses) == 1
    assert result.agent_responses[0].agent_name == "Scout Agent"
    # Player-name extraction (no player field set) + scout + draft synthesis.
    assert len(fake_llm_client.calls) == 3


def test_manager_never_performs_specialist_analysis_itself(fake_llm_client, man_utd_config):
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [])
    request = AgentRequest(query="Any question.", club_id="manchester_united")

    result = manager.handle_query(request)

    assert result.agent_responses == []
    # Even with zero specialists, the manager still only calls the LLM for
    # name extraction (no player field set) and synthesis, never fabricating
    # a specialist-style finding.
    assert len(fake_llm_client.calls) == 2


def test_manager_runs_devils_advocate_challenge_when_configured(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    devils_advocate = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout], devils_advocate)

    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.devils_advocate_challenge is not None
    assert result.devils_advocate_challenge.agent_name == "Devil's Advocate"
    assert result.challenge_resolution == "Fake resolution."
    # extraction + scout + draft synthesis + challenge + resolution synthesis
    assert len(fake_llm_client.calls) == 5


def test_manager_fires_status_and_agent_response_callbacks(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    devils_advocate = DevilsAdvocateAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout], devils_advocate)

    statuses: list[str] = []
    agent_responses: list[AgentResponse] = []
    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")

    manager.handle_query(
        request,
        on_status=statuses.append,
        on_agent_response=agent_responses.append,
    )

    assert statuses == [
        "Checking whether the query names specific players...",
        "Scout Agent is analyzing...",
        "Drafting initial recommendation...",
        "Devil's Advocate is challenging the recommendation...",
        "Finalizing the recommendation...",
    ]
    assert [r.agent_name for r in agent_responses] == ["Scout Agent", "Devil's Advocate"]


def test_manager_handle_query_works_without_callbacks(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout])
    request = AgentRequest(query="Should United sign Joao Neves?", club_id="manchester_united")

    # No on_status/on_agent_response passed — must not raise.
    result = manager.handle_query(request)

    assert result.agent_responses[0].agent_name == "Scout Agent"


def test_extract_player_names_prefers_explicit_context_without_calling_llm(fake_llm_client, man_utd_config):
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [])
    request = AgentRequest(
        query="Should we sign Tielemans?",
        club_id="manchester_united",
        context={"player": "Tielemans"},
    )

    names = manager._extract_player_names(request)

    assert names == ["Tielemans"]
    assert fake_llm_client.calls == []


def test_extract_player_names_parses_query_when_field_left_blank(man_utd_config):
    client = SequencedLLMClient([PlayerNameExtraction(players=["Tielemans"])])
    manager = GeneralManagerAgent(client, man_utd_config, [])
    request = AgentRequest(query="Should Manchester United sign Tielemans?", club_id="manchester_united")

    names = manager._extract_player_names(request)

    assert names == ["Tielemans"]
    assert len(client.calls) == 1


def test_extract_player_names_returns_multiple_for_a_comparison_query(man_utd_config):
    client = SequencedLLMClient([PlayerNameExtraction(players=["Sample Striker", "Sample Winger"])])
    manager = GeneralManagerAgent(client, man_utd_config, [])
    request = AgentRequest(
        query="Compare Sample Striker vs Sample Winger", club_id="manchester_united"
    )

    names = manager._extract_player_names(request)

    assert names == ["Sample Striker", "Sample Winger"]


def test_resolve_player_profile_resolves_real_player_identity(man_utd_config, fake_llm_client):
    identity = ResolvedIdentity(
        full_name="Youri Tielemans", club="Aston Villa", as_of_date="2026-07-13", source="test"
    )
    identity_gateway = PlayerIdentityGateway(
        providers=[MockPlayerResolver(records={"tielemans": identity})]
    )
    manager = GeneralManagerAgent(
        fake_llm_client, man_utd_config, [], identity_gateway=identity_gateway
    )

    profile = manager._resolve_player_profile("Tielemans")

    assert profile.resolved is True
    assert profile.full_name == "Youri Tielemans"
    assert profile.club == "Aston Villa"


def test_resolve_player_profile_records_gaps_when_nothing_configured(fake_llm_client, man_utd_config):
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [])

    profile = manager._resolve_player_profile("Tielemans")

    assert profile.resolved is False
    assert profile.full_name is None
    assert any("could not be verified" in gap for gap in profile.evidence_gaps)
    assert any("statistics unavailable" in gap for gap in profile.evidence_gaps)
    assert any("injury" in gap.lower() for gap in profile.evidence_gaps)
    assert {entry.domain: entry.status for entry in profile.data_quality} == {
        "Identity": DataQualityStatus.UNAVAILABLE,
        "Statistics": DataQualityStatus.UNAVAILABLE,
        "Injuries": DataQualityStatus.UNAVAILABLE,
        "Transfer": DataQualityStatus.UNAVAILABLE,
    }
    assert profile.overall_data_quality_score == 0.0


def test_resolve_player_profile_populates_verified_data_when_resolved(fake_llm_client, man_utd_config):
    identity = ResolvedIdentity(
        full_name="Sample Striker",
        club="Manchester United",
        as_of_date="2025-05-25",
        source="test",
    )
    identity_gateway = PlayerIdentityGateway(
        providers=[MockPlayerResolver(records={"sample striker": identity})]
    )
    stats_record = PlayerStatsRecord(
        name="Sample Striker",
        position="Forward",
        club="Manchester United",
        age=23,
        as_of_date="2025-05-25",
        source="test",
    )
    data_gateway = PlayerDataGateway(
        providers=[MockPlayerDataProvider(records={"sample striker": stats_record})]
    )
    injury_record = InjuryRecord(
        player="Sample Striker", status="Available", as_of_date="2025-05-25", source="test"
    )
    injury_gateway = InjuryGateway(
        providers=[MockInjuryProvider(records={"sample striker": injury_record})]
    )
    transfer_record = TransferRecord(
        player="Sample Striker",
        transfer_fee="£50m",
        as_of_date="2025-05-25",
        source="test",
    )
    transfer_gateway = TransferGateway(
        providers=[MockTransferProvider(records={"sample striker": transfer_record})]
    )
    manager = GeneralManagerAgent(
        fake_llm_client,
        man_utd_config,
        [],
        identity_gateway=identity_gateway,
        data_gateway=data_gateway,
        injury_gateway=injury_gateway,
        transfer_gateway=transfer_gateway,
    )
    profile = manager._resolve_player_profile("Sample Striker")

    assert profile.resolved is True
    assert profile.full_name == "Sample Striker"
    assert profile.club == "Manchester United"
    assert len(profile.stats_evidence) == 1
    assert len(profile.injury_evidence) == 1
    assert len(profile.transfer_evidence) == 1
    assert profile.evidence_gaps == []
    # Every domain found real evidence — status is AVAILABLE or (since this
    # fixture uses a fixed past date) OUTDATED, never UNAVAILABLE/PROVIDER_ERROR.
    assert all(
        entry.status in (DataQualityStatus.AVAILABLE, DataQualityStatus.OUTDATED)
        for entry in profile.data_quality
    )


def test_manager_threads_player_profile_to_specialists(fake_llm_client, man_utd_config):
    identity = ResolvedIdentity(
        full_name="Sample Striker",
        club="Manchester United",
        as_of_date="2025-05-25",
        source="test",
    )
    identity_gateway = PlayerIdentityGateway(
        providers=[MockPlayerResolver(records={"sample striker": identity})]
    )
    stats_record = PlayerStatsRecord(
        name="Sample Striker",
        position="Forward",
        club="Manchester United",
        age=23,
        as_of_date="2025-05-25",
        source="test",
    )
    data_gateway = PlayerDataGateway(
        providers=[MockPlayerDataProvider(records={"sample striker": stats_record})]
    )
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(
        fake_llm_client,
        man_utd_config,
        [scout],
        identity_gateway=identity_gateway,
        data_gateway=data_gateway,
    )
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    manager.handle_query(request)

    scout_call = next(c for c in fake_llm_client.calls if "Scouting task" in c["user_prompt"])
    assert "Sample Striker: Forward at Manchester United" in scout_call["user_prompt"]
    assert "2025-05-25" in scout_call["user_prompt"]


class SequencedLLMClient(LLMClient):
    """Returns pre-scripted responses in call order, so a test can prove the
    Manager actually uses the resolution call's output rather than quietly
    keeping the draft — something the shared, canned FakeLLMClient can't
    demonstrate since it returns identical data every call."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate_structured(self, *, system_prompt, user_prompt, response_model, max_tokens=2048):
        self.calls.append({"response_model": response_model})
        return self._responses.pop(0)


def test_manager_uses_resolution_synthesis_not_draft(man_utd_config):
    specialist_response = AgentResponse(
        agent_name="Scout Agent", summary="Scout finding.", confidence=0.6
    )
    draft = ManagerSynthesis(
        executive_summary="Draft summary.",
        recommendation="Draft recommendation.",
        verdict=RecommendationVerdict.BUY,
        confidence=0.7,
        key_risks=["Draft risk."],
        next_steps=["Draft step."],
    )
    challenge_response = AgentResponse(
        agent_name="Devil's Advocate", summary="Strong opposing argument.", confidence=0.5
    )
    final_synthesis = ManagerSynthesis(
        executive_summary="Final summary.",
        recommendation="Final recommendation, revised.",
        verdict=RecommendationVerdict.MONITOR,
        confidence=0.55,
        key_risks=["Final risk."],
        next_steps=["Final step."],
        challenge_resolution="Partially accepted the challenge; lowered confidence and added a risk.",
    )
    extraction = PlayerNameExtraction(players=[])
    client = SequencedLLMClient(
        [extraction, specialist_response, draft, challenge_response, final_synthesis]
    )
    scout = ScoutAgent(client, man_utd_config)
    devils_advocate = DevilsAdvocateAgent(client, man_utd_config)
    manager = GeneralManagerAgent(client, man_utd_config, [scout], devils_advocate)

    request = AgentRequest(query="Should United sign Sample Striker?", club_id="manchester_united")
    result = manager.handle_query(request)

    assert result.recommendation == "Final recommendation, revised."
    assert result.recommendation != draft.recommendation
    assert result.verdict == RecommendationVerdict.MONITOR
    assert result.verdict != draft.verdict
    assert result.challenge_resolution == (
        "Partially accepted the challenge; lowered confidence and added a risk."
    )
    assert result.devils_advocate_challenge.summary == "Strong opposing argument."
    assert len(client.calls) == 5


def test_rating_tier_thresholds():
    assert rating_tier_for_confidence(0.6) == ComparisonRatingTier.STRONG
    assert rating_tier_for_confidence(0.75) == ComparisonRatingTier.STRONG
    assert rating_tier_for_confidence(0.59) == ComparisonRatingTier.MODERATE
    assert rating_tier_for_confidence(0.4) == ComparisonRatingTier.MODERATE
    assert rating_tier_for_confidence(0.39) == ComparisonRatingTier.WEAK
    assert rating_tier_for_confidence(0.0) == ComparisonRatingTier.WEAK


def test_comparison_recommendation_computes_preferred_player_from_index():
    synthesis = ComparisonSynthesis(
        executive_summary="x",
        outcome=ComparisonOutcome.CLEAR_PREFERENCE,
        preferred_player_index=1,
        verdict_rationale="x",
        confidence=0.6,
    )
    rec = ComparisonRecommendation.from_synthesis(synthesis, ["A", "B"], [], [])
    assert rec.preferred_player == "B"


def test_comparison_recommendation_preferred_player_none_when_multiple_viable():
    synthesis = ComparisonSynthesis(
        executive_summary="x",
        outcome=ComparisonOutcome.MULTIPLE_VIABLE,
        preferred_player_index=None,
        verdict_rationale="x",
        confidence=0.5,
    )
    rec = ComparisonRecommendation.from_synthesis(synthesis, ["A", "B"], [], [])
    assert rec.preferred_player is None


def test_comparison_recommendation_preferred_player_none_when_index_out_of_range():
    synthesis = ComparisonSynthesis(
        executive_summary="x",
        outcome=ComparisonOutcome.CLEAR_PREFERENCE,
        preferred_player_index=5,
        verdict_rationale="x",
        confidence=0.5,
    )
    rec = ComparisonRecommendation.from_synthesis(synthesis, ["A", "B"], [], [])
    assert rec.preferred_player is None


def test_handle_comparison_runs_each_specialist_once_per_candidate(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    tactical = TacticalAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout, tactical])
    request = AgentRequest(
        query="Compare Sample Striker vs Sample Winger", club_id="manchester_united"
    )

    result = manager._handle_comparison(request, ["Sample Striker", "Sample Winger"])

    assert isinstance(result, ComparisonRecommendation)
    assert result.player_names == ["Sample Striker", "Sample Winger"]
    assert [e.player_name for e in result.players] == ["Sample Striker", "Sample Winger"]
    assert len(result.players[0].responses) == 2
    assert len(result.players[1].responses) == 2
    assert result.players[0].responses[0].agent_name == "Scout Agent (Sample Striker)"
    assert result.players[1].responses[0].agent_name == "Scout Agent (Sample Winger)"
    assert result.players[0].responses[1].agent_name == "Tactical Agent (Sample Striker)"


def test_handle_comparison_builds_decision_matrix_from_real_responses(fake_llm_client, man_utd_config):
    scout = ScoutAgent(fake_llm_client, man_utd_config)
    manager = GeneralManagerAgent(fake_llm_client, man_utd_config, [scout])
    request = AgentRequest(query="Compare A vs B", club_id="manchester_united")

    result = manager._handle_comparison(request, ["Sample Striker", "Sample Winger"])

    assert len(result.decision_matrix) == 1
    criterion = result.decision_matrix[0]
    assert criterion.criterion_name == "Scout Agent"
    assert len(criterion.ratings) == 2
    # FakeLLMClient's canned AgentResponse always has confidence 0.7.
    assert criterion.ratings[0].confidence == 0.7
    assert criterion.ratings[0].tier == ComparisonRatingTier.STRONG
    assert criterion.ratings[0].summary == "Fake specialist finding."


def test_handle_query_routes_to_comparison_when_two_players_extracted(man_utd_config):
    client = SequencedLLMClient(
        [
            PlayerNameExtraction(players=["Sample Striker", "Sample Winger"]),
            AgentResponse(agent_name="Scout Agent", summary="Striker take.", confidence=0.6),
            AgentResponse(agent_name="Scout Agent", summary="Winger take.", confidence=0.5),
            ComparisonSynthesis(
                executive_summary="Comparison summary.",
                outcome=ComparisonOutcome.CLEAR_PREFERENCE,
                preferred_player_index=0,
                verdict_rationale="Striker fits the system better.",
                confidence=0.6,
            ),
        ]
    )
    scout = ScoutAgent(client, man_utd_config)
    manager = GeneralManagerAgent(client, man_utd_config, [scout])
    request = AgentRequest(
        query="Compare Sample Striker vs Sample Winger", club_id="manchester_united"
    )

    result = manager.handle_query(request)

    assert isinstance(result, ComparisonRecommendation)
    assert result.preferred_player == "Sample Striker"
    assert result.outcome == ComparisonOutcome.CLEAR_PREFERENCE


def _evidence(as_of_date: str) -> Evidence:
    return Evidence(source="data_provider", description="x", as_of_date=as_of_date)


def test_is_outdated_true_when_all_dates_older_than_threshold():
    evidence = [_evidence("2020-01-01")]

    assert _is_outdated(evidence, max_age_days=30) is True


def test_is_outdated_false_when_a_date_is_within_threshold():
    from datetime import date

    recent = date.today().isoformat()
    evidence = [_evidence("2020-01-01"), _evidence(recent)]

    assert _is_outdated(evidence, max_age_days=30) is False


def test_is_outdated_false_when_no_dated_evidence():
    assert _is_outdated([], max_age_days=30) is False


def test_data_quality_entry_unavailable_when_no_evidence_and_no_error():
    entry = _data_quality_entry("Statistics", [], None)

    assert entry.status == DataQualityStatus.UNAVAILABLE


def test_data_quality_entry_provider_error_when_no_evidence_but_error_reported():
    entry = _data_quality_entry("Identity", [], "network timeout")

    assert entry.status == DataQualityStatus.PROVIDER_ERROR
    assert entry.detail == "network timeout"


def test_data_quality_entry_available_when_evidence_present():
    entry = _data_quality_entry("Injuries", [_evidence("2025-01-01")], None)

    assert entry.status == DataQualityStatus.AVAILABLE


def test_data_quality_entry_outdated_when_evidence_stale():
    entry = _data_quality_entry("Statistics", [_evidence("2020-01-01")], None, max_age_days=30)

    assert entry.status == DataQualityStatus.OUTDATED


def test_resolve_player_profile_reports_provider_error_for_identity():
    class ErroringResolver:
        def resolve(self, name: str):
            return None

    from tools.player_identity_gateway import PlayerIdentityGateway

    identity_gateway = PlayerIdentityGateway(providers=[])
    # Force a provider-level error to be recorded on the gateway, exactly as
    # a live SportsAPIPro provider would when the API errors rather than
    # simply having no record.
    identity_gateway._providers = [ErroringResolver()]
    identity_gateway._providers[0].last_error = "boom"

    manager = GeneralManagerAgent(
        llm_client=None, club_config=None, specialist_agents=[], identity_gateway=identity_gateway
    )
    profile = manager._resolve_player_profile("Anyone")

    identity_entry = next(e for e in profile.data_quality if e.domain == "Identity")
    assert identity_entry.status == DataQualityStatus.PROVIDER_ERROR
