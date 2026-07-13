from __future__ import annotations

from models.agent_io import Evidence, EvidenceSource
from prompts.data_prompts import (
    build_identity_section,
    build_injury_section,
    build_news_section,
    build_player_data_section,
)
from tests.conftest import make_player_profile as _profile


def _evidence(**overrides) -> Evidence:
    defaults = dict(
        source=EvidenceSource.DATA_PROVIDER,
        description="Sample Striker stats",
        as_of_date="2025-05-25",
    )
    defaults.update(overrides)
    return Evidence(**defaults)


def test_identity_section_no_profile_produces_generic_disclosure():
    section = build_identity_section(None)

    assert "no player named" in section.lower()


def test_identity_section_unresolved_flags_the_gap():
    profile = _profile(resolved=False, full_name=None, club=None, identity_as_of=None, identity_source=None)

    section = build_identity_section(profile)

    assert "could not be verified" in section
    assert "Sample Striker" in section


def test_identity_section_resolved_cites_source_and_date():
    section = build_identity_section(_profile())

    assert "Sample Striker" in section
    assert "Manchester United" in section
    assert "2025-05-25" in section


def test_player_data_section_no_profile_produces_generic_disclosure():
    section = build_player_data_section(None)

    assert "none available" in section.lower()


def test_player_data_section_unresolved_produces_generic_disclosure():
    section = build_player_data_section(_profile(resolved=False))

    assert "none available" in section.lower()


def test_player_data_section_resolved_but_no_stats_flags_the_gap():
    section = build_player_data_section(_profile(stats_evidence=[]))

    assert "no data-provider record found" in section


def test_player_data_section_cites_evidence_with_as_of_date():
    evidence = [_evidence(description="Sample Striker: Forward at Manchester United")]

    section = build_player_data_section(_profile(stats_evidence=evidence))

    assert "Verified statistics (ground your assessment in this" in section
    assert "Sample Striker" in section
    assert "2025-05-25" in section


def test_injury_section_unresolved_flags_the_gap():
    section = build_injury_section(_profile(resolved=False))

    assert "none available" in section.lower()


def test_injury_section_resolved_but_no_record_flags_the_gap():
    section = build_injury_section(_profile(injury_evidence=[]))

    assert "no injury-provider record found" in section


def test_injury_section_cites_evidence():
    evidence = [_evidence(description="Sample Striker: Available")]

    section = build_injury_section(_profile(injury_evidence=evidence))

    assert "Injury data (cite this" in section
    assert "Sample Striker" in section


def test_news_section_no_subject_produces_generic_disclosure():
    section = build_news_section([], None)

    assert "none available" in section.lower()


def test_news_section_no_items_flags_the_gap():
    section = build_news_section([], "Sample Striker")

    assert "no news-provider items found for 'Sample Striker'" in section


def test_news_section_cites_evidence():
    evidence = [_evidence(description="[rumour] Sample Striker linked with a move")]

    section = build_news_section(evidence, "Sample Striker")

    assert "Recent news on Sample Striker" in section
    assert "Sample Striker linked with a move" in section
