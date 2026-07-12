from __future__ import annotations

from models.agent_io import AgentRequest
from services.platform_factory import build_general_manager


def test_build_general_manager_assembles_all_specialists(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    result = manager.handle_query(request)

    names = {response.agent_name for response in result.agent_responses}
    assert names == {"Scout Agent", "Tactical Agent", "Transfer Market Agent"}
    assert result.devils_advocate_challenge is not None
    assert result.devils_advocate_challenge.agent_name == "Devil's Advocate"
    # 3 specialists + draft synthesis + Devil's Advocate challenge + final resolution.
    assert len(fake_llm_client.calls) == 6


def test_build_general_manager_uses_csv_backed_data_for_bundled_sample_player(fake_llm_client):
    manager = build_general_manager("manchester_united", llm_client=fake_llm_client)
    request = AgentRequest(
        query="Should we sign Sample Striker?",
        club_id="manchester_united",
        context={"player": "Sample Striker"},
    )

    manager.handle_query(request)

    scout_call = next(
        call
        for call in fake_llm_client.calls
        if "Scouting task" in call["user_prompt"]
    )
    assert "Sample Striker: Forward at Manchester United" in scout_call["user_prompt"]
