# AI Football Operations Platform

A multi-agent AI system that simulates a professional football club's
operations department. Manchester United is the default configured club;
any club can be added under `data/clubs/` without touching source code.

## Phase 1 (current)

- `config/` — runtime settings (`.env`-driven) and club configuration
  loader (`ClubConfig`, `data/clubs/*.yaml`).
- `models/` — shared Pydantic contracts (`AgentRequest`, `AgentResponse`,
  `Evidence`, `FinalRecommendation`) that all agents communicate through.
- `prompts/` — shared prompt scaffolding (club context injection,
  data-honesty rules) reused by every agent.
- `services/` — `LLMClient` (Anthropic-backed, schema-enforced structured
  output), logging config, and the platform factory that wires agents
  together.
- `agents/` — `BaseAgent` (shared LLM-call/validation flow), `ScoutAgent`
  (first specialist), `GeneralManagerAgent` (orchestrator; delegates,
  never analyzes itself).
- `streamlit/app.py` — minimal UI wired to the real agent pipeline.
- `tests/` — unit tests using a fake LLM client (no network calls).

Not yet built (later phases): remaining specialist agents (Performance
Analytics, Tactical, Transfer Market, Medical, Opponent Analysis, Squad
Planning, News & Context, Devil's Advocate), Report Agent, live
data-provider adapters, multi-round agent debate/challenge logic, charts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Run tests

```bash
pytest
```

## Run the app

```bash
streamlit run streamlit/app.py
```

## Switching clubs

Set `ACTIVE_CLUB` in `.env` to any file name under `data/clubs/` (e.g.
`arsenal`), or add a new YAML file there following the same schema as
`data/clubs/manchester_united.yaml`.
