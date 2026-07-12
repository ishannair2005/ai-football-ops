# AI Football Operations Platform

A multi-agent AI system that simulates a professional football club's
operations department. Manchester United is the default configured club;
any club can be added under `data/clubs/` without touching source code.

## Phase 1

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

## Phase 2 (current)

Adds a real, swappable data layer so agents can ground answers in actual
records instead of LLM recall alone:

- `tools/data_provider.py` — `PlayerDataProvider` Protocol every adapter
  implements. Agents and the gateway depend only on this interface.
- `tools/csv_provider.py` — `CSVPlayerDataProvider`, reads structured
  local data. Deliberately **not** a scraper: sites like FBref,
  Transfermarkt, Sofascore, and FotMob often prohibit scraping in their
  ToS and change HTML frequently, which makes scraper adapters brittle
  and risky. A licensed/official feed export would plug in the same way.
- `tools/mock_provider.py` — `MockPlayerDataProvider`, an in-memory
  fallback used by tests and as a safe default.
- `tools/data_gateway.py` — `PlayerDataGateway` queries all configured
  providers, converts records into `Evidence`, and flags disagreements
  between sources instead of silently picking one.
- `data/player_stats/sample_players.csv` — **illustrative fixture data**
  for exercising the pipeline end to end. Not a live feed; replace with a
  real export before relying on it for actual scouting decisions.
- `ScoutAgent` now accepts an optional `PlayerDataGateway`. When a
  request's `context["player"]` is set, it fetches and cites that
  player's data-provider evidence (with an `as_of_date`); otherwise it
  behaves exactly as in Phase 1, with an explicit "no data" disclosure.
- Streamlit gained an optional "Player name" field — try `Sample Striker`
  against the bundled demo CSV.

Not yet built (later phases): scraping/licensed-API adapters for
FBref/Understat/Transfermarkt/Sofascore/FotMob, News API integration,
remaining specialist agents (Performance Analytics, Tactical, Transfer
Market, Medical, Opponent Analysis, Squad Planning, News & Context,
Devil's Advocate), Report Agent, multi-round agent debate/challenge
logic, charts.

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
