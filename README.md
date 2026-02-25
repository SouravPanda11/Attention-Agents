# Attention Agents

`Attention Agents` has two parts:

- `survey-site/`: Next.js survey app.
- `Agent/`: Python Playwright + LangGraph agent that completes the survey and logs run artifacts.

## Prerequisites

- Node.js 20+
- npm 10+
- Python 3.10+
- A running OpenAI-compatible endpoint (LM Studio or similar) for LLM/VLM calls

## First-Time Setup

### 1) Install survey-site dependencies

```powershell
cd survey-site
npm install
```

### 2) Create Agent virtual environment

```powershell
cd ..\Agent
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
```

### 3) Configure Agent environment

Edit `Agent/.env` and set at least:

- `MODEL_NAME` (used as run output folder name)
- `AGENT_BRAIN_MODE` (`llm_only`, `vlm_only`, or `hybrid`)
- `LLM_*` variables
- `VLM_*` variables if using `vlm_only` or `hybrid`

## Run Locally

Use two terminals.

### Terminal 1: Start the survey app

```powershell
cd survey-site
npm run dev
```

Site URL: `http://localhost:3000`

### Terminal 2: Run the agent

```powershell
cd Agent
.venv\Scripts\activate
python agent.py
```

Agent target URL: `http://localhost:3000/survey`

Run outputs are saved under:

`Agent/runs/survey_v0/<MODEL_NAME>/<completion|unconstrained>/run_*`

Each run now includes `submission_snapshot.json` with normalized response payload
captured from browser session storage for offline aggregation/debugging.
For planner attribution, inspect:
- `trace.json` events: `model_raw`, `model_plan`, `plan_provenance`, `plan`
- `run_summary.json` keys: `model_raw_source_counts`, `model_plan_source_counts`, `accepted_plan_source_counts`

## Useful Commands

Run multiple times:

```powershell
cd Agent
.venv\Scripts\activate
python run_n_times.py
```

Compare runs:

```powershell
cd Agent
.venv\Scripts\activate
python compare_runs.py
```

Aggregate all run artifacts into a local DB (after pulling runs from workers):

```powershell
cd Agent
.venv\Scripts\activate
python ingest_runs.py --db-path runs_aggregate.sqlite
```

If you only want runs that have `submission_snapshot.json`:

```powershell
python ingest_runs.py --db-path runs_aggregate.sqlite --snapshot-only
```
