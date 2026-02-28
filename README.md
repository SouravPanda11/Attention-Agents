# Attention Agents

Automation and evaluation stack for running LLM/VLM agents on a web survey and
analyzing run quality from recorded artifacts.

## Repository Layout

- `survey-site/`: Next.js 16 + React 19 survey app with SQLite logging (`data.sqlite`).
- `Agent/`: Playwright + LangGraph agent, batch runners, comparison scripts, plotting, and ingestion.
- `evaluation/answer_key.json`: Offline answer keys for `survey_v0` and `survey_v1`.

## Prerequisites

- Node.js 20+
- npm 10+
- Python 3.10+
- OpenAI-compatible endpoint(s) for configured LLM/VLM models (LM Studio or similar)

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

### 3) Configure `Agent/.env`

Set the model stack and target survey route:

- `AGENT_BRAIN_MODE`: `llm_only` | `vlm_only` | `hybrid`
- `LLM_ENABLED`, `VLM_ENABLED`
- `LLM_MODEL`, `VLM_MODEL`, `MODEL_NAME`
- `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_TEMPERATURE`, `LLM_TIMEOUT_S`
- `VLM_BASE_URL`, `VLM_API_KEY`, `VLM_TEMPERATURE`, `VLM_TIMEOUT_S`
- `SURVEY_TARGET` + `SURVEY_VERSION`:
  - `http://localhost:3000/survey` + `survey_v0`, or
  - `http://localhost:3000/survey_v1` + `survey_v1`

Important: `.env` is parsed top-to-bottom, so if a key appears multiple times,
the last assignment wins.

## Run Locally

Use two terminals.

### Terminal 1: start survey app

```powershell
cd survey-site
npm run dev
```

Site URL: `http://localhost:3000`

Routes:
- `http://localhost:3000/survey` (`survey_v0`)
- `http://localhost:3000/survey_v1` (`survey_v1`)

### Terminal 2: run agent

```powershell
cd Agent
.venv\Scripts\activate
python agent.py
```

Run outputs are written to:

`Agent/runs/<SURVEY_VERSION>/<MODEL_NAME>/<completion|unconstrained>/run_*`

`completion|unconstrained` comes from `PROMPT_BEHAVIOR_MODE` in
`Agent/brain.py` (in-code toggle).

Typical artifacts per run include:

- `trace.json`
- `run_summary.json`
- `submission_snapshot.json`
- `action_space_*.json`
- `ui_layout_trace*.json` (when layout trace is available)

## Analysis Workflow

Run from `Agent/` with the virtual environment activated.

### 1) Batch agent runs

```powershell
python run_n_times.py
```

Edit `NUM_RUNS`, `HEADLESS`, `DELAY_S`, `FAIL_FAST` in `run_n_times.py`.

### 2) Compare run outcomes

```powershell
python compare_runs.py
```

Outputs (per mode):
- `completed_compared_runs.csv`
- `unconstrained_compare_runs.csv`
- plots in `_compare_plots/`

This script validates against:
- `survey-site/data.sqlite`
- `evaluation/answer_key.json`

### 3) Compare planning failures

```powershell
python compare_plan_errors.py
```

Outputs (per mode):
- `completed_plan_error_compare.csv`
- `unconstrained_plan_error_compare.csv`
- plots in `_plan_error_plots/`

Optional merge/event exports are toggled in-code in `compare_plan_errors.py`.

### 4) Plot summary charts

```powershell
python overall_summary_plotting.py
python text_page_summary_plotting.py
python image_page_summary_plotting.py
```

- `overall_summary_plotting.py` writes to `Agent/runs/<SURVEY_VERSION>/_overall_summary_plots/`
- page-level scripts write to `Agent/runs/<SURVEY_VERSION>/_page_summary_plots/`
- `image_page_summary_plotting.py` does cross-version comparison when
  `SURVEY_VERSION != survey_v0` (baseline is `survey_v0`)

### 5) Ingest runs into an aggregate SQLite DB

```powershell
python ingest_runs.py --db-path runs_aggregate.sqlite
```

Useful flags:
- `--snapshot-only`: ingest only runs with `submission_snapshot.json`
- `--dry-run`: summarize without writing
- `--runs-root <path>`: override runs root (default: `Agent/runs`)

## Utility Scripts

### Capture manual full-page screenshots (survey_v0 only)

```powershell
python capture_survey_v0_screenshots.py --start-url http://localhost:3000/survey
```

This script intentionally rejects `/survey_v1`.

### Remove `run_dir` column from CSV files

From workspace root:

```powershell
python Agent\remove_run_dir_column.py Agent\runs --dry-run
```

### Survey DB utilities

From `survey-site/`:

```powershell
python scripts/db_report.py
python scripts/reset_db.py
```
