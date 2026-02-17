# Attention Agents

This repository contains:

- `survey-site/`: a Next.js survey web app (text + image survey flow, APIs, SQLite-backed session/log storage).
- `Agent/`: a Playwright + LangGraph automation agent that navigates the survey, plans actions with an LLM/VLM, and stores run artifacts.

## What It Is About

The project is a sandbox for evaluating agentic behavior on a controlled survey task:

- The survey app provides multi-step forms and attention checks.
- The agent observes DOM + screenshots, plans tool actions, executes them, and records traces.

## Prerequisites

- Node.js 20+
- Python 3.10+
- A running OpenAI-compatible model endpoint (for example LM Studio) for agent planning

## Setup

### 1) Survey app

```powershell
cd survey-site
npm install
```

### 2) Agent

```powershell
cd Agent
python -m venv .venv
.venv\Scripts\activate
pip install playwright langgraph lxml httpx
python -m playwright install chromium
```

## Run

### 1) Start the survey site

```powershell
cd survey-site
npm run dev
```

The survey will be available at `http://localhost:3000`.

### 2) Configure model endpoint for the agent

Set environment variables (PowerShell example):

```powershell
$env:AGENT_BRAIN_MODE="llm_only"        # llm_only | vlm_only | hybrid
$env:LLM_ENABLED="1"
$env:VLM_ENABLED="0"
$env:LLM_BASE_URL="http://127.0.0.1:1234/v1"
$env:LLM_MODEL="<your-llm-model-id>"
$env:LLM_API_KEY="local"
```

Optional VLM variables for `vlm_only` or `hybrid`:

```powershell
$env:VLM_ENABLED="1"
$env:VLM_BASE_URL="http://127.0.0.1:1234/v1"
$env:VLM_MODEL="<your-vlm-model-id>"
$env:VLM_API_KEY="local"
```

### 3) Run the agent

```powershell
cd Agent
python agent.py
```

The agent targets `http://localhost:3000/survey` and writes artifacts under `Agent/runs/run_*`:

- DOM snapshots
- screenshots
- `trace.json`
