"""Independent o1 browser agent and sequential model/order experiment runner."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse
import uuid

from brain import PROMPT_BEHAVIOR_MODE, PROMPT_VERSION, normalize_endpoint, request_json
from browser import execute, observe
from config import load_env, local_path, runtime_defaults
from questions import EXECUTION_POLICY, QUESTION_TURN_LIMIT, run_question
from evaluation import evaluate_submission, format_attempts, reevaluate, report, write_json


def select_workflows(manifest, orders, themes=("all",)):
    available_themes = [theme["id"] for theme in manifest.get("themes", [])]
    if not available_themes or "themeWorkflows" not in manifest:
        raise ValueError("The benchmark manifest has no theme workflows. Start the updated survey-benchmark app.")
    if not orders or len(set(orders)) != len(orders) or any(o not in {"order01", "order02", "order03"} for o in orders):
        raise ValueError("Select distinct order01, order02, and/or order03 values")
    selected_themes = available_themes if list(themes) == ["all"] else list(themes)
    if not selected_themes or len(set(selected_themes)) != len(selected_themes) or any(t not in available_themes for t in selected_themes):
        raise ValueError(f"SURVEY_THEMES must be all or distinct names from {available_themes}")
    selected = []
    for theme in selected_themes:
        variants = []
        for order in orders:
            matches = [w for w in manifest["themeWorkflows"] if w.get("themeId") == theme and w["orderId"] == order]
            if len(matches) != 1:
                raise ValueError(f"Expected one o1 theme workflow for {theme}/{order}")
            workflow = matches[0]
            ids = workflow["orderedQuestionIds"]
            if (workflow["profile"] != "standard" or workflow["occurrence"] != 1 or workflow["layout"] != "item"
                    or workflow["pageCount"] != 1 or workflow["questionCount"] != 11
                    or workflow["substantiveQuestionCount"] != 11 or workflow["attentionCheckCount"] != 0
                    or workflow["attentionCheckContentVersion"] != 0 or len(ids) != 11 or len(set(ids)) != 11
                    or not all(q.startswith(f"main-{theme}-") for q in ids) or not workflow["hasWelcomePage"]):
                raise ValueError(f"Expected 11 substantive questions and no attention checks: {workflow['id']}")
            if workflow["url"] != f"/survey/themes/{theme}/{order}":
                raise ValueError(f"Unexpected theme workflow route: {workflow['url']}")
            variants.append(workflow)
        if len({tuple(w["orderedQuestionIds"]) for w in variants}) != len(variants):
            raise ValueError(f"Theme {theme} contains duplicate presentation orders")
        if any(set(w["orderedQuestionIds"]) != set(variants[0]["orderedQuestionIds"]) for w in variants):
            raise ValueError(f"Theme {theme} must use the same questions in every order")
        selected.extend(variants)
    return selected


def build_schedule(workflows, repeats):
    """Finish one theme before the next, cycling orders on every repetition."""
    if repeats < 1:
        raise ValueError("Repeats must be positive")
    by_theme = {}
    for workflow in workflows:
        by_theme.setdefault(workflow["themeId"], []).append(workflow)
    return [(workflow, repeat, (repeat - 1) * len(variants) + index + 1)
            for variants in by_theme.values() for repeat in range(1, repeats + 1)
            for index, workflow in enumerate(variants)]


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:100] or "model"


async def run_one(browser, args, workflow, repeat, batch_dir, run_index=1):
    run_dir = batch_dir / f"run-{run_index:03d}-{workflow['orderId']}-repeat-{repeat:03d}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "workflow.json", workflow)
    summary = {"schema_version": 3, "condition": "theme-only",
               "theme_id": workflow["themeId"], "theme_label": workflow["themeLabel"],
               "order_id": workflow["orderId"], "run_index": run_index, "model": args.model, "observation": args.observation,
               "model_name": args.model_name or args.model,
               "behavior": args.behavior, "temperature": 0,
               "execution_policy": EXECUTION_POLICY, "question_turn_limit": QUESTION_TURN_LIMIT,
               "question_turn_budget": workflow["questionCount"] * QUESTION_TURN_LIMIT,
               "lm_base_url": args.lm_base_url, "base_url": args.base_url,
               "prompt_version": PROMPT_VERSION, "workflow_id": workflow["id"],
               "content_version": workflow["contentVersion"],
               "attention_check_content_version": workflow["attentionCheckContentVersion"],
               "question_count": workflow["questionCount"], "repeat": repeat,
               "started_at": datetime.now(timezone.utc).isoformat(),
               "status": "running", "model_calls": 0, "model_seconds": 0.0,
               "plan_errors": 0, "action_errors": 0, "actions_executed": 0,
               "model_errors": 0, "invalid_answer_turns": 0, "question_results": [],
               "prompt_tokens": 0, "completion_tokens": 0, "usage_reported_calls": 0}
    started = time.perf_counter()
    context = None
    snapshot = None
    fields = []
    trace = []
    try:
        context = await browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        page = await context.new_page()
        page.set_default_timeout(0)
        page.set_default_navigation_timeout(0)
        await page.goto(urljoin(args.base_url.rstrip("/") + "/", workflow["url"]), wait_until="domcontentloaded")
        welcome = await observe(page)
        if welcome["screen"] != "welcome":
            raise RuntimeError("A fresh run must begin at the welcome screen")
        write_json(run_dir / "welcome.json", welcome)
        await execute(page, {"tool": "click", "key": "start-survey"})
        trace.append({"kind": "harness_navigation", "action": "start-survey"})
        observation = await observe(page)
        fields = [field for field in observation["fields"] if "kind" in field]
        if [field["key"] for field in fields] != workflow["orderedQuestionIds"]:
            raise RuntimeError("Rendered question sequence does not match frozen manifest")
        instructions = welcome["instructions"] + observation["instructions"]
        history = []
        for number, field in enumerate(fields, 1):
            result = await run_question(page, args, workflow, field, number, run_dir,
                                        summary, trace, history, instructions)
            summary["question_results"].append(result)
            history.append({"question_id": field["key"], "prompt": field["prompt"],
                            "status": result["status"], "answer": result["answer"]})
            write_json(run_dir / "run_summary.json", dict(summary, wall_seconds=time.perf_counter() - started,
                       evaluation=evaluate_submission(None, workflow, args.db_path)))
        snapshot = await execute(page, {"tool": "click", "key": "submit-survey"})
        write_json(run_dir / "submission_snapshot.json", snapshot)
        trace.append({"kind": "harness_navigation", "action": "submit-survey", "status": snapshot["status"]})
        if snapshot["status"] != 200 or snapshot["response"].get("accepted") is not True:
            raise RuntimeError(f"Submission rejected: {snapshot['response']}")
        # Budget skips must be absent from the actual server submission, not just labeled in reports.
        for result in summary["question_results"]:
            if result["status"] == "unanswered" and result["question_id"] in snapshot["request"]["answers"]:
                raise RuntimeError(f"Unanswered question retained a submitted answer: {result['question_id']}")
        summary["status"] = "submitted"
        await page.locator('.completion-card').wait_for()
        if args.observation == "vision":
            await page.screenshot(path=str(run_dir / "completed.png"), full_page=True)
    except asyncio.CancelledError:
        summary["status"] = "interrupted"
        raise
    except Exception as exc:
        summary.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        summary["evaluation"] = evaluate_submission(snapshot, workflow, args.db_path)
        summary["format_attempts"] = format_attempts(fields, snapshot)
        summary["benchmark_run_id"] = snapshot["request"]["runId"] if snapshot else None
        summary["budget_exhausted_questions"] = sum(
            result["reason"] == "turn_budget_exhausted" for result in summary["question_results"])
        summary["answered_questions"] = sum(result["status"] == "answered" for result in summary["question_results"])
        write_json(run_dir / "trace.json", trace)
        write_json(run_dir / "run_summary.json", summary)
        write_json(run_dir / "evaluation.json", summary["evaluation"])
        if context:
            await context.close()
    print(f"{args.model} | {workflow['id']} | repeat {repeat}: {summary['status']} "
          f"({summary['model_calls']}/{summary['question_turn_budget']} question turns, "
          f"{summary['wall_seconds']:.1f}s)", flush=True)
    return summary


async def run_batch(args):
    if args.occurrence != "o1":
        raise ValueError("This runner currently supports SURVEY_OCCURRENCE=o1 only")
    if not args.orders or any(order not in {"order01", "order02", "order03"} for order in args.orders):
        raise ValueError("SURVEY_ORDERS must contain order01, order02, or order03, separated by spaces")
    manifest = await asyncio.to_thread(request_json, args.base_url.rstrip("/") + "/api/manifest")
    if manifest.get("suiteVersion") != args.suite_version:
        raise ValueError(f"SURVEY_VERSION={args.suite_version} does not match the site's suite version {manifest.get('suiteVersion')}")
    workflows = select_workflows(manifest, args.orders, args.themes)
    schedule = build_schedule(workflows, args.repeats)
    if args.dry_run:
        print(json.dumps({"workflows": workflows, "models": args.models or [args.model],
                          "repeats_per_order": args.repeats, "runs_per_model": len(schedule),
                          "schedule": [{"theme": w["themeId"], "order": w["orderId"], "repeat": r,
                                        "theme_run_index": i} for w, r, i in schedule]}, indent=2))
        return 0
    models = list(dict.fromkeys(args.models or [args.model]))
    if not all(models):
        raise ValueError("Set LLM_MODEL (or VLM_MODEL in vlm_only mode), or pass --model. Use `python agent.py models` for identifiers.")
    if args.model_name and len(models) > 1:
        raise ValueError("Leave MODEL_NAME empty when running multiple models so each gets its own folder")
    catalog = await asyncio.to_thread(request_json, normalize_endpoint(args.lm_base_url) + "/models",
                                     None, args.api_key)
    available = {entry["id"] for entry in catalog["data"]}
    missing = set(models) - available
    if missing:
        raise ValueError(f"Models not advertised by LM Studio: {sorted(missing)}. Available: {sorted(available)}")
    from playwright.async_api import async_playwright
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    report_dirs = set()
    failed = False
    print(f"{len({w['themeId'] for w in workflows})} themes x {len(args.orders)} orders x "
          f"{args.repeats} repeats = {len(schedule)} runs per model", flush=True)
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=not args.headed)
            try:
                for model in models:
                    args.model = model
                    model_dir = args.runs_dir / safe_name(args.model_name or model)
                    for workflow, repeat, run_index in schedule:
                        theme_dir = model_dir / f"o1_{workflow['themeId']}"
                        batch_dir = theme_dir / batch_id
                        if not batch_dir.exists():
                            batch_dir.mkdir(parents=True)
                            write_json(batch_dir / "manifest.json", manifest)
                            write_json(batch_dir / "experiment.json", {
                                "models": [model], "requested_models": models,
                                "model_name": args.model_name or model, "theme": workflow["themeId"],
                                "orders": args.orders, "repeats_per_order": args.repeats,
                                "planned_runs": len(args.orders) * args.repeats,
                                "total_runs_per_model": len(schedule), "attention_checks": False,
                                "observation": args.observation, "execution_policy": EXECUTION_POLICY,
                                "question_turn_limit": QUESTION_TURN_LIMIT,
                                "schedule": [{"theme": w["themeId"], "order": w["orderId"],
                                              "repeat": r, "theme_run_index": i} for w, r, i in schedule]})
                        report_dirs.update((batch_dir, theme_dir, model_dir, args.runs_dir))
                        summary = await run_one(browser, args, workflow, repeat, batch_dir, run_index)
                        failed |= not summary["evaluation"]["submitted"]
                        # Refresh metrics after every run, including failures.
                        for directory in (batch_dir, theme_dir, model_dir, args.runs_dir):
                            report(directory)
            finally:
                await browser.close()
    finally:
        for directory in sorted(report_dirs):
            report(directory)
        if report_dirs:
            print(f"Artifacts and model comparisons: {args.runs_dir}", flush=True)

    return 1 if failed else 0


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def parser():
    load_env()
    defaults = runtime_defaults()
    env = lambda key, default: os.getenv("AGENTIC_" + key, default)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=defaults["base_url"])
    common.add_argument("--lm-base-url", default=defaults["lm_base_url"])
    common.add_argument("--api-key", default=defaults["api_key"])
    common.add_argument("--db-path", type=local_path, default=env("DB_PATH", "../survey-benchmark/benchmark.sqlite"))
    common.add_argument("--runs-dir", type=local_path, default=env("RUNS_DIR", "runs"))
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", parents=[common], help="Run theme-only o1 workflows")
    run.add_argument("--model", default=defaults["model"])
    run.add_argument("--model-name", default=defaults["model_name"], help="Optional output-folder label; defaults to model ID")
    run.add_argument("--suite-version", default=defaults["suite_version"])
    run.add_argument("--occurrence", choices=["o1"], default=defaults["occurrence"])
    run.add_argument("--models", nargs="+", help="Exact model IDs; runs sequentially")
    run.add_argument("--orders", nargs="+", choices=["order01", "order02", "order03"],
                     default=defaults["orders"])
    run.add_argument("--themes", nargs="+", default=defaults["themes"], help="all or theme IDs in the desired sequence")
    run.add_argument("--repeats", type=positive_int, default=defaults["repeats"], help="Repeats per order within each theme")
    run.add_argument("--observation", choices=["dom", "vision"], default=defaults["observation"])
    run.add_argument("--behavior", choices=["completion", "unconstrained"], default=env("BEHAVIOR", PROMPT_BEHAVIOR_MODE))
    run.add_argument("--temperature", type=float, default=defaults["temperature"], help="Fixed at 0")
    run.add_argument("--headed", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Read manifest and print schedule; no browser or model calls")
    commands.add_parser("models", parents=[common], help="List model IDs from LM Studio")
    commands.add_parser("report", parents=[common], help="Aggregate existing runs to comparison CSV/JSON")
    commands.add_parser("evaluate", parents=[common], help="Refresh attention scores from the benchmark SQLite DB")
    return cli


def main():
    try:
        args = parser().parse_args()
        if args.command == "run":
            if args.observation not in {"dom", "vision"} or args.behavior not in {"completion", "unconstrained"}:
                raise ValueError("Invalid observation or behavior in Agentic/.env")
            if args.temperature != 0:
                raise ValueError("temperature is fixed at 0 for this experiment")
            for url in (args.base_url, args.lm_base_url):
                if urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc:
                    raise ValueError("Server URLs must be absolute HTTP(S) URLs")
            return asyncio.run(run_batch(args))
        if args.command == "models":
            response = request_json(normalize_endpoint(args.lm_base_url) + "/models", api_key=args.api_key)
            for entry in response["data"]:
                print(entry["id"])
        else:
            if args.command == "evaluate":
                print(f"Refreshed {reevaluate(args.runs_dir, args.db_path)} runs")
            rows = report(args.runs_dir)
            print(f"Wrote {len(rows)} comparison groups to {args.runs_dir}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed run artifacts are retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
