"""Execute exported AC plans in file order, fetching one frozen workflow at a time."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlencode
import uuid

from brain import normalize_endpoint, request_json
from config import local_path
from evaluation import report, write_json


def ensure_plan(path):
    if path.is_file():
        return
    default = local_path("../survey-benchmark/run-v1.jsonl")
    if path != default:
        raise ValueError(f"Run plan does not exist: {path}")
    node = shutil.which("node")
    if not node:
        raise ValueError("Install Node.js to generate the default v1 plan, or supply --plan FILE")
    subprocess.run([node, str(default.with_name("run-v1.cjs"))], check=True)


def read_plan(path):
    """Stream rows; reject invalid identities and backwards occurrence/layout order."""
    previous = None
    with Path(path).open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                occurrence = row["occurrence"]
                layout = row["layout"]
                repeat = row["repeatIndex"]
                if (type(occurrence) is not int or not 1 <= occurrence <= 8 or
                        layout not in {"navigation", "item"} or
                        (occurrence == 1 and layout != "navigation") or
                        type(repeat) is not int or repeat < 1):
                    raise ValueError("Invalid occurrence, layout, or repetition")
                sample = row["sampleId"]
                if not re.fullmatch(rf"o{occurrence}-s[0-9]{{4}}", sample) or sample.endswith("s0000"):
                    raise ValueError("Invalid sample ID")
                order = row["orderId"]
                if order not in {"order01", "order02", "order03"}:
                    raise ValueError("Invalid order ID")
                suffix = "-item" if layout == "item" else ""
                workflow_id = f"v1-standard-{sample}{suffix}-{order}"
                url = f"/survey/samples/{sample}/{order}" + ("/item" if layout == "item" else "")
                if (row["suiteVersion"] != "v1" or row["condition"] != "attention-horizon" or
                        row["workflowId"] != workflow_id or row["url"] != url or
                        row["planId"] != f"{workflow_id}-repeat{repeat:03d}" or
                        row["questionCount"] != 13 * occurrence + 1 or
                        row["pageCount"] != (1 if layout == "item" else occurrence) or
                        not isinstance(row["orderSeed"], str)):
                    raise ValueError("Plan identity or counts do not match the v1 AC suite")
                key = (occurrence, 0 if layout == "navigation" else 1, sample, order, repeat)
                if previous is not None and key <= previous:
                    raise ValueError("Plan must advance by occurrence, navigation/item, sample, order, repeat without duplicates")
                previous = key
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid plan row {line_number}: {exc}") from exc
            yield row


def selected_rows(args):
    for index, row in enumerate(read_plan(args.plan), 1):
        if index < args.start_index:
            continue
        if args.limit is not None and index >= args.start_index + args.limit:
            break
        yield index, row


async def fetch_workflow(args, row):
    query = urlencode({"sampleId": row["sampleId"], "order": row["orderId"],
                       "layout": row["layout"], "limit": 1})
    manifest = await asyncio.to_thread(request_json, args.base_url.rstrip("/") + "/api/manifest?" + query)
    workflows = manifest.get("workflows", [])
    if manifest.get("suiteVersion") != "v1" or len(workflows) != 1 or manifest["pagination"]["total"] != 1:
        raise ValueError(f"Expected exactly one v1 workflow for {row['planId']}")
    workflow = workflows[0]
    for plan_key, workflow_key in (("workflowId", "id"), ("sampleId", "sampleId"),
                                   ("condition", "condition"), ("occurrence", "occurrence"),
                                   ("layout", "layout"), ("orderId", "orderId"),
                                   ("orderSeed", "orderSeed"), ("questionCount", "questionCount"),
                                   ("pageCount", "pageCount"), ("url", "url")):
        if row[plan_key] != workflow[workflow_key]:
            raise ValueError(f"Live workflow differs from plan: {row['planId']} ({plan_key})")
    pages = workflow["pageQuestionIds"]
    ids = workflow["orderedQuestionIds"]
    if (len(pages) != workflow["pageCount"] or any(not page for page in pages) or
            [key for page in pages for key in page] != ids or
            len(ids) != workflow["questionCount"] or len(set(ids)) != len(ids) or not workflow["hasWelcomePage"]):
        raise ValueError(f"Invalid frozen question sequence: {workflow['id']}")
    return workflow


async def run_plan_batch(args, run_one):
    if args.suite_version != "v1":
        raise ValueError("The ordered AC run plan requires SURVEY_VERSION=v1")
    await asyncio.to_thread(ensure_plan, args.plan)
    # Validate the whole file before any browser or model calls, even with --limit.
    total = sum(1 for _ in read_plan(args.plan))
    groups = Counter(f"o{row['occurrence']}_{row['layout']}" for _, row in selected_rows(args))
    count = sum(groups.values())
    if not count:
        raise ValueError(f"No plan rows selected (plan contains {total} rows)")
    if args.dry_run:
        print(json.dumps({"plan": str(args.plan), "total_plan_rows": total, "runs_per_model": count,
                          "start_index": args.start_index, "models": args.models or [args.model],
                          "groups_in_execution_order": dict(groups),
                          "first_run": next(selected_rows(args))[1]}, indent=2))
        return 0
    models = list(dict.fromkeys(args.models or [args.model]))
    if not all(models):
        raise ValueError("Set LLM_MODEL (or VLM_MODEL), or pass --model")
    if args.model_name and len(models) > 1:
        raise ValueError("Leave MODEL_NAME empty when running multiple models")
    manifest = await asyncio.to_thread(request_json, args.base_url.rstrip("/") + "/api/manifest?limit=1")
    if manifest.get("suiteVersion") != "v1":
        raise ValueError("Start the v1 survey-benchmark app at SURVEY_TARGET")
    catalog = await asyncio.to_thread(request_json, normalize_endpoint(args.lm_base_url) + "/models", None, args.api_key)
    available = {entry["id"] for entry in catalog["data"]}
    if set(models) - available:
        raise ValueError(f"Models not advertised by LM Studio: {sorted(set(models) - available)}")
    from playwright.async_api import async_playwright
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    failed = False
    print(f"Executing {count:,} v1 plan rows per model from {args.plan}", flush=True)
    # Reports stay local to the sample while running; full plan reports refresh at the end.
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=not args.headed)
            try:
                for model in models:
                    args.model = model
                    name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_name or model).strip("._-")[:100] or "model"
                    model_dir = args.runs_dir / name / "v1-plans" / batch_id
                    model_dir.mkdir(parents=True)
                    write_json(model_dir / "experiment.json", {"model": model, "plan": str(args.plan),
                               "start_index": args.start_index, "limit": args.limit, "planned_runs": count,
                               "groups_in_execution_order": dict(groups), "suite_version": "v1"})
                    write_json(model_dir / "manifest.json", manifest)
                    cached = None
                    with (model_dir / "schedule.jsonl").open("w", encoding="utf-8") as schedule_file:
                        for index, row in selected_rows(args):
                            schedule_file.write(json.dumps(dict(row, plan_row_index=index)) + "\n")
                    for index, row in selected_rows(args):
                        if cached is None or cached["id"] != row["workflowId"]:
                            cached = await fetch_workflow(args, row)
                        workflow = dict(cached, planId=row["planId"])
                        batch_dir = model_dir / f"o{row['occurrence']}_{row['layout']}" / row["sampleId"]
                        batch_dir.mkdir(parents=True, exist_ok=True)
                        summary = await run_one(browser, args, workflow, row["repeatIndex"], batch_dir, index)
                        failed |= not summary["evaluation"]["submitted"]
                        report(batch_dir)
            finally:
                await browser.close()
    finally:
        # Preserve useful reports even when a later request or user interrupt stops the batch.
        for model_dir in args.runs_dir.glob(f"*/v1-plans/{batch_id}"):
            report(model_dir)
        print(f"Artifacts: {args.runs_dir} (v1-plans/{batch_id})", flush=True)
    return 1 if failed else 0
