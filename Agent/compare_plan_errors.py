import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from model_stack import ensure_env_loaded, get_model_name_for_path


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as _plt

        return _plt
    except Exception:
        pass

    # Fallback: when script is run with system Python, try local AgenticAI env packages.
    local_site = Path(__file__).resolve().parent / "AgenticAI" / "Lib" / "site-packages"
    if local_site.exists():
        site_text = str(local_site)
        if site_text not in sys.path:
            sys.path.insert(0, site_text)
        try:
            import matplotlib.pyplot as _plt

            return _plt
        except Exception:
            return None
    return None


plt = _import_matplotlib_pyplot()


# -----------------------------
# Editable in-code configuration
# -----------------------------
ensure_env_loaded()
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
SURVEY_VERSION = os.getenv("SURVEY_VERSION", "").strip()
MODEL_NAME = get_model_name_for_path()
# "all" | "completion" | "unconstrained"
MODE_FILTER = "completion"

# Per-run CSV export path. Keep empty to disable.
CSV_OUT = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_plan_error_compare.csv")
# Per-plan_reject event CSV export path. Keep empty to disable.
# EVENTS_CSV_OUT = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_plan_error_events.csv")
EVENTS_CSV_OUT = ""


# Join this report with compare_runs CSV on run identity.
MERGE_WITH_COMPARE = False
COMPARE_CSV_IN = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_compare_runs.csv")
MERGED_CSV_OUT = str(
    Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_plan_error_joined_with_compare.csv"
)

# Plot output controls
WRITE_PLOTS = True
PLOTS_DIR = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_plan_error_plots")
# Use 0 to show all runs (same behavior style as compare_runs table).
MAX_RUN_ROWS_IN_PLOT = 0


CATEGORY_ORDER = [
    "planning_format",
    "planning_completeness",
    "planning_schema_key",
    "answer_option_mapping",
    "answer_value_constraints",
    "infra_or_transport",
    "other",
]

CATEGORY_TO_TYPE = {
    "planning_format": "planning",
    "planning_completeness": "planning",
    "planning_schema_key": "planning",
    "answer_option_mapping": "answer",
    "answer_value_constraints": "answer",
    "infra_or_transport": "infra",
    "other": "other",
}

CATEGORY_REGEX: List[Tuple[str, re.Pattern[str]]] = [
    (
        "planning_format",
        re.compile(
            r"did not return json array|model output must be a json array|got:\s*dict|got:\s*str|each step must be an object",
            re.IGNORECASE,
        ),
    ),
    (
        "planning_completeness",
        re.compile(
            r"requires answering all answerable items before navigation|requires exactly one final navigation click|next/submit to be the last step|missing keys:",
            re.IGNORECASE,
        ),
    ),
    (
        "planning_schema_key",
        re.compile(
            r"key not in action_space|selector not in action_space|missing/invalid key in step|unsupported interaction in action_space",
            re.IGNORECASE,
        ),
    ),
    (
        "answer_option_mapping",
        re.compile(r"select value must match action_space option|check value must match action_space option", re.IGNORECASE),
    ),
    (
        "answer_value_constraints",
        re.compile(
            r"missing/invalid value for fill|fill value for number field must be numeric|number value below min constraint|number value above max constraint|missing/invalid numeric value for set_range|select value cannot be placeholder/empty|check option missing selector",
            re.IGNORECASE,
        ),
    ),
    (
        "infra_or_transport",
        re.compile(r"\b400 bad request\b|\b500\b|timed out|connection|refused|chat/completions|http", re.IGNORECASE),
    ),
]


@dataclass
class RunErrorRecord:
    mode: str
    run_name: str
    run_dir: Path
    session_id: str
    plan_reject_count: int
    stalled: bool
    empty_plan_count: int
    category_counts: Dict[str, int]
    type_counts: Dict[str, int]
    dominant_category: str
    dominant_type: str
    first_error: str
    page_attempts: Dict[str, Any]
    sample_errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class EventRecord:
    mode: str
    run_name: str
    run_dir: Path
    seq: Optional[int]
    attempt: Optional[int]
    category: str
    failure_type: str
    error: str


def _classify_page_kind(url: str) -> Optional[str]:
    text = str(url or "").strip().lower()
    if not text:
        return None
    if "/text" in text:
        return "text"
    if "/image" in text:
        return "image"
    return None


def _extract_page_attempt_metrics(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    page_blocks: Dict[str, List[Dict[str, Any]]] = {"text": [], "image": []}
    current_block: Optional[Dict[str, Any]] = None
    current_page: Optional[str] = None

    def _finalize_current() -> None:
        nonlocal current_block, current_page
        if current_block is None or current_page not in page_blocks:
            current_block = None
            current_page = None
            return
        page_blocks[current_page].append(current_block)
        current_block = None
        current_page = None

    for ev in trace:
        if not isinstance(ev, dict):
            continue

        kind = str(ev.get("kind") or "")
        if kind == "page_marker" and str(ev.get("stage") or "") == "observe":
            _finalize_current()
            page = _classify_page_kind(str(ev.get("url") or ""))
            current_page = page
            if page is None:
                current_block = None
                continue
            current_block = {
                "reject_attempts": [],
                "success_attempt": None,
                "plan_sizes": [],
                "success": False,
            }
            continue

        if current_block is None:
            continue

        if kind == "plan_reject":
            attempt = ev.get("attempt")
            if isinstance(attempt, int) and attempt > 0:
                current_block["reject_attempts"].append(attempt)
        elif kind == "plan_recovered":
            attempt = ev.get("attempt")
            if isinstance(attempt, int) and attempt > 0:
                current_block["success_attempt"] = attempt
        elif kind == "plan":
            steps = ev.get("steps")
            size = len(steps) if isinstance(steps, list) else -1
            current_block["plan_sizes"].append(size)
            if size > 0:
                if not isinstance(current_block.get("success_attempt"), int):
                    reject_attempts = [a for a in current_block.get("reject_attempts", []) if isinstance(a, int) and a > 0]
                    current_block["success_attempt"] = (max(reject_attempts) + 1) if reject_attempts else 1
                current_block["success"] = True

    _finalize_current()

    out: Dict[str, Any] = {}
    for page in ("text", "image"):
        blocks = page_blocks[page]
        out[f"{page}_observe_blocks"] = len(blocks)

        success_idx: Optional[int] = None
        success_attempt: Optional[int] = None
        total_attempts_until_success = 0
        total_attempts_observed = 0

        for idx, block in enumerate(blocks, start=1):
            reject_attempts = [a for a in block.get("reject_attempts", []) if isinstance(a, int) and a > 0]
            block_success_attempt = block.get("success_attempt") if isinstance(block.get("success_attempt"), int) else None
            block_attempts = block_success_attempt if block_success_attempt is not None else (max(reject_attempts) if reject_attempts else 0)
            total_attempts_observed += int(block_attempts)

            if success_idx is None:
                total_attempts_until_success += int(block_attempts)
                if bool(block.get("success")) and block_success_attempt is not None:
                    success_idx = idx
                    success_attempt = int(block_success_attempt)

        out[f"{page}_success_observe_index"] = success_idx
        out[f"{page}_success_attempt"] = success_attempt
        out[f"{page}_total_attempts_until_success"] = total_attempts_until_success if success_idx is not None else None
        out[f"{page}_total_attempts_observed"] = total_attempts_observed
        out[f"{page}_success_found"] = success_idx is not None

    return out


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"run_(\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def _default_csv_out(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_plan_error_compare.csv")


def _default_events_csv_out(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_plan_error_events.csv")


def _default_merged_csv_out(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_plan_error_joined_with_compare.csv")


def _default_plots_dir(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_plan_error_plots")


def _default_compare_csv(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_compare_runs.csv")


def _count_mode_runs(model_dir: Path, mode: str) -> int:
    mode_dir = model_dir / mode
    if not mode_dir.exists():
        return 0
    count = 0
    for run_dir in mode_dir.iterdir():
        if run_dir.is_dir() and (run_dir / "trace.json").exists():
            count += 1
    return count


def _resolve_model_name(root: Path, survey_version: str, configured_model_name: str, modes: List[str]) -> str:
    survey_dir = root / survey_version
    if not survey_dir.exists():
        return configured_model_name
    candidates = [p for p in survey_dir.iterdir() if p.is_dir()]
    if not candidates:
        return configured_model_name

    def _score(model_dir: Path) -> int:
        return sum(_count_mode_runs(model_dir, mode) for mode in modes)

    configured_dir = survey_dir / configured_model_name
    configured_score = _score(configured_dir) if configured_model_name and configured_dir.exists() else -1
    if configured_score > 0:
        return configured_model_name

    scored = [(p, _score(p)) for p in candidates]
    scored_with_runs = [item for item in scored if item[1] > 0]
    if scored_with_runs:
        chosen = sorted(scored_with_runs, key=lambda item: (item[1], item[0].stat().st_mtime), reverse=True)[0][0]
        return chosen.name

    chosen = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return chosen.name


def _classify_error(error: str) -> str:
    text = str(error or "")
    for category, regex in CATEGORY_REGEX:
        if regex.search(text):
            return category
    return "other"


def _normalize_path_text(text: str) -> str:
    return str(text or "").strip().replace("\\", "/").lower()


def _collect_mode_records(
    root: Path,
    survey_version: str,
    model_name: str,
    mode: str,
) -> Tuple[List[RunErrorRecord], List[EventRecord]]:
    mode_dir = root / survey_version / model_name / mode
    if not mode_dir.exists():
        return [], []

    run_records: List[RunErrorRecord] = []
    event_records: List[EventRecord] = []
    for run_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()], key=_run_sort_key):
        trace = _load_json(run_dir / "trace.json")
        if not isinstance(trace, list):
            continue

        summary = _load_json(run_dir / "run_summary.json")
        session_id = ""
        if isinstance(summary, dict):
            session_id = str(summary.get("session_id") or "").strip()

        category_counts = Counter({name: 0 for name in CATEGORY_ORDER})
        type_counts = Counter({"planning": 0, "answer": 0, "infra": 0, "other": 0})
        sample_errors: Dict[str, str] = {}
        first_error = ""
        stalled = False
        empty_plan_count = 0
        page_attempts = _extract_page_attempt_metrics(trace)

        for ev in trace:
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("kind") or "")
            if kind == "plan_stalled":
                stalled = True
            elif kind == "plan":
                steps = ev.get("steps")
                if isinstance(steps, list) and len(steps) == 0:
                    empty_plan_count += 1
            elif kind == "plan_reject":
                err = str(ev.get("error") or "")
                if not first_error:
                    first_error = err
                category = _classify_error(err)
                category_counts[category] += 1
                failure_type = CATEGORY_TO_TYPE.get(category, "other")
                type_counts[failure_type] += 1
                if category not in sample_errors:
                    sample_errors[category] = err[:220].replace("\n", " ")
                event_records.append(
                    EventRecord(
                        mode=mode,
                        run_name=run_dir.name,
                        run_dir=run_dir,
                        seq=ev.get("seq") if isinstance(ev.get("seq"), int) else None,
                        attempt=ev.get("attempt") if isinstance(ev.get("attempt"), int) else None,
                        category=category,
                        failure_type=failure_type,
                        error=err,
                    )
                )

        plan_reject_count = sum(category_counts.values())
        dominant_category = ""
        if plan_reject_count > 0:
            dominant_category = sorted(
                [(k, v) for k, v in category_counts.items() if v > 0],
                key=lambda item: (item[1], CATEGORY_ORDER.index(item[0]) if item[0] in CATEGORY_ORDER else 999),
                reverse=True,
            )[0][0]
        dominant_type = ""
        type_total = sum(type_counts.values())
        if type_total > 0:
            dominant_type = sorted(type_counts.items(), key=lambda item: item[1], reverse=True)[0][0]

        run_records.append(
            RunErrorRecord(
                mode=mode,
                run_name=run_dir.name,
                run_dir=run_dir,
                session_id=session_id,
                plan_reject_count=plan_reject_count,
                stalled=stalled,
                empty_plan_count=empty_plan_count,
                category_counts=dict(category_counts),
                type_counts=dict(type_counts),
                dominant_category=dominant_category,
                dominant_type=dominant_type,
                first_error=first_error,
                page_attempts=page_attempts,
                sample_errors=sample_errors,
            )
        )

    return run_records, event_records


def _build_run_rows(records_by_mode: Dict[str, List[RunErrorRecord]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, records in records_by_mode.items():
        for rec in records:
            row: Dict[str, Any] = {
                "mode": mode,
                "run_name": rec.run_name,
                "run_dir": str(rec.run_dir),
                "session_id": rec.session_id,
                "plan_reject_count": rec.plan_reject_count,
                "planning_failure_count": rec.type_counts.get("planning", 0),
                "answer_failure_count": rec.type_counts.get("answer", 0),
                "infra_failure_count": rec.type_counts.get("infra", 0),
                "other_failure_count": rec.type_counts.get("other", 0),
                "dominant_category": rec.dominant_category,
                "dominant_failure_type": rec.dominant_type,
                "stalled": rec.stalled,
                "empty_plan_count": rec.empty_plan_count,
                "first_error": rec.first_error,
                "text_success_attempt": rec.page_attempts.get("text_success_attempt"),
                "text_success_observe_index": rec.page_attempts.get("text_success_observe_index"),
                "text_total_attempts_until_success": rec.page_attempts.get("text_total_attempts_until_success"),
                "text_total_attempts_observed": rec.page_attempts.get("text_total_attempts_observed"),
                "text_observe_blocks": rec.page_attempts.get("text_observe_blocks"),
                "text_success_found": rec.page_attempts.get("text_success_found"),
                "image_success_attempt": rec.page_attempts.get("image_success_attempt"),
                "image_success_observe_index": rec.page_attempts.get("image_success_observe_index"),
                "image_total_attempts_until_success": rec.page_attempts.get("image_total_attempts_until_success"),
                "image_total_attempts_observed": rec.page_attempts.get("image_total_attempts_observed"),
                "image_observe_blocks": rec.page_attempts.get("image_observe_blocks"),
                "image_success_found": rec.page_attempts.get("image_success_found"),
            }
            for category in CATEGORY_ORDER:
                row[f"category_{category}"] = rec.category_counts.get(category, 0)
                row[f"sample_{category}"] = rec.sample_errors.get(category, "")
            rows.append(row)
    return rows


def _build_event_rows(event_records_by_mode: Dict[str, List[EventRecord]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, events in event_records_by_mode.items():
        for ev in events:
            rows.append(
                {
                    "mode": mode,
                    "run_name": ev.run_name,
                    "run_dir": str(ev.run_dir),
                    "seq": ev.seq,
                    "attempt": ev.attempt,
                    "category": ev.category,
                    "failure_type": ev.failure_type,
                    "error": ev.error,
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_filename_for_mode(base_name: str, mode: str) -> str:
    if base_name == "_plan_error_compare.csv":
        if mode == "completion":
            return "completed_plan_error_compare.csv"
        if mode == "unconstrained":
            return "unconstrained_plan_error_compare.csv"
    if base_name == "_plan_error_events.csv":
        if mode == "completion":
            return "completed_plan_error_events.csv"
        if mode == "unconstrained":
            return "unconstrained_plan_error_events.csv"
    if base_name == "_plan_error_joined_with_compare.csv":
        if mode == "completion":
            return "completed_plan_error_joined_with_compare.csv"
        if mode == "unconstrained":
            return "unconstrained_plan_error_joined_with_compare.csv"
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".csv"
    return f"{stem}_{mode}{suffix}"


def _csv_path_for_mode(base_path: Path, mode: str) -> Path:
    if base_path.name in {
        "_plan_error_compare.csv",
        "_plan_error_events.csv",
        "_plan_error_joined_with_compare.csv",
    }:
        return base_path.with_name(_csv_filename_for_mode(base_path.name, mode))
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{mode}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}_{mode}")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def _compare_csv_filename_for_mode(base_name: str, mode: str) -> str:
    if base_name == "_compare_runs.csv":
        if mode == "completion":
            return "completed_compared_runs.csv"
        if mode == "unconstrained":
            return "unconstrained_compare_runs.csv"
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".csv"
    return f"{stem}_{mode}{suffix}"


def _compare_csv_path_for_mode(base_path: Path, mode: str) -> Path:
    if base_path.name == "_compare_runs.csv":
        return base_path.with_name(_compare_csv_filename_for_mode(base_path.name, mode))
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{mode}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}_{mode}")


def _merge_with_compare(run_rows: List[Dict[str, Any]], compare_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_run_dir: Dict[str, Dict[str, Any]] = {}
    by_mode_run: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in compare_rows:
        run_dir_key = _normalize_path_text(row.get("run_dir", ""))
        if run_dir_key:
            by_run_dir[run_dir_key] = row
        mode_key = str(row.get("mode") or "").strip().lower()
        run_name_key = str(row.get("run_name") or "").strip().lower()
        if mode_key and run_name_key:
            by_mode_run[(mode_key, run_name_key)] = row

    merged: List[Dict[str, Any]] = []
    for row in run_rows:
        run_dir_key = _normalize_path_text(row.get("run_dir", ""))
        mode_key = str(row.get("mode") or "").strip().lower()
        run_name_key = str(row.get("run_name") or "").strip().lower()
        cmp = by_run_dir.get(run_dir_key)
        if cmp is None:
            cmp = by_mode_run.get((mode_key, run_name_key))

        out = dict(row)
        if cmp:
            out["compare_match_found"] = True
            for key, value in cmp.items():
                out[f"cmp_{key}"] = value
        else:
            out["compare_match_found"] = False
        merged.append(out)
    return merged


def _plot_mode_table(
    mode: str,
    records: List[RunErrorRecord],
    out_dir: Path,
    survey_version: str,
    model_name: str,
) -> Optional[Path]:
    if plt is None or not records:
        return None

    ordered = sorted(records, key=lambda r: r.plan_reject_count, reverse=True)
    rows_to_show = ordered if MAX_RUN_ROWS_IN_PLOT <= 0 else ordered[:MAX_RUN_ROWS_IN_PLOT]
    col_labels = [
        "plan_rejects",
        "planning",
        "answer",
        "infra",
        "dominant",
        "text_try",
        "image_try",
        "stalled",
        "empty_plans",
    ]
    row_labels = [r.run_name for r in rows_to_show]
    rows: List[List[str]] = []
    for rec in rows_to_show:
        rows.append(
            [
                str(rec.plan_reject_count),
                str(rec.type_counts.get("planning", 0)),
                str(rec.type_counts.get("answer", 0)),
                str(rec.type_counts.get("infra", 0)),
                rec.dominant_category or "-",
                str(rec.page_attempts.get("text_success_attempt")) if rec.page_attempts.get("text_success_attempt") is not None else "-",
                str(rec.page_attempts.get("image_success_attempt")) if rec.page_attempts.get("image_success_attempt") is not None else "-",
                "Y" if rec.stalled else "N",
                str(rec.empty_plan_count),
            ]
        )

    fig_h = max(4.0, 1.2 + 0.45 * len(rows_to_show))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    title = f"Plan Rejects - {survey_version} - {model_name} - {mode}"
    if MAX_RUN_ROWS_IN_PLOT > 0 and len(ordered) > len(rows_to_show):
        title += f" (top {len(rows_to_show)} of {len(ordered)})"
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_plan_error_table.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_mode_category_totals(
    mode: str,
    records: List[RunErrorRecord],
    out_dir: Path,
    survey_version: str,
    model_name: str,
) -> Optional[Path]:
    if plt is None or not records:
        return None

    totals = Counter()
    for rec in records:
        totals.update(rec.category_counts)

    categories = [c for c in CATEGORY_ORDER if totals.get(c, 0) > 0]
    if not categories:
        return None

    values = [int(totals.get(c, 0)) for c in categories]
    fig_h = max(4.0, 0.6 + 0.55 * len(categories))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(categories, values)
    ax.set_xlabel("plan_reject count")
    ax.set_title(f"Plan Reject Categories - {survey_version} - {model_name} - {mode}", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for idx, v in enumerate(values):
        ax.text(v + 0.5, idx, str(v), va="center", fontsize=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_plan_error_category_totals.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _print_mode_summary(mode: str, records: List[RunErrorRecord]) -> None:
    print(f"\n=== Mode: {mode} ===")
    if not records:
        print("No runs found.")
        return
    run_count = len(records)
    plan_reject_total = sum(r.plan_reject_count for r in records)
    type_totals = Counter()
    category_totals = Counter()
    stalled_count = 0
    for rec in records:
        type_totals.update(rec.type_counts)
        category_totals.update(rec.category_counts)
        if rec.stalled:
            stalled_count += 1

    print(f"runs={run_count} | total_plan_rejects={plan_reject_total} | stalled_runs={stalled_count}")
    print(
        "failure_type_counts:"
        f" planning={type_totals.get('planning', 0)}"
        f", answer={type_totals.get('answer', 0)}"
        f", infra={type_totals.get('infra', 0)}"
        f", other={type_totals.get('other', 0)}"
    )
    top_categories = sorted(
        [(k, v) for k, v in category_totals.items() if v > 0],
        key=lambda item: item[1],
        reverse=True,
    )[:6]
    if top_categories:
        parts = [f"{k}={v}" for k, v in top_categories]
        print("top_categories: " + ", ".join(parts))

    def _attempt_stats(page: str) -> str:
        vals = [r.page_attempts.get(f"{page}_success_attempt") for r in records]
        vals = [int(v) for v in vals if isinstance(v, int) and v > 0]
        success_found = sum(1 for r in records if bool(r.page_attempts.get(f"{page}_success_found")))
        if not vals:
            return f"{page}: success_runs={success_found}/{len(records)}, avg_attempt=-"
        avg = sum(vals) / len(vals)
        return f"{page}: success_runs={success_found}/{len(records)}, avg_attempt={avg:.2f}"

    print("plan_success_attempts: " + _attempt_stats("text") + " | " + _attempt_stats("image"))


def main() -> int:
    if MODE_FILTER not in {"all", "completion", "unconstrained"}:
        raise SystemExit("MODE_FILTER must be one of: all, completion, unconstrained")
    if not SURVEY_VERSION:
        raise SystemExit("SURVEY_VERSION must be set in environment (for example in Agent/.env).")

    root = Path(RUNS_ROOT)
    modes = ["completion", "unconstrained"] if MODE_FILTER == "all" else [MODE_FILTER]
    resolved_model_name = _resolve_model_name(
        root=root,
        survey_version=SURVEY_VERSION,
        configured_model_name=MODEL_NAME,
        modes=modes,
    )

    records_by_mode: Dict[str, List[RunErrorRecord]] = {}
    events_by_mode: Dict[str, List[EventRecord]] = {}
    for mode in modes:
        run_records, event_records = _collect_mode_records(
            root=root,
            survey_version=SURVEY_VERSION,
            model_name=resolved_model_name,
            mode=mode,
        )
        records_by_mode[mode] = run_records
        events_by_mode[mode] = event_records

    print(f"runs_root={root}")
    print(f"survey_version={SURVEY_VERSION}")
    print(f"model_name_configured={MODEL_NAME}")
    print(f"model_name_resolved={resolved_model_name}")
    if resolved_model_name != MODEL_NAME:
        print("note: configured model has no valid runs for this survey_version/mode; using resolved model folder.")
    if WRITE_PLOTS and plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")

    for mode in modes:
        _print_mode_summary(mode, records_by_mode[mode])

    configured_csv_default = _default_csv_out(SURVEY_VERSION, MODEL_NAME)
    effective_csv_out = _default_csv_out(SURVEY_VERSION, resolved_model_name) if CSV_OUT == configured_csv_default else CSV_OUT
    if effective_csv_out:
        base_path = Path(effective_csv_out)
        if len(modes) > 1:
            print("\nWrote per-run plan-error CSVs:")
            for mode in modes:
                rows = _build_run_rows({mode: records_by_mode[mode]})
                mode_path = _csv_path_for_mode(base_path, mode)
                _write_csv(mode_path, rows)
                print(f"- {mode}: {mode_path.resolve()} ({len(rows)} rows)")
        else:
            rows = _build_run_rows(records_by_mode)
            out_path = _csv_path_for_mode(base_path, modes[0]) if base_path.name == "_plan_error_compare.csv" else base_path
            _write_csv(out_path, rows)
            print(f"\nWrote per-run plan-error CSV: {out_path.resolve()} ({len(rows)} rows)")

    configured_events_default = _default_events_csv_out(SURVEY_VERSION, MODEL_NAME)
    effective_events_out = (
        _default_events_csv_out(SURVEY_VERSION, resolved_model_name)
        if EVENTS_CSV_OUT == configured_events_default
        else EVENTS_CSV_OUT
    )
    if effective_events_out:
        events_base = Path(effective_events_out)
        if len(modes) > 1:
            print("\nWrote event-level plan-error CSVs:")
            for mode in modes:
                rows = _build_event_rows({mode: events_by_mode[mode]})
                mode_path = _csv_path_for_mode(events_base, mode)
                _write_csv(mode_path, rows)
                print(f"- {mode}: {mode_path.resolve()} ({len(rows)} rows)")
        else:
            rows = _build_event_rows(events_by_mode)
            out_path = _csv_path_for_mode(events_base, modes[0]) if events_base.name == "_plan_error_events.csv" else events_base
            _write_csv(out_path, rows)
            print(f"\nWrote event-level plan-error CSV: {out_path.resolve()} ({len(rows)} rows)")

    if MERGE_WITH_COMPARE:
        configured_compare_default = _default_compare_csv(SURVEY_VERSION, MODEL_NAME)
        effective_compare_csv = (
            _default_compare_csv(SURVEY_VERSION, resolved_model_name)
            if COMPARE_CSV_IN == configured_compare_default
            else COMPARE_CSV_IN
        )
        configured_merged_default = _default_merged_csv_out(SURVEY_VERSION, MODEL_NAME)
        effective_merged_out = (
            _default_merged_csv_out(SURVEY_VERSION, resolved_model_name)
            if MERGED_CSV_OUT == configured_merged_default
            else MERGED_CSV_OUT
        )
        merged_base = Path(effective_merged_out)
        compare_base = Path(effective_compare_csv)

        if len(modes) > 1:
            print("\nJoined with compare_runs CSV:")
            for mode in modes:
                run_rows = _build_run_rows({mode: records_by_mode[mode]})
                compare_path = _compare_csv_path_for_mode(compare_base, mode)
                compare_rows = _read_csv_rows(compare_path)
                merged_rows = _merge_with_compare(run_rows, compare_rows)
                merged_path = _csv_path_for_mode(merged_base, mode)
                _write_csv(merged_path, merged_rows)
                print(
                    f"- {mode}: {merged_path.resolve()} ({len(merged_rows)} rows, compare_rows={len(compare_rows)})"
                )
        else:
            mode = modes[0]
            run_rows = _build_run_rows(records_by_mode)
            compare_path = _compare_csv_path_for_mode(compare_base, mode) if compare_base.name == "_compare_runs.csv" else compare_base
            compare_rows = _read_csv_rows(compare_path)
            merged_rows = _merge_with_compare(run_rows, compare_rows)
            merged_path = _csv_path_for_mode(merged_base, mode) if merged_base.name == "_plan_error_joined_with_compare.csv" else merged_base
            _write_csv(merged_path, merged_rows)
            print(
                f"\nJoined with compare_runs CSV: {merged_path.resolve()} ({len(merged_rows)} rows, compare_rows={len(compare_rows)})"
            )

    plot_paths: List[Path] = []
    configured_plot_default = _default_plots_dir(SURVEY_VERSION, MODEL_NAME)
    effective_plot_dir = _default_plots_dir(SURVEY_VERSION, resolved_model_name) if PLOTS_DIR == configured_plot_default else PLOTS_DIR
    plot_dir = Path(effective_plot_dir)
    for mode in modes:
        if not WRITE_PLOTS:
            continue
        p1 = _plot_mode_table(
            mode=mode,
            records=records_by_mode[mode],
            out_dir=plot_dir,
            survey_version=SURVEY_VERSION,
            model_name=resolved_model_name,
        )
        if p1:
            plot_paths.append(p1)
        p2 = _plot_mode_category_totals(
            mode=mode,
            records=records_by_mode[mode],
            out_dir=plot_dir,
            survey_version=SURVEY_VERSION,
            model_name=resolved_model_name,
        )
        if p2:
            plot_paths.append(p2)

    if WRITE_PLOTS and plt is not None:
        if plot_paths:
            print(f"\nWrote plan-error plots to: {plot_dir.resolve()}")
            for path in plot_paths:
                print(f"- {path.resolve()}")
        else:
            print("\nNo plan-error plots were written (no runs found).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
