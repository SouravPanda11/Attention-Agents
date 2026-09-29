"""Load each raw run once and preserve experiment boundaries."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

from evaluation import PROTOCOL_KEYS

AGENTIC = Path(__file__).resolve().parents[1]
GROUP_KEYS = tuple(key for key in PROTOCOL_KEYS if key != "model")
THEMES = ("consumer", "digital", "wellbeing", "education", "work", "finance", "civic", "lifestyle")
KINDS = ("single-radio", "single-checkbox", "multiple-checkbox", "single-dropdown", "likert",
         "image-single-select", "ranking", "short-text", "long-text", "numeric", "slider")
LABELS = {"single-radio": "Radio", "single-checkbox": "Single checkbox",
          "multiple-checkbox": "Multiple checkboxes", "single-dropdown": "Dropdown",
          "likert": "Likert", "image-single-select": "Image selection (DOM labels in DOM mode)",
          "ranking": "Ranking", "short-text": "Short text", "long-text": "Long text",
          "numeric": "Numeric", "slider": "Slider"}
COLORS = ["#197f86", "#c96935", "#5969b1", "#9c4773", "#718331", "#79583e"]


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (AGENTIC / path).resolve()


def load_groups(root, models=None, batch=None):
    groups = defaultdict(list)
    for path in sorted(root.rglob("run_summary.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if models and run["model"] not in models:
            continue
        if batch and path.parent.parent.name != batch:
            continue
        if run.get("status") == "running":
            continue
        if not run.get("question_count") or "evaluation" not in run:
            raise ValueError(f"Missing question counts or evaluation: {path}")
        run["_source"] = str(path)
        groups[tuple(run.get(key) for key in GROUP_KEYS)].append(run)
    if not groups:
        raise ValueError("No completed/interrupted run summaries match the selected directory and filters.")
    return [(dict(zip(GROUP_KEYS, key)), runs) for key, runs in groups.items()]


def group_id(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:10]


def by_model(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["model"]].append(run)
    return dict(sorted(groups.items()))


def accepted(run):
    return run["evaluation"].get("submitted") is True


def valid_count(run):
    return run["evaluation"].get("valid_count", 0) if accepted(run) else 0


def rate(runs):
    total = sum(r["question_count"] for r in runs)
    return 100 * sum(valid_count(r) for r in runs) / total if total else None


def summary_rows(runs):
    rows = []
    for model, items in by_model(runs).items():
        submitted = [r for r in items if accepted(r)]
        questions = sum(r["question_count"] for r in items)
        submitted_questions = sum(r["question_count"] for r in submitted)
        valid = sum(valid_count(r) for r in items)
        rows.append({"model": model, "runs": len(items), "submitted_runs": len(submitted),
                     "submission_pct": 100 * len(submitted) / len(items), "questions": questions,
                     "valid_answers": valid, "end_to_end_valid_pct": 100 * valid / questions,
                     "valid_pct_submitted": 100 * valid / submitted_questions if submitted_questions else None,
                     "skipped_submitted": sum(r["evaluation"].get("skipped_count", 0) for r in submitted),
                     "invalid_submitted": sum(r["evaluation"].get("invalid_count", 0) for r in submitted),
                     "unsubmitted_questions": questions - submitted_questions,
                     "fully_valid_runs": sum(valid_count(r) == r["question_count"] for r in submitted),
                     "model_calls": sum(r["model_calls"] for r in items),
                     "budget_exhausted_questions": sum(r.get("budget_exhausted_questions", 0) for r in items),
                     "mean_run_seconds": mean(r["wall_seconds"] for r in items),
                     "mean_model_seconds": mean(r["model_seconds"] for r in items),
                     "plan_errors": sum(r.get("plan_errors", 0) for r in items),
                     "invalid_answer_turns": sum(r.get("invalid_answer_turns", 0) for r in items),
                     "model_errors": sum(r.get("model_errors", 0) for r in items),
                     "action_errors": sum(r.get("action_errors", 0) for r in items)})
    return rows


def format_rows(runs):
    groups = defaultdict(list)
    for run in runs:
        if accepted(run):
            for item in run["evaluation"].get("question_results", []):
                groups[run["model"], item["kind"]].append(item)
    rows = []
    for (model, kind), items in sorted(groups.items()):
        counts = {status: sum(q["status"] == status for q in items) for status in ("valid", "invalid", "skipped")}
        rows.append({"model": model, "kind": kind, "submitted_items": len(items), **counts,
                     "valid_pct": 100 * counts["valid"] / len(items)})
    return rows


def theme_rows(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[run["model"], run.get("theme_id"), run.get("order_id")].append(run)
    return [{"model": model, "theme": theme, "order": order, "runs": len(items),
             "valid_answers": sum(valid_count(r) for r in items),
             "questions": sum(r["question_count"] for r in items), "valid_pct": rate(items)}
            for (model, theme, order), items in sorted(groups.items(), key=lambda pair: str(pair[0]))]


def attempt_rows(runs):
    groups = defaultdict(lambda: defaultdict(int))
    for run in runs:
        for q in run.get("question_results", []):
            key = (run["model"], q["turns"])
            groups[key]["answered" if q["status"] == "answered" else "unanswered"] += 1
            groups[key]["exhausted"] += q.get("reason") == "turn_budget_exhausted"
    return [{"model": model, "attempts": turns, "answered": counts["answered"],
             "unanswered": counts["unanswered"], "exhausted": counts["exhausted"]}
            for (model, turns), counts in sorted(groups.items())]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(AGENTIC / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 13,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    return plt


def finish(fig, out, name, note, formats, runs):
    first = runs[0]
    note += (f"\n{first.get('prompt_version', 'Unknown prompt')} · {first.get('observation')} · "
             f"{first.get('behavior')} · temperature {first.get('temperature')} · "
             f"{first.get('question_turn_limit')} attempts/question max · {len(runs)} runs")
    fig.text(0.02, 0.02, note, fontsize=9, color="#505969", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    for extension in formats:
        fig.savefig(out / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    pyplot().close(fig)


def parser(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--runs-dir", type=local_path, default=AGENTIC / "runs")
    cli.add_argument("--output-dir", type=local_path, help="Default: <runs-dir>/plots")
    cli.add_argument("--models", nargs="+", help="Exact model IDs to include")
    cli.add_argument("--batch", help="One batch directory name, such as 20260914T003048Z-ef770e57")
    cli.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    return cli


def standalone(draw, name):
    args = parser(f"Plot {name} from saved run summaries.").parse_args()
    groups = load_groups(args.runs_dir, args.models, args.batch)
    root = args.output_dir or args.runs_dir / "plots"
    for protocol, runs in groups:
        out = root if len(groups) == 1 else root / f"protocol-{group_id(protocol)}"
        out.mkdir(parents=True, exist_ok=True)
        draw(runs, out, args.formats)
        print(out / name)
