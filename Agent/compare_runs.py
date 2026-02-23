import json
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_stack import get_model_name_for_path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------------
# Editable in-code configuration
# -----------------------------
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
SURVEY_VERSION = "survey_v1"
MODEL_NAME = get_model_name_for_path()
# "all" | "completion" | "unconstrained"
MODE_FILTER = "completion"
# Optional JSON export path. Keep empty to disable.
JSON_OUT = ""
# CSV export path. Keep empty to disable.
CSV_OUT = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_compare_runs.csv")
# Plot output controls
WRITE_PLOTS = True
PLOTS_DIR = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_compare_plots")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _common_prefix_len(a: List[str], b: List[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _sequence_ratio(a: List[str], b: List[str]) -> float:
    return SequenceMatcher(a=a, b=b).ratio() if (a or b) else 1.0


def _tri(v: Any) -> str:
    if v is True:
        return "Y"
    if v is False:
        return "N"
    return "-"


@dataclass
class RunRecord:
    mode: str
    run_name: str
    run_dir: Path
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    answer_eval: Dict[str, Any]


def _collect_runs(root: Path, survey_version: str, mode: str) -> List[RunRecord]:
    mode_dir = root / survey_version / MODEL_NAME / mode
    if not mode_dir.exists():
        return []

    records: List[RunRecord] = []
    for run_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()]):
        metrics = _load_json(run_dir / "run_metrics.json")
        summary = _load_json(run_dir / "run_summary.json")
        answer_eval = _load_json(run_dir / "answer_eval.json") or {}
        if not metrics or not summary:
            continue
        records.append(
            RunRecord(
                mode=mode,
                run_name=run_dir.name,
                run_dir=run_dir,
                metrics=metrics,
                summary=summary,
                answer_eval=answer_eval,
            )
        )
    return records


def _image_acc_text(answer_eval: Dict[str, Any]) -> str:
    known = int(answer_eval.get("image_known_count") or 0)
    correct = int(answer_eval.get("image_correct_count") or 0)
    acc = answer_eval.get("image_accuracy_known")
    if known == 0 and acc is None:
        return "-"
    if acc is None:
        return f"{correct}/{known}"
    return f"{correct}/{known}({float(acc):.2f})"


def _completion_text(section: Dict[str, Any]) -> str:
    answered = int(section.get("answered_count") or 0)
    total = int(section.get("total_count") or 0)
    rate = section.get("completion_rate")
    if total <= 0:
        return "-"
    if not isinstance(rate, (int, float)):
        return f"{answered}/{total}"
    return f"{answered}/{total}({float(rate):.2f})"


def _has_gt_trace(answer_eval: Dict[str, Any]) -> bool:
    source = answer_eval.get("ground_truth_source") or {}
    trace_path = source.get("ui_layout_trace_path")
    return isinstance(trace_path, str) and bool(trace_path.strip())


def _print_mode_report(mode: str, records: List[RunRecord]):
    print(f"\n=== Mode: {mode} ===")
    if not records:
        print("No runs found.")
        return

    baseline_order = records[0].metrics.get("answer_order", []) or []
    header = (
        "run",
        "text_comp",
        "text_attn",
        "img_comp",
        "img_acc",
        "img_attn",
        "captcha",
        "thankyou",
        "submitted",
        "exec",
        "uniq_ans",
        "gt_trace",
        "next",
        "submit",
        "ord_same",
        "ord_pref",
        "ord_jacc",
        "ord_seq",
    )
    print(" | ".join(header))
    print("-" * 170)

    for rec in records:
        order = rec.metrics.get("answer_order", []) or []
        same = order == baseline_order
        pref = _common_prefix_len(order, baseline_order)
        jacc = _jaccard(order, baseline_order)
        seq = _sequence_ratio(order, baseline_order)

        img_attn_ok = ((rec.answer_eval.get("image_attention") or {}).get("is_correct"))
        txt_attn_ok = ((rec.answer_eval.get("text_attention") or {}).get("is_correct"))
        captcha_ok = ((rec.answer_eval.get("captcha") or {}).get("is_correct"))
        text_comp = _completion_text(rec.answer_eval.get("text_page_completion") or {})
        image_comp = _completion_text(rec.answer_eval.get("image_page_completion") or {})

        row = (
            rec.run_name,
            text_comp,
            _tri(txt_attn_ok),
            image_comp,
            _image_acc_text(rec.answer_eval),
            _tri(img_attn_ok),
            _tri(captcha_ok),
            _tri(rec.metrics.get("reached_thank_you")),
            _tri(rec.metrics.get("submitted")),
            str(rec.metrics.get("total_exec_steps", 0)),
            str(rec.metrics.get("unique_answer_items_touched", 0)),
            _tri(_has_gt_trace(rec.answer_eval)),
            str(rec.metrics.get("next_click_count", 0)),
            str(rec.metrics.get("submit_click_count", 0)),
            _tri(same),
            str(pref),
            f"{jacc:.2f}",
            f"{seq:.2f}",
        )
        print(" | ".join(row))


def _build_json_report(records_by_mode: Dict[str, List[RunRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"modes": {}}
    for mode, records in records_by_mode.items():
        if not records:
            out["modes"][mode] = {"runs": []}
            continue
        baseline_order = records[0].metrics.get("answer_order", []) or []
        runs_payload = []
        for rec in records:
            order = rec.metrics.get("answer_order", []) or []
            runs_payload.append(
                {
                    "run_name": rec.run_name,
                    "run_dir": str(rec.run_dir),
                    "metrics": rec.metrics,
                    "answer_eval": rec.answer_eval,
                    "order_compare_to_baseline": {
                        "same_order": order == baseline_order,
                        "common_prefix_len": _common_prefix_len(order, baseline_order),
                        "jaccard_overlap": _jaccard(order, baseline_order),
                        "sequence_ratio": _sequence_ratio(order, baseline_order),
                    },
                }
            )
        out["modes"][mode] = {"baseline_run": records[0].run_name, "runs": runs_payload}
    return out


def _build_csv_rows(records_by_mode: Dict[str, List[RunRecord]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, records in records_by_mode.items():
        baseline_order = records[0].metrics.get("answer_order", []) or [] if records else []
        for rec in records:
            order = rec.metrics.get("answer_order", []) or []
            img_attn_ok = ((rec.answer_eval.get("image_attention") or {}).get("is_correct"))
            txt_attn_ok = ((rec.answer_eval.get("text_attention") or {}).get("is_correct"))
            captcha_ok = ((rec.answer_eval.get("captcha") or {}).get("is_correct"))
            text_comp = rec.answer_eval.get("text_page_completion") or {}
            image_comp = rec.answer_eval.get("image_page_completion") or {}
            img_known = int(rec.answer_eval.get("image_known_count") or 0)
            img_correct = int(rec.answer_eval.get("image_correct_count") or 0)
            rows.append(
                {
                    "mode": mode,
                    "run_name": rec.run_name,
                    "run_dir": str(rec.run_dir),
                    "text_completion_answered": int(text_comp.get("answered_count") or 0),
                    "text_completion_total": int(text_comp.get("total_count") or 0),
                    "text_completion_rate": text_comp.get("completion_rate"),
                    "text_attention_ok": txt_attn_ok,
                    "image_completion_answered": int(image_comp.get("answered_count") or 0),
                    "image_completion_total": int(image_comp.get("total_count") or 0),
                    "image_completion_rate": image_comp.get("completion_rate"),
                    "image_accuracy_correct": img_correct,
                    "image_accuracy_known_total": img_known,
                    "image_accuracy_rate": rec.answer_eval.get("image_accuracy_known"),
                    "image_attention_ok": img_attn_ok,
                    "captcha_ok": captcha_ok,
                    "reached_thank_you": rec.metrics.get("reached_thank_you"),
                    "submitted": rec.metrics.get("submitted"),
                    "total_exec_steps": rec.metrics.get("total_exec_steps", 0),
                    "unique_answer_items_touched": rec.metrics.get("unique_answer_items_touched", 0),
                    "next_click_count": rec.metrics.get("next_click_count", 0),
                    "submit_click_count": rec.metrics.get("submit_click_count", 0),
                    "order_same_as_baseline": order == baseline_order,
                    "order_common_prefix_len": _common_prefix_len(order, baseline_order),
                    "order_jaccard_overlap": _jaccard(order, baseline_order),
                    "order_sequence_ratio": _sequence_ratio(order, baseline_order),
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "run_name",
        "run_dir",
        "text_completion_answered",
        "text_completion_total",
        "text_completion_rate",
        "text_attention_ok",
        "image_completion_answered",
        "image_completion_total",
        "image_completion_rate",
        "image_accuracy_correct",
        "image_accuracy_known_total",
        "image_accuracy_rate",
        "image_attention_ok",
        "captcha_ok",
        "reached_thank_you",
        "submitted",
        "total_exec_steps",
        "unique_answer_items_touched",
        "next_click_count",
        "submit_click_count",
        "order_same_as_baseline",
        "order_common_prefix_len",
        "order_jaccard_overlap",
        "order_sequence_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_mode_table(mode: str, records: List[RunRecord], out_dir: Path) -> Optional[Path]:
    if plt is None:
        return None
    if not records:
        return None

    col_labels = [
        "text_completion",
        "text_attention_ok",
        "image_completion",
        "image_accuracy",
        "image_attention_ok",
        "captcha_ok",
        "reached_thank_you",
        "submitted",
    ]
    row_labels = [r.run_name for r in records]
    rows: List[List[str]] = []
    for rec in records:
        img_acc = rec.answer_eval.get("image_accuracy_known")
        img_acc_text = "-" if not isinstance(img_acc, (int, float)) else f"{float(img_acc):.2f}"
        text_comp = _completion_text(rec.answer_eval.get("text_page_completion") or {})
        image_comp = _completion_text(rec.answer_eval.get("image_page_completion") or {})
        img_attn_ok = (rec.answer_eval.get("image_attention") or {}).get("is_correct")
        txt_attn_ok = (rec.answer_eval.get("text_attention") or {}).get("is_correct")
        captcha_ok = (rec.answer_eval.get("captcha") or {}).get("is_correct")
        reached = rec.metrics.get("reached_thank_you")
        submitted = rec.metrics.get("submitted")
        rows.append(
            [
                text_comp,
                _tri(txt_attn_ok),
                image_comp,
                img_acc_text,
                _tri(img_attn_ok),
                _tri(captcha_ok),
                _tri(reached),
                _tri(submitted),
            ]
        )

    # Height scales with number of runs to keep table readable.
    fig_h = max(4.0, 1.2 + 0.45 * len(records))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    ax.set_title(f"Compare Runs - {mode} (Outcomes Table)", fontsize=14, fontweight="bold", pad=14)

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
    out_path = out_dir / f"{mode}_outcomes_table.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    if MODE_FILTER not in {"all", "completion", "unconstrained"}:
        raise SystemExit("MODE_FILTER must be one of: all, completion, unconstrained")

    root = Path(RUNS_ROOT)
    modes = ["completion", "unconstrained"] if MODE_FILTER == "all" else [MODE_FILTER]
    records_by_mode: Dict[str, List[RunRecord]] = {
        mode: _collect_runs(root=root, survey_version=SURVEY_VERSION, mode=mode)
        for mode in modes
    }

    print(f"runs_root={root}")
    print(f"survey_version={SURVEY_VERSION}")
    print(f"model_name={MODEL_NAME}")
    if WRITE_PLOTS and plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")
    plot_paths: List[Path] = []
    plot_dir = Path(PLOTS_DIR)
    for mode in modes:
        _print_mode_report(mode, records_by_mode[mode])
        if WRITE_PLOTS:
            p = _plot_mode_table(mode, records_by_mode[mode], out_dir=plot_dir)
            if p:
                plot_paths.append(p)

    if WRITE_PLOTS and plt is not None:
        if plot_paths:
            print(f"\nWrote comparison plots to: {plot_dir.resolve()}")
            for p in plot_paths:
                print(f"- {p.resolve()}")
        else:
            print("\nNo plots were written (no runs found).")

    if JSON_OUT:
        report = _build_json_report(records_by_mode)
        out_path = Path(JSON_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {out_path.resolve()}")

    if CSV_OUT:
        csv_rows = _build_csv_rows(records_by_mode)
        csv_path = Path(CSV_OUT)
        _write_csv(csv_path, csv_rows)
        print(f"\nWrote CSV report to: {csv_path.resolve()} ({len(csv_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
