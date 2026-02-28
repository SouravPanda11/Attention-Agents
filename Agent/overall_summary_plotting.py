import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from model_stack import ensure_env_loaded

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:
    plt = None
    Patch = None


COMPARE_FILES = {
    "completion": "completed_compared_runs.csv",
    "unconstrained": "unconstrained_compare_runs.csv",
}

PLAN_ERROR_FILES = {
    "completion": "completed_plan_error_compare.csv",
    "unconstrained": "unconstrained_plan_error_compare.csv",
}

GROUP_SPACING = 0.4
PAIR_OFFSET = 0.07
BAR_THICKNESS = 0.1
TITLE_FONT_SIZE = 19
AXIS_LABEL_FONT_SIZE = 14*1.3
TICK_LABEL_FONT_SIZE = 12*1.3
LEGEND_FONT_SIZE = 12*1.3
BAR_LABEL_FONT_SIZE = 12*1.3
FAIL_COUNT_LABEL_FONT_SIZE = 9*1.3
SEGMENT_LABEL_FONT_SIZE = 10*1.3
FIGSIZE_STANDARD = (12, 7)

# -----------------------------
# Editable in-code configuration
# -----------------------------
ensure_env_loaded()
# Survey version folder under Agent/runs (loaded from Agent/.env).
SURVEY_VERSION = os.getenv("SURVEY_VERSION", "").strip() or "survey_v0"
# Root runs directory (default: Agent/runs)
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
# Output directory. Keep empty to use:
# Agent/runs/<SURVEY_VERSION>/_overall_summary_plots
OUTPUT_DIR = ""
# Optional model folder names to include.
# Keep empty list to auto-discover all models with required CSV files.
MODELS_FILTER: List[str] = []


def _parse_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_int(value: object, default: int = 0) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def _discover_models(survey_dir: Path) -> List[Path]:
    if not survey_dir.exists():
        return []
    found: List[Path] = []
    for model_dir in sorted([p for p in survey_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        # Include models if completion CSVs are present; unconstrained is optional
        # for VLM-only runs.
        has_minimum = (model_dir / COMPARE_FILES["completion"]).exists() and (
            model_dir / PLAN_ERROR_FILES["completion"]
        ).exists()
        if has_minimum:
            found.append(model_dir)
    return found


def _short_model_label(model_name: str) -> str:
    mapping = {
        "meta-llama-3-8b-instruct": "llama3-8b",
        "mistral-7b-instruct-v0.3": "mistral-7b",
        "qwen2.5-7b-instruct-1m": "qwen2.5-7b",
        "qwen2.5-vl-7b-instruct": "qwen2.5-vl-7b",
        "llava-1.6-mistral-7b": "llava-1.6-7b",
        "minicpm-v-2_6": "minicpm-v-2.6",
    }
    return mapping.get(model_name, model_name)


def _is_vlm_model_name(model_name: str) -> bool:
    text = model_name.lower()
    return ("-vl-" in text) or text.startswith("llava") or ("minicpm" in text)


def _split_model_dirs(selected: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {"llm": [], "vlm": []}
    for model_dir in selected:
        if _is_vlm_model_name(model_dir.name):
            groups["vlm"].append(model_dir)
        else:
            groups["llm"].append(model_dir)
    return groups


def _successful_runs(compare_csv: Path) -> Tuple[int, int]:
    rows = _read_csv_rows(compare_csv)
    success = 0
    for row in rows:
        submitted = _parse_bool(row.get("submitted"))
        if submitted is None:
            submitted = _parse_bool(row.get("overall_ok"))
        if submitted is True:
            success += 1
    return success, len(rows)


def _failed_run_reason_counts(compare_csv: Path, plan_error_csv: Path) -> Tuple[int, int]:
    compare_rows = _read_csv_rows(compare_csv)
    failed_runs = set()
    for row in compare_rows:
        submitted = _parse_bool(row.get("submitted"))
        if submitted is None:
            submitted = _parse_bool(row.get("overall_ok"))
        if submitted is True:
            continue
        run_name = str(row.get("run_name") or "").strip()
        if run_name:
            failed_runs.add(run_name)

    plan_rows = _read_csv_rows(plan_error_csv)
    per_run: Dict[str, Tuple[str, int, int]] = {}
    for row in plan_rows:
        run_name = str(row.get("run_name") or "").strip()
        if not run_name:
            continue
        dominant = str(row.get("dominant_failure_type") or "").strip().lower()
        p_count = _parse_int(row.get("planning_failure_count"), 0)
        a_count = _parse_int(row.get("answer_failure_count"), 0)
        per_run[run_name] = (dominant, p_count, a_count)

    planning = 0
    answer = 0
    for run_name in failed_runs:
        dominant, p_count, a_count = per_run.get(run_name, ("", 0, 0))
        if dominant == "planning":
            planning += 1
        elif dominant == "answer":
            answer += 1
        else:
            # Fallback to higher per-run failure count when dominant type is missing/non-binary.
            if p_count > a_count:
                planning += 1
            elif a_count > p_count:
                answer += 1
            else:
                planning += 1
    return planning, answer


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo * 100.0, hi * 100.0


def _fmt_pct(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return f"{int(rounded)}%"
    return f"{value:.1f}%"


def _plot_success_counts(
    labels: List[str],
    summary: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    _draw_success_counts(ax, labels, summary, survey_version, modes=modes)
    # Force layout order: title, legend (outside axes), gap, plot.
    legend = ax.get_legend()
    legend_handles = []
    legend_labels = []
    if legend is not None:
        legend_handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        legend_labels = [txt.get_text() for txt in legend.get_texts()]
        legend.remove()
    ax.set_title("")
    fig.suptitle(f"Successful Submissions Out of 100 Runs ({survey_version})", fontsize=TITLE_FONT_SIZE, y=0.95)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.9),
            ncol=max(1, len(legend_labels)),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.84)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_success_counts(
    ax,
    labels: List[str],
    summary: Dict[str, Dict[str, Tuple[int, int]]],
    survey_version: str,
    orientation: str = "vertical",
    show_y_axis: bool = True,
    modes: Optional[List[str]] = None,
) -> None:
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if modes is None:
        modes = ["completion", "unconstrained"]

    pos = [i * GROUP_SPACING for i in range(len(labels))]
    if len(modes) == 1:
        offsets = [0.0]
        bar_size = BAR_THICKNESS * 1.4
    else:
        offsets = [PAIR_OFFSET * (2 * i - (len(modes) - 1)) for i in range(len(modes))]
        bar_size = BAR_THICKNESS
    mode_colors = {"completion": "#4C78A8", "unconstrained": "#F58518"}
    vals_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    err_lo_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    err_hi_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    for label in labels:
        for mode in modes:
            success, total = summary[label][mode]
            pct = (100.0 * success / total) if total > 0 else 0.0
            lo, hi = _wilson_ci(success, total)
            vals_by_mode[mode].append(pct)
            err_lo_by_mode[mode].append(max(0.0, pct - lo))
            err_hi_by_mode[mode].append(max(0.0, hi - pct))

    bars_by_mode = {}
    if orientation == "vertical":
        for mode, offset in zip(modes, offsets):
            bars_by_mode[mode] = ax.bar(
                [p + offset for p in pos],
                vals_by_mode[mode],
                width=bar_size,
                label=mode,
                color=mode_colors[mode],
                yerr=[err_lo_by_mode[mode], err_hi_by_mode[mode]],
                capsize=4,
                error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
            )
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Successful runs (%)")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 102)
    else:
        for mode, offset in zip(modes, offsets):
            bars_by_mode[mode] = ax.barh(
                [p + offset for p in pos],
                vals_by_mode[mode],
                height=bar_size,
                label=mode,
                color=mode_colors[mode],
                xerr=[err_lo_by_mode[mode], err_hi_by_mode[mode]],
                capsize=4,
                error_kw={"elinewidth": 1.0, "ecolor": "#222222"},
            )
        ax.set_yticks(pos)
        ax.set_yticklabels(labels if show_y_axis else [])
        ax.set_xlabel("Successful runs (%)")
        ax.set_ylabel("Model" if show_y_axis else "")
        ax.set_xlim(0, 102)
        if not show_y_axis:
            ax.tick_params(axis="y", left=False, labelleft=False)

    ax.set_title(f"Successful Submissions Out of 100 Runs ({survey_version})", fontsize=TITLE_FONT_SIZE, pad=18)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y" if orientation == "vertical" else "x", linestyle="--", alpha=0.35)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=max(1, len(modes)),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    for bars in bars_by_mode.values():
        for bar in bars:
            val = float(bar.get_height())
            if orientation == "vertical":
                y = max(1.0, val * 0.5)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    _fmt_pct(val),
                    ha="center",
                    va="center",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    color="white",
                    fontweight="bold",
                )
            else:
                val = float(bar.get_width())
                x_center = max(2.5, val * 0.5)
                ax.text(
                    x_center,
                    bar.get_y() + bar.get_height() / 2,
                    _fmt_pct(val),
                    ha="center",
                    va="center",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    color="white",
                    fontweight="bold",
                )


def _plot_error_breakdown(
    labels: List[str],
    error_totals: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None or Patch is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    _draw_error_breakdown(ax, labels, error_totals, survey_version, modes=modes)
    # Force layout order: title, legend (outside axes), gap, plot.
    legend = ax.get_legend()
    legend_handles = []
    legend_labels = []
    if legend is not None:
        legend_handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        legend_labels = [txt.get_text() for txt in legend.get_texts()]
        legend.remove()
    ax.set_title("")
    fig.suptitle(f"Failure-Reason Breakdown by Failed Runs ({survey_version})", fontsize=TITLE_FONT_SIZE, y=0.95)
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.9),
            ncol=2,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.84)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _draw_error_breakdown(
    ax,
    labels: List[str],
    error_totals: Dict[str, Dict[str, Tuple[int, int]]],
    survey_version: str,
    orientation: str = "vertical",
    show_y_axis: bool = True,
    modes: Optional[List[str]] = None,
) -> None:
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    if modes is None:
        modes = ["completion", "unconstrained"]

    pos = [i * GROUP_SPACING for i in range(len(labels))]
    if len(modes) == 1:
        offsets = [0.0]
        bar_size = BAR_THICKNESS * 1.4
    else:
        offsets = [PAIR_OFFSET * (2 * i - (len(modes) - 1)) for i in range(len(modes))]
        bar_size = BAR_THICKNESS

    planning_color = "#2A9D8F"
    answer_color = "#E76F51"

    plan_counts_by_mode: Dict[str, List[int]] = {mode: [] for mode in modes}
    answer_counts_by_mode: Dict[str, List[int]] = {mode: [] for mode in modes}
    plan_pcts_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    answer_pcts_by_mode: Dict[str, List[float]] = {mode: [] for mode in modes}
    total_counts = []

    for label in labels:
        for mode in modes:
            p_count, a_count = error_totals[label][mode]
            total = p_count + a_count
            plan_counts_by_mode[mode].append(p_count)
            answer_counts_by_mode[mode].append(a_count)
            plan_pcts_by_mode[mode].append((100.0 * p_count / total) if total > 0 else 0.0)
            answer_pcts_by_mode[mode].append((100.0 * a_count / total) if total > 0 else 0.0)
            total_counts.append(total)

    bars_plan_by_mode = {}
    bars_answer_by_mode = {}
    if orientation == "vertical":
        for mode, offset in zip(modes, offsets):
            mode_pos = [p + offset for p in pos]
            bars_plan_by_mode[mode] = ax.bar(mode_pos, plan_counts_by_mode[mode], width=bar_size, color=planning_color)
            bars_answer_by_mode[mode] = ax.bar(
                mode_pos,
                answer_counts_by_mode[mode],
                width=bar_size,
                bottom=plan_counts_by_mode[mode],
                color=answer_color,
            )
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Failed runs (count)")
        ax.set_xlabel("Model")
    else:
        for mode, offset in zip(modes, offsets):
            mode_pos = [p + offset for p in pos]
            bars_plan_by_mode[mode] = ax.barh(mode_pos, plan_counts_by_mode[mode], height=bar_size, color=planning_color)
            bars_answer_by_mode[mode] = ax.barh(
                mode_pos,
                answer_counts_by_mode[mode],
                height=bar_size,
                left=plan_counts_by_mode[mode],
                color=answer_color,
            )
        ax.set_yticks(pos)
        ax.set_yticklabels(labels if show_y_axis else [])
        ax.set_xlabel("Failed runs (count)")
        ax.set_ylabel("Model" if show_y_axis else "")
        if not show_y_axis:
            ax.tick_params(axis="y", left=False, labelleft=False)

    max_total = max(total_counts) if total_counts else 0
    axis_top = max(10.0, max_total * 1.22)
    if orientation == "vertical":
        ax.set_ylim(0, axis_top)
    else:
        ax.set_xlim(0, axis_top)
    ax.set_title(f"Failure-Reason Breakdown by Failed Runs ({survey_version})", fontsize=TITLE_FONT_SIZE, pad=18)
    ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y" if orientation == "vertical" else "x", linestyle="--", alpha=0.35)

    legend_items = [
        Patch(facecolor=planning_color, label="planning error"),
        Patch(facecolor=answer_color, label="answer error"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )

    for idx in range(len(labels)):
        for mode, offset in zip(modes, offsets):
            mode_total = plan_counts_by_mode[mode][idx] + answer_counts_by_mode[mode][idx]
            mode_pos = pos[idx] + offset
            tag = mode[0].upper()
            if orientation == "vertical":
                ax.text(
                    mode_pos,
                    mode_total + axis_top * 0.02,
                    f"{tag}\nfailed={mode_total}",
                    ha="center",
                    va="bottom",
                    fontsize=FAIL_COUNT_LABEL_FONT_SIZE,
                    fontweight="bold",
                )
            else:
                ax.text(
                    min(axis_top * 0.995, mode_total + axis_top * 0.01),
                    mode_pos,
                    f"{tag} failed={mode_total}",
                    ha="left",
                    va="center",
                    fontsize=FAIL_COUNT_LABEL_FONT_SIZE,
                    fontweight="bold",
                )

    def _label_segments(
        bars,
        pcts: List[float],
        bottoms: Optional[List[int]] = None,
    ) -> None:
        for idx, bar in enumerate(bars):
            size = float(bar.get_height()) if orientation == "vertical" else float(bar.get_width())
            if size <= 0:
                continue
            base = 0.0 if bottoms is None else float(bottoms[idx])
            text = _fmt_pct(pcts[idx])
            if orientation == "vertical":
                y = base + (size * 0.5)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=SEGMENT_LABEL_FONT_SIZE,
                    color="white",
                    fontweight="bold",
                )
            else:
                x = base + (size * 0.5)
                ax.text(
                    x,
                    bar.get_y() + bar.get_height() / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=SEGMENT_LABEL_FONT_SIZE,
                    color="white",
                    fontweight="bold",
                )

    for mode in modes:
        _label_segments(bars_plan_by_mode[mode], plan_pcts_by_mode[mode])
        _label_segments(bars_answer_by_mode[mode], answer_pcts_by_mode[mode], plan_counts_by_mode[mode])


def _plot_stacked_summary(
    labels: List[str],
    success_summary: Dict[str, Dict[str, Tuple[int, int]]],
    error_totals: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None or Patch is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, (ax_top, ax_bottom) = plt.subplots(nrows=2, ncols=1, figsize=FIGSIZE_STANDARD)
    _draw_success_counts(ax_top, labels, success_summary, survey_version, orientation="vertical", modes=modes)
    _draw_error_breakdown(ax_bottom, labels, error_totals, survey_version, orientation="vertical", modes=modes)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, hspace=0.7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_success_counts_horizontal(
    labels: List[str],
    summary: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    _draw_success_counts(ax, labels, summary, survey_version, orientation="horizontal", modes=modes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_error_breakdown_horizontal(
    labels: List[str],
    error_totals: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None or Patch is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    _draw_error_breakdown(ax, labels, error_totals, survey_version, orientation="horizontal", modes=modes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_stacked_summary_horizontal(
    labels: List[str],
    success_summary: Dict[str, Dict[str, Tuple[int, int]]],
    error_totals: Dict[str, Dict[str, Tuple[int, int]]],
    out_path: Path,
    survey_version: str,
    modes: Optional[List[str]] = None,
) -> None:
    if plt is None or Patch is None:
        raise RuntimeError("matplotlib is not available in this environment.")
    fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE_STANDARD, gridspec_kw={"width_ratios": [1, 1]})
    _draw_success_counts(
        ax_left,
        labels,
        success_summary,
        survey_version,
        orientation="horizontal",
        show_y_axis=True,
        modes=modes,
    )
    _draw_error_breakdown(
        ax_right,
        labels,
        error_totals,
        survey_version,
        orientation="horizontal",
        show_y_axis=False,
        modes=modes,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92, wspace=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")
        return 1

    survey_dir = Path(RUNS_ROOT) / SURVEY_VERSION
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else survey_dir / "_overall_summary_plots"

    available = _discover_models(survey_dir)
    if MODELS_FILTER:
        selected = [p for p in available if p.name in set(MODELS_FILTER)]
    else:
        selected = available
    if not selected:
        print(f"No model folders with required CSVs found under: {survey_dir}")
        return 1

    grouped = _split_model_dirs(selected)
    non_empty_groups = [(family, models) for family, models in grouped.items() if models]
    if not non_empty_groups:
        print(f"No LLM/VLM models found under: {survey_dir}")
        return 1

    multiple_groups = len(non_empty_groups) > 1
    wrote_any = False
    for family, models in non_empty_groups:
        modes_to_plot = ["completion"] if family == "vlm" else ["completion", "unconstrained"]
        labels: List[str] = []
        success_summary: Dict[str, Dict[str, Tuple[int, int]]] = {}
        error_summary: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for model_dir in models:
            label = _short_model_label(model_dir.name)
            labels.append(label)
            success_summary[label] = {}
            error_summary[label] = {}
            for mode in modes_to_plot:
                compare_csv = model_dir / COMPARE_FILES[mode]
                plan_csv = model_dir / PLAN_ERROR_FILES[mode]
                success_summary[label][mode] = _successful_runs(compare_csv)
                error_summary[label][mode] = _failed_run_reason_counts(compare_csv, plan_csv)

        suffix = f"_{family}" if multiple_groups else ""
        title_version = f"{SURVEY_VERSION} ({family})" if multiple_groups else SURVEY_VERSION
        plot1 = output_dir / f"overall_success_counts{suffix}.png"
        plot2 = output_dir / f"overall_error_type_normalized{suffix}.png"
        plot3 = output_dir / f"overall_summary_stacked_vertical{suffix}.png"
        _plot_success_counts(labels, success_summary, plot1, title_version, modes_to_plot)
        _plot_error_breakdown(labels, error_summary, plot2, title_version, modes_to_plot)
        _plot_stacked_summary(labels, success_summary, error_summary, plot3, title_version, modes_to_plot)
        print(f"Wrote {family} plot 1: {plot1}")
        print(f"Wrote {family} plot 2: {plot2}")
        print(f"Wrote {family} plot 3: {plot3}")
        wrote_any = True

    if multiple_groups:
        for old_name in (
            "overall_success_counts.png",
            "overall_error_type_normalized.png",
            "overall_summary_stacked_vertical.png",
        ):
            old_path = output_dir / old_name
            if old_path.exists():
                old_path.unlink()

    # Remove older horizontal artifacts if present.
    for old_name in (
        "overall_success_counts_horizontal.png",
        "overall_error_type_normalized_horizontal.png",
        "overall_summary_stacked_horizontal.png",
    ):
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.unlink()
    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
