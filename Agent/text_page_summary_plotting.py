import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

from model_stack import ensure_env_loaded

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


COMPARE_FILES = {
    "completion": "completed_compared_runs.csv",
    "unconstrained": "unconstrained_compare_runs.csv",
}

# -----------------------------
# Editable in-code configuration
# -----------------------------
ensure_env_loaded()
# Survey version folder under Agent/runs (loaded from Agent/.env).
SURVEY_VERSION = os.getenv("SURVEY_VERSION", "").strip() or "survey_v0"
# Root runs directory (default: Agent/runs)
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
# Output directory. Keep empty to use:
# Agent/runs/<SURVEY_VERSION>/_page_summary_plots
OUTPUT_DIR = ""
# Optional model folder names to include.
# Keep empty list to auto-discover all models with required CSVs.
MODELS_FILTER: List[str] = []

GROUP_SPACING = 0.4
PAIR_OFFSET = 0.07
BAR_WIDTH = 0.1
COMPLETION_COLOR = "#4C78A8"
UNCONSTRAINED_COLOR = "#F58518"
TITLE_FONT_SIZE = 19
AXIS_LABEL_FONT_SIZE = 14*1.3
TICK_LABEL_FONT_SIZE = 12*1.3
LEGEND_FONT_SIZE = 12*1.3
BAR_LABEL_FONT_SIZE = 12*1.3
FIGSIZE_STANDARD = (12, 7)


def _parse_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _parse_float(value: object, default: float = 0.0) -> float:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return float(text)
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
        # Include models if completion compare CSV exists; unconstrained is optional
        # for VLM-only runs.
        if (model_dir / COMPARE_FILES["completion"]).exists():
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


def _is_successful_row(row: Dict[str, str]) -> bool:
    submitted = _parse_bool(row.get("submitted"))
    if submitted is not None:
        return submitted
    overall_ok = _parse_bool(row.get("overall_ok"))
    return overall_ok is True


def _summarize_mode(compare_csv: Path) -> float:
    rows = _read_csv_rows(compare_csv)
    if not rows:
        return 0.0

    success_rows = [row for row in rows if _is_successful_row(row)]
    if not success_rows:
        return 0.0

    answered_vals: List[float] = []
    for row in success_rows:
        answered = _parse_float(row.get("text_completion_answered"), 0.0)
        if answered < 0:
            answered = 0.0
        if answered > 9:
            answered = 9.0
        answered_vals.append(answered)

    return (sum(answered_vals) / len(answered_vals)) if answered_vals else 0.0


def _fmt_out_of_9(v: float) -> str:
    return f"{v:.1f}/9"


def _draw_vertical(
    labels: List[str],
    values: Dict[str, Dict[str, float]],
    survey_version: str,
    out_path: Path,
    modes: List[str],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this Python environment.")

    pos = [i * GROUP_SPACING for i in range(len(labels))]
    if len(modes) == 1:
        offsets = [0.0]
        bar_width = BAR_WIDTH * 1.4
    else:
        offsets = [PAIR_OFFSET * (2 * i - (len(modes) - 1)) for i in range(len(modes))]
        bar_width = BAR_WIDTH
    mode_colors = {"completion": COMPLETION_COLOR, "unconstrained": UNCONSTRAINED_COLOR}

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars_by_mode = {}
    for mode, offset in zip(modes, offsets):
        mode_vals = [values[label][mode] for label in labels]
        bars_by_mode[mode] = ax.bar(
            [p + offset for p in pos],
            mode_vals,
            width=bar_width,
            color=mode_colors[mode],
            label=mode,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Avg text completion (answered out of 9)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Model", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(0, 9.6)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in bars_by_mode.values():
        for bar in bars:
            v = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(0.4, v * 0.5),
                _fmt_out_of_9(v),
                ha="center",
                va="center",
                fontsize=BAR_LABEL_FONT_SIZE,
                color="white",
                fontweight="bold",
            )

    ax.set_title("")
    fig.suptitle(f"Avg text completion ({survey_version})", fontsize=TITLE_FONT_SIZE, y=0.95)
    handles = [bars_by_mode[m][0] for m in modes if len(bars_by_mode.get(m, [])) > 0]
    if handles:
        fig.legend(
            handles,
            modes,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=max(1, len(modes)),
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.84)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")
        return 1

    survey_dir = Path(RUNS_ROOT) / SURVEY_VERSION
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else survey_dir / "_page_summary_plots"
    discovered = _discover_models(survey_dir)
    selected = [m for m in discovered if m.name in set(MODELS_FILTER)] if MODELS_FILTER else discovered
    if not selected:
        print(f"No model folders with required compare CSVs found under: {survey_dir}")
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
        values: Dict[str, Dict[str, float]] = {}
        for model_dir in models:
            label = _short_model_label(model_dir.name)
            labels.append(label)
            values[label] = {}
            for mode in modes_to_plot:
                csv_path = model_dir / COMPARE_FILES[mode]
                values[label][mode] = _summarize_mode(csv_path)

        suffix = f"_{family}" if multiple_groups else ""
        title_version = f"{SURVEY_VERSION} ({family})" if multiple_groups else SURVEY_VERSION
        plot_v = output_dir / f"text_page_completion_vertical{suffix}.png"
        _draw_vertical(labels, values, title_version, plot_v, modes_to_plot)
        print(f"Wrote {family} vertical plot: {plot_v}")
        wrote_any = True

    if multiple_groups:
        old_base = output_dir / "text_page_completion_vertical.png"
        if old_base.exists():
            old_base.unlink()

    # Remove older horizontal artifact if present.
    old_horizontal = output_dir / "text_page_completion_horizontal.png"
    if old_horizontal.exists():
        old_horizontal.unlink()
    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())

