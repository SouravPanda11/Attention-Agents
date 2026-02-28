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
# Survey version loaded from Agent/.env (for example: survey_v1).
SURVEY_VERSION = os.getenv("SURVEY_VERSION", "").strip() or "survey_v0"
# Baseline survey used for version comparison.
BASELINE_SURVEY_VERSION = "survey_v0"
QWEN_ONLY_COMPARISON_SURVEY_VERSION = "survey_v1"
QWEN_VLM_MODEL_HINT = "qwen2.5-vl-7b"
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
# Keep empty to use: Agent/runs/<SURVEY_VERSION>/_page_summary_plots
OUTPUT_DIR = ""
# Keep empty list to auto-discover all models with required CSVs.
MODELS_FILTER: List[str] = []

PREFERRED_MODEL_ORDER = [
    "meta-llama-3-8b-instruct",
    "mistral-7b-instruct-v0.3",
    "qwen2.5-7b-instruct-1m",
    "llava-1.6-mistral-7b",
    "minicpm-v-2_6",
    "qwen2.5-vl-7b-instruct",
]

GROUP_SPACING = 0.42
PAIR_OFFSET = 0.07
BAR_WIDTH = 0.12
TITLE_FONT_SIZE = 19
PANEL_TITLE_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 14*1.3
TICK_LABEL_FONT_SIZE = 12*1.3
LEGEND_FONT_SIZE = 12*1.3
BAR_LABEL_FONT_SIZE = 12
MODE_COLORS = {"completion": "#4C78A8", "unconstrained": "#F58518"}
VERSION_COLORS = ["#6D597A", "#2A9D8F"]
FIGSIZE_STANDARD = (12, 7)
FIGSIZE_SINGLE_MODEL = (8, 7)
QWEN_ONLY_VERSION_OFFSET_SCALE = 2.5


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


def _discover_models(survey_dir: Path) -> Dict[str, Path]:
    if not survey_dir.exists():
        return {}
    found: Dict[str, Path] = {}
    for model_dir in sorted([p for p in survey_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        # Include models if completion compare CSV exists; unconstrained is optional
        # for VLM-only runs.
        if (model_dir / COMPARE_FILES["completion"]).exists():
            found[model_dir.name] = model_dir
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


def _ordered_model_names(model_names: List[str]) -> List[str]:
    as_set = set(model_names)
    ordered = [name for name in PREFERRED_MODEL_ORDER if name in as_set]
    extras = sorted([name for name in model_names if name not in set(PREFERRED_MODEL_ORDER)], key=lambda n: n.lower())
    return ordered + extras


def _is_successful_row(row: Dict[str, str]) -> bool:
    submitted = _parse_bool(row.get("submitted"))
    if submitted is not None:
        return submitted
    overall_ok = _parse_bool(row.get("overall_ok"))
    return overall_ok is True


def _clip_0_4(v: float) -> float:
    if v < 0:
        return 0.0
    if v > 4:
        return 4.0
    return v


def _summarize_image_accuracy(compare_csv: Path) -> float:
    rows = _read_csv_rows(compare_csv)
    if not rows:
        return 0.0

    success_rows = [row for row in rows if _is_successful_row(row)]
    if not success_rows:
        return 0.0

    accuracy_vals: List[float] = []
    for row in success_rows:
        accuracy_vals.append(_clip_0_4(_parse_float(row.get("image_accuracy_correct"), 0.0)))
    return (sum(accuracy_vals) / len(accuracy_vals)) if accuracy_vals else 0.0


def _summarize_image_attention(compare_csv: Path) -> float:
    rows = _read_csv_rows(compare_csv)
    if not rows:
        return 0.0

    success_rows = [row for row in rows if _is_successful_row(row)]
    if not success_rows:
        return 0.0

    attended = 0
    for row in success_rows:
        if _parse_bool(row.get("image_attention_ok")) is True:
            attended += 1
    return 100.0 * attended / len(success_rows)


def _fmt_out_of_4(v: float) -> str:
    return f"{v:.1f}/4"


def _fmt_pct(v: float) -> str:
    rounded = round(v)
    if abs(v - rounded) < 0.05:
        return f"{int(rounded)}%"
    return f"{v:.1f}%"


def _select_model_names(model_names: List[str]) -> List[str]:
    if MODELS_FILTER:
        return [name for name in MODELS_FILTER if name in set(model_names)]
    return _ordered_model_names(model_names)


def _is_vlm_model_name(model_name: str) -> bool:
    text = model_name.lower()
    return ("-vl-" in text) or text.startswith("llava") or ("minicpm" in text)


def _is_qwen_vlm_model_name(model_name: str) -> bool:
    return QWEN_VLM_MODEL_HINT in model_name.lower()


def _split_model_names(selected_models: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {"llm": [], "vlm": []}
    for model_name in selected_models:
        if _is_vlm_model_name(model_name):
            groups["vlm"].append(model_name)
        else:
            groups["llm"].append(model_name)
    return groups


def _draw_single_version_accuracy(
    *,
    labels: List[str],
    values: Dict[str, Dict[str, float]],
    survey_version: str,
    out_path: Path,
    modes: List[str],
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this Python environment.")

    positions = [i * GROUP_SPACING for i in range(len(labels))]
    if len(modes) == 1:
        offsets = [0.0]
        bar_width = BAR_WIDTH * 1.4
    else:
        offsets = [PAIR_OFFSET * (2 * i - (len(modes) - 1)) for i in range(len(modes))]
        bar_width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    bars_by_mode = {}
    for mode, offset in zip(modes, offsets):
        mode_vals = [values[label][mode] for label in labels]
        bars_by_mode[mode] = ax.bar(
            [p + offset for p in positions],
            mode_vals,
            width=bar_width,
            color=MODE_COLORS[mode],
            label=mode,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel("Avg image accuracy (correct out of 4)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xlabel("Model", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(0, 4.2)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bars in bars_by_mode.values():
        for bar in bars:
            v = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(0.25, v * 0.5),
                _fmt_out_of_4(v),
                ha="center",
                va="center",
                fontsize=BAR_LABEL_FONT_SIZE,
                color="white",
                fontweight="bold",
            )

    fig.suptitle(f"Avg image accuracy ({survey_version})", fontsize=TITLE_FONT_SIZE, y=0.95)
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


def _draw_two_panel_version_bars(
    *,
    labels: List[str],
    values: Dict[str, Dict[str, Dict[str, float]]],
    versions: List[str],
    modes: List[str],
    y_label: str,
    suptitle: str,
    out_path: Path,
    y_min: float,
    y_max: float,
    value_formatter,
    figsize=FIGSIZE_STANDARD,
    reference_group_count: Optional[float] = None,
    version_offset_step: Optional[float] = None,
    left_margin: float = 0.08,
    save_tight: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not available in this Python environment.")

    fig, axes = plt.subplots(nrows=1, ncols=len(modes), figsize=figsize, sharey=True)
    if len(modes) == 1:
        axes = [axes]
    positions = [i * GROUP_SPACING for i in range(len(labels))]

    for ax, mode in zip(axes, modes):
        offset_step = version_offset_step if version_offset_step is not None else BAR_WIDTH
        for idx, version in enumerate(versions):
            offset = (idx - (len(versions) - 1) / 2.0) * offset_step
            bar_positions = [p + offset for p in positions]
            mode_vals = [values[label][mode].get(version, 0.0) for label in labels]
            bars = ax.bar(
                bar_positions,
                mode_vals,
                width=BAR_WIDTH,
                color=VERSION_COLORS[idx % len(VERSION_COLORS)],
                label=version,
            )
            for bar in bars:
                v = float(bar.get_height())
                y = max(y_min + (0.03 * (y_max - y_min)), v * 0.5)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y,
                    value_formatter(v),
                    ha="center",
                    va="center",
                    fontsize=BAR_LABEL_FONT_SIZE,
                    color="white",
                    fontweight="bold",
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=TICK_LABEL_FONT_SIZE)
        if len(modes) > 1:
            ax.set_title(mode, fontsize=PANEL_TITLE_FONT_SIZE, pad=8)
        else:
            ax.set_title("")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE)
        ax.set_xlabel("Model", fontsize=AXIS_LABEL_FONT_SIZE)
        if reference_group_count and reference_group_count > 0 and positions:
            target_span = reference_group_count * GROUP_SPACING
            center = (positions[0] + positions[-1]) / 2.0
            ax.set_xlim(center - (target_span / 2.0), center + (target_span / 2.0))

    axes[0].set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle(suptitle, fontsize=TITLE_FONT_SIZE, y=0.93)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=max(1, len(legend_labels)),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(left=left_margin, right=0.98, bottom=0.10, top=0.82, wspace=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": 180}
    if save_tight:
        save_kwargs["bbox_inches"] = "tight"
        save_kwargs["pad_inches"] = 0.03
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)


def main() -> int:
    if plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")
        return 1

    single_version_mode = SURVEY_VERSION == BASELINE_SURVEY_VERSION
    survey_dir = Path(RUNS_ROOT) / SURVEY_VERSION
    output_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else survey_dir / "_page_summary_plots"

    if single_version_mode:
        discovered = _discover_models(survey_dir)
        if not discovered:
            print(f"No model folders with required compare CSVs found under: {survey_dir}")
            return 1

        selected_models = _select_model_names(list(discovered.keys()))
        if not selected_models:
            print("No models left after applying MODELS_FILTER.")
            return 1

        grouped = _split_model_names(selected_models)
        non_empty_groups = [(family, models) for family, models in grouped.items() if models]
        if not non_empty_groups:
            print("No LLM/VLM models left after grouping.")
            return 1

        multiple_groups = len(non_empty_groups) > 1
        for family, models in non_empty_groups:
            modes_to_plot = ["completion"] if family == "vlm" else ["completion", "unconstrained"]
            labels = [_short_model_label(name) for name in models]
            single_accuracy_values: Dict[str, Dict[str, float]] = {label: {} for label in labels}
            for model_name, label in zip(models, labels):
                model_dir = discovered[model_name]
                for mode in modes_to_plot:
                    csv_path = model_dir / COMPARE_FILES[mode]
                    single_accuracy_values[label][mode] = _summarize_image_accuracy(csv_path)

            suffix = f"_{family}" if multiple_groups else ""
            title_version = f"{SURVEY_VERSION} ({family})" if multiple_groups else SURVEY_VERSION
            plot_accuracy = output_dir / f"image_page_summary_vertical_stacked{suffix}.png"
            _draw_single_version_accuracy(
                labels=labels,
                values=single_accuracy_values,
                survey_version=title_version,
                out_path=plot_accuracy,
                modes=modes_to_plot,
            )
            print(f"Wrote {family} image accuracy plot: {plot_accuracy}")

        if multiple_groups:
            old_base = output_dir / "image_page_summary_vertical_stacked.png"
            if old_base.exists():
                old_base.unlink()

        # Remove comparison artifacts if present.
        for old_name in (
            "image_page_accuracy_by_survey_version.png",
            "image_page_attention_by_survey_version.png",
            "image_page_accuracy_by_survey_version_llm.png",
            "image_page_attention_by_survey_version_llm.png",
            "image_page_accuracy_by_survey_version_vlm.png",
            "image_page_attention_by_survey_version_vlm.png",
            "image_page_accuracy_by_survey_version_vlm_qwen_only.png",
            "image_page_attention_by_survey_version_vlm_qwen_only.png",
        ):
            old_path = output_dir / old_name
            if old_path.exists():
                old_path.unlink()
        return 0

    comparison_versions = [BASELINE_SURVEY_VERSION, SURVEY_VERSION]
    discovered_by_version: Dict[str, Dict[str, Path]] = {}
    for version in comparison_versions:
        version_dir = Path(RUNS_ROOT) / version
        discovered_by_version[version] = _discover_models(version_dir)
        if not discovered_by_version[version]:
            print(f"No model folders with required compare CSVs found under: {version_dir}")
            return 1

    common_model_names = set(discovered_by_version[comparison_versions[0]].keys())
    for version in comparison_versions[1:]:
        common_model_names &= set(discovered_by_version[version].keys())
    if not common_model_names:
        print(f"No common model folders found across versions: {', '.join(comparison_versions)}")
        return 1

    selected_models = _select_model_names(list(common_model_names))
    if not selected_models:
        print("No models left after applying MODELS_FILTER.")
        return 1

    grouped = _split_model_names(selected_models)
    non_empty_groups = [(family, models) for family, models in grouped.items() if models]
    if not non_empty_groups:
        print("No LLM/VLM models left after grouping.")
        return 1

    multiple_groups = len(non_empty_groups) > 1
    wrote_qwen_only_vlm = False
    for family, models in non_empty_groups:
        modes_to_plot = ["completion"] if family == "vlm" else ["completion", "unconstrained"]
        labels = [_short_model_label(name) for name in models]
        accuracy_values: Dict[str, Dict[str, Dict[str, float]]] = {
            label: {mode: {} for mode in modes_to_plot} for label in labels
        }
        attention_values: Dict[str, Dict[str, Dict[str, float]]] = {
            label: {mode: {} for mode in modes_to_plot} for label in labels
        }

        for model_name, label in zip(models, labels):
            for version in comparison_versions:
                model_dir = discovered_by_version[version][model_name]
                for mode in modes_to_plot:
                    csv_path = model_dir / COMPARE_FILES[mode]
                    accuracy_values[label][mode][version] = _summarize_image_accuracy(csv_path)
                    attention_values[label][mode][version] = _summarize_image_attention(csv_path)

        suffix = f"_{family}" if multiple_groups else ""
        group_tag = f" ({family})" if multiple_groups else ""
        plot_accuracy = output_dir / f"image_page_accuracy_by_survey_version{suffix}.png"
        plot_attention = output_dir / f"image_page_attention_by_survey_version{suffix}.png"

        _draw_two_panel_version_bars(
            labels=labels,
            values=accuracy_values,
            versions=comparison_versions,
            modes=modes_to_plot,
            y_label="Avg image accuracy (correct out of 4)",
            suptitle=f"Image accuracy{group_tag}",
            out_path=plot_accuracy,
            y_min=0.0,
            y_max=4.2,
            value_formatter=_fmt_out_of_4,
        )
        _draw_two_panel_version_bars(
            labels=labels,
            values=attention_values,
            versions=comparison_versions,
            modes=modes_to_plot,
            y_label="Image attention (%)",
            suptitle=f"Image attention{group_tag}",
            out_path=plot_attention,
            y_min=0.0,
            y_max=102.0,
            value_formatter=_fmt_pct,
        )
        print(f"Wrote {family} image accuracy comparison plot: {plot_accuracy}")
        print(f"Wrote {family} image attention comparison plot: {plot_attention}")

        if family == "vlm" and SURVEY_VERSION == QWEN_ONLY_COMPARISON_SURVEY_VERSION:
            qwen_labels = [
                _short_model_label(model_name)
                for model_name in models
                if _is_qwen_vlm_model_name(model_name)
            ]
            qwen_labels = list(dict.fromkeys(qwen_labels))
            if qwen_labels:
                qwen_reference_group_count = max(
                    1.0,
                    len(labels) * (FIGSIZE_SINGLE_MODEL[0] / FIGSIZE_STANDARD[0]),
                )
                qwen_accuracy_values = {label: accuracy_values[label] for label in qwen_labels}
                qwen_attention_values = {label: attention_values[label] for label in qwen_labels}
                plot_qwen_accuracy = output_dir / "image_page_accuracy_by_survey_version_vlm_qwen_only.png"
                plot_qwen_attention = output_dir / "image_page_attention_by_survey_version_vlm_qwen_only.png"

                _draw_two_panel_version_bars(
                    labels=qwen_labels,
                    values=qwen_accuracy_values,
                    versions=comparison_versions,
                    modes=modes_to_plot,
                    y_label="Avg image accuracy (correct out of 4)",
                    suptitle="Image accuracy (vlm, qwen only)",
                    out_path=plot_qwen_accuracy,
                    y_min=0.0,
                    y_max=4.2,
                    value_formatter=_fmt_out_of_4,
                    figsize=FIGSIZE_SINGLE_MODEL,
                    reference_group_count=qwen_reference_group_count,
                    version_offset_step=BAR_WIDTH * QWEN_ONLY_VERSION_OFFSET_SCALE,
                    left_margin=0.14,
                    save_tight=True,
                )
                _draw_two_panel_version_bars(
                    labels=qwen_labels,
                    values=qwen_attention_values,
                    versions=comparison_versions,
                    modes=modes_to_plot,
                    y_label="Image attention (%)",
                    suptitle="Image attention (vlm, qwen only)",
                    out_path=plot_qwen_attention,
                    y_min=0.0,
                    y_max=102.0,
                    value_formatter=_fmt_pct,
                    figsize=FIGSIZE_SINGLE_MODEL,
                    reference_group_count=qwen_reference_group_count,
                    version_offset_step=BAR_WIDTH * QWEN_ONLY_VERSION_OFFSET_SCALE,
                    left_margin=0.14,
                    save_tight=True,
                )
                print(f"Wrote vlm qwen-only image accuracy comparison plot: {plot_qwen_accuracy}")
                print(f"Wrote vlm qwen-only image attention comparison plot: {plot_qwen_attention}")
                wrote_qwen_only_vlm = True

    if multiple_groups:
        for old_name in ("image_page_accuracy_by_survey_version.png", "image_page_attention_by_survey_version.png"):
            old_path = output_dir / old_name
            if old_path.exists():
                old_path.unlink()

    if not wrote_qwen_only_vlm:
        for old_name in (
            "image_page_accuracy_by_survey_version_vlm_qwen_only.png",
            "image_page_attention_by_survey_version_vlm_qwen_only.png",
        ):
            old_path = output_dir / old_name
            if old_path.exists():
                old_path.unlink()

    # Remove single-version artifact if present.
    for old_name in (
        "image_page_summary_vertical_stacked.png",
        "image_page_summary_vertical_stacked_llm.png",
        "image_page_summary_vertical_stacked_vlm.png",
    ):
        old_path = output_dir / old_name
        if old_path.exists():
            old_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
