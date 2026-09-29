"""Run duration and retry-related counters, with raw observations visible."""
from .common import COLORS, by_model, finish, pyplot, standalone, summary_rows


def draw(runs, out, formats):
    plt = pyplot()
    grouped = by_model(runs)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.7, 1.1 * len(grouped) + 3)))
    for i, (model, items) in enumerate(grouped.items()):
        values = [r["wall_seconds"] for r in items]
        offsets = [i + (j % 7 - 3) * .035 for j in range(len(items))]
        axes[0].scatter(values, offsets, alpha=.65, color=COLORS[i % len(COLORS)], s=25)
        axes[0].scatter(sum(values) / len(values), i, marker="D", color="black", s=45, zorder=3)
    axes[0].set(yticks=range(len(grouped)), yticklabels=list(grouped), xlabel="Wall time per run (seconds)",
                title="Run duration: dots = runs, diamond = mean", xlim=(0, None))
    axes[0].invert_yaxis()
    keys = [("plan_errors", "Rejected plans"), ("invalid_answer_turns", "Incomplete/invalid answer turns"),
            ("model_errors", "Model request errors"), ("action_errors", "Browser action errors")]
    rows = summary_rows(runs)
    width = .8 / len(rows)
    for i, row in enumerate(rows):
        y = [j - .4 + width * (i + .5) for j in range(len(keys))]
        values = [100 * row[k] / row["model_calls"] if row["model_calls"] else 0 for k, _ in keys]
        bars = axes[1].barh(y, values, height=width * .9, label=row["model"], color=COLORS[i % len(COLORS)])
        axes[1].bar_label(bars, labels=[f"{row[k]} / {row['model_calls']} calls" for k, _ in keys], padding=3, fontsize=8)
    axes[1].set(yticks=range(len(keys)), yticklabels=[label for _, label in keys], xlabel="Events per 100 model calls",
                title="Why the agent retries")
    axes[1].set_xlim(0, max([100 * row[k] / row["model_calls"] for row in rows for k, _ in keys if row["model_calls"]] or [0]) * 1.5 + 15)
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.18), frameon=False, fontsize=8)
    finish(fig, out, "05_efficiency", "All attempted runs. Durations depend on local hardware and server settings.\nError counters can overlap within a turn; they are not mutually exclusive outcomes or unique question counts.", formats, runs)


if __name__ == "__main__":
    standalone(draw, "05_efficiency")
