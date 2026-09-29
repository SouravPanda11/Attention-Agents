"""Survey submission versus answer validity, with separate denominators."""
from .common import COLORS, finish, pyplot, standalone, summary_rows


def draw(runs, out, formats):
    plt = pyplot()
    rows = summary_rows(runs)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.2, 1.1 * len(rows) + 3)))
    labels = [r["model"] for r in rows]
    series = [("submission_pct", "Surveys submitted", COLORS[0]),
              ("end_to_end_valid_pct", "Valid answers / all run questions", COLORS[1]),
              (None, "Surveys with every answer valid", COLORS[2])]
    for index, (key, label, color) in enumerate(series):
        y = [i + (index - 1) * .24 for i in range(len(rows))]
        values = [r[key] if key else 100 * r["fully_valid_runs"] / r["runs"] for r in rows]
        bars = axes[0].barh(y, values, height=.22, color=color, label=label)
        axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=9)
    axes[0].set(yticks=range(len(rows)), yticklabels=labels, xlim=(0, 117), xlabel="Percent", title="Submission is only part of completion")
    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.2), frameon=False, fontsize=8)
    left = [0] * len(rows)
    for key, label, color in [("valid_answers", "Valid", COLORS[0]), ("skipped_submitted", "Skipped", "#e3b25b"),
                               ("invalid_submitted", "Invalid", "#c85252"), ("unsubmitted_questions", "Not submitted", "#afb5bf")]:
        values = [100 * r[key] / r["questions"] for r in rows]
        bars = axes[1].barh(range(len(rows)), values, left=left, color=color, label=label)
        axes[1].bar_label(bars, labels=[str(r[key]) if v > 6 else "" for r, v in zip(rows, values)], label_type="center")
        left = [a + b for a, b in zip(left, values)]
    axes[1].set(yticks=range(len(rows)), yticklabels=[f"{r['questions']} questions\n{r['runs']} runs" for r in rows],
                xlim=(0, 100), xlabel="Share of questions (%)", title="Final question outcomes (counts inside bars)")
    axes[1].invert_yaxis()
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -.2), ncol=2, frameon=False, fontsize=8)
    finish(fig, out, "01_overview", "Valid = meets form constraints; it does not measure semantic correctness. Failed submissions remain in the overall denominator.", formats, runs)


if __name__ == "__main__":
    standalone(draw, "01_overview")
