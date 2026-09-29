"""Question-type completion, pooled over the selected themes and orders."""
from .common import COLORS, KINDS, LABELS, finish, format_rows, pyplot, standalone

def draw(runs, out, formats):
    plt = pyplot()
    rows = format_rows(runs)
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    kinds = [k for k in KINDS if any(r["kind"] == k for r in rows)]
    kinds += sorted({r["kind"] for r in rows} - set(kinds))
    lookup = {(r["model"], r["kind"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(12, max(7, len(kinds) * (.35 + .15 * len(models)) + 2)))
    width = .8 / len(models)
    for i, model in enumerate(models):
        for j, kind in enumerate(kinds):
            row = lookup.get((model, kind))
            y = j - .4 + width * (i + .5)
            if row is None:
                ax.text(1, y, "No submitted data", va="center", fontsize=8)
                continue
            ax.barh(y, row["valid_pct"], height=width * .9, color=COLORS[i % len(COLORS)], label=model if j == 0 else None)
            ax.text(row["valid_pct"] + 1, y, f"{row['valid']}/{row['submitted_items']} ({row['valid_pct']:.1f}%)", va="center", fontsize=9)
    ax.set(yticks=range(len(kinds)), yticklabels=[LABELS.get(k, k) for k in kinds], xlim=(0, 128),
           xlabel="Valid answers / submitted questions of this type (%)", title="Which question types get answered successfully?")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    finish(fig, out, "02_question_types", "Accepted submissions only; failed runs are excluded here. Labels show valid / total items.\nImage selection in DOM mode uses text/alt labels, with no screenshots.", formats, runs)


if __name__ == "__main__":
    standalone(draw, "02_question_types")
