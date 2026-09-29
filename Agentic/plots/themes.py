"""Fixed-scale theme/order heatmaps with denominators and missing cells."""
from .common import THEMES, finish, pyplot, standalone, theme_rows


def draw(runs, out, formats):
    import numpy as np
    plt = pyplot()
    rows = theme_rows(runs)
    models = sorted({r["model"] for r in rows})
    themes = [t for t in THEMES if any(r["theme"] == t for r in rows)]
    themes += sorted({r["theme"] for r in rows if r["theme"] not in themes}, key=str)
    orders = sorted({r["order"] for r in rows}, key=str)
    cols = min(2, len(models))
    fig, axes = plt.subplots((len(models) + cols - 1) // cols, cols, squeeze=False,
                             figsize=(6.5 * cols, max(6, len(themes) * .6 + 2) * ((len(models) + cols - 1) // cols)))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    for ax, model in zip(axes.flat, models):
        lookup = {(r["theme"], r["order"]): r for r in rows if r["model"] == model}
        data = np.full((len(themes), len(orders)), np.nan)
        for i, theme in enumerate(themes):
            for j, order in enumerate(orders):
                row = lookup.get((theme, order))
                if row:
                    data[i, j] = row["valid_pct"]
                    text = f"{row['valid_pct']:.1f}%\n{row['valid_answers']}/{row['questions']} · n={row['runs']}"
                else:
                    text = "No runs"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="white" if row and row["valid_pct"] > 65 else "#202936")
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set(xticks=range(len(orders)), xticklabels=orders, yticks=range(len(themes)),
               yticklabels=[str(t).capitalize() for t in themes], title=model)
        fig.colorbar(im, ax=ax, shrink=.7, label="Valid answers (%)")
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle("Answer validity by theme and question order", fontsize=15, fontweight="bold")
    finish(fig, out, "03_theme_order", "Each cell: valid answers / all run questions; n = runs. Missing runs are gray.\nOne run per cell is descriptive evidence, not an estimate of reliable order effects.", formats, runs)


if __name__ == "__main__":
    standalone(draw, "03_theme_order")
