"""Actual per-question attempt counts, split by local completion outcome."""
from .common import COLORS, attempt_rows, by_model, finish, pyplot, standalone


def draw(runs, out, formats):
    plt = pyplot()
    rows = attempt_rows(runs)
    models = list(by_model(runs))
    fig, axes = plt.subplots(len(models), 1, squeeze=False, figsize=(10, 4.8 * len(models)))
    for ax, model in zip(axes.flat, models):
        selected = [r for r in rows if r["model"] == model]
        turns = sorted({r["attempts"] for r in selected})
        answered = [r["answered"] for r in selected]
        unanswered = [r["unanswered"] for r in selected]
        bottom = ax.bar(turns, answered, color=COLORS[0], label="Answered")
        top = ax.bar(turns, unanswered, bottom=answered, color="#e3b25b", label="Left unanswered")
        ax.bar_label(bottom, labels=[str(n) if n else "" for n in answered], label_type="center")
        ax.bar_label(top, labels=[str(n) if n else "" for n in unanswered], label_type="center")
        ax.set(xticks=turns, xlabel="Actual model calls used for this question", ylabel="Number of questions",
               title=f"{model}\n{sum(answered)} answered · {sum(unanswered)} unanswered · {sum(r['exhausted'] for r in selected)} exhausted budget")
        ax.legend(frameon=False, loc="upper right")
        ax.set_ylim(0, max([a + b for a, b in zip(answered, unanswered)] or [1]) * 1.25)
    finish(fig, out, "04_attempts", "Question outcomes recorded by the agent, including runs that did not submit.\nA third attempt can succeed or end unanswered. Questions without a final local record are excluded.", formats, runs)


if __name__ == "__main__":
    standalone(draw, "04_attempts")
