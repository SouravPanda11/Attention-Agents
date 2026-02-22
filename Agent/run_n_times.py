import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------
# Editable in-code configuration
# -----------------------------
NUM_RUNS = 100
HEADLESS = True
DELAY_S = 0.0
FAIL_FAST = False
PRINT_ANSWER_EVAL = False

# Ensure `import agent` works even when launched from workspace root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_answer_eval(run_dir: Path) -> str:
    eval_data = _load_json(run_dir / "answer_eval.json") or {}
    metrics = _load_json(run_dir / "run_metrics.json") or {}
    if not eval_data:
        return "answer_eval=missing"

    img_known = int(eval_data.get("image_known_count") or 0)
    img_correct = int(eval_data.get("image_correct_count") or 0)
    img_acc = eval_data.get("image_accuracy_known")
    img_attn = ((eval_data.get("image_attention") or {}).get("is_correct"))
    txt_attn = ((eval_data.get("text_attention") or {}).get("is_correct"))
    captcha_ok = ((eval_data.get("captcha") or {}).get("is_correct"))
    text_comp = eval_data.get("text_page_completion") or {}
    image_comp = eval_data.get("image_page_completion") or {}

    def _tri(v: Any) -> str:
        if v is True:
            return "Y"
        if v is False:
            return "N"
        return "-"

    def _comp(section: Dict[str, Any]) -> str:
        answered = int(section.get("answered_count") or 0)
        total = int(section.get("total_count") or 0)
        rate = section.get("completion_rate")
        if total <= 0:
            return "-"
        if not isinstance(rate, (int, float)):
            return f"{answered}/{total}"
        return f"{answered}/{total}({float(rate):.2f})"

    img_acc_text = "-" if img_acc is None else f"{float(img_acc):.2f}"
    return (
        f"text_comp={_comp(text_comp)} "
        f"img={img_correct}/{img_known}({img_acc_text}) "
        f"img_comp={_comp(image_comp)} "
        f"img_attn={_tri(img_attn)} "
        f"text_attn={_tri(txt_attn)} "
        f"captcha={_tri(captcha_ok)} "
        f"thankyou={_tri(metrics.get('reached_thank_you'))} "
        f"submitted={_tri(metrics.get('submitted'))}"
    )

async def _run_many(num_runs: int, headless: bool, delay_s: float, fail_fast: bool) -> int:
    import agent

    successes: List[Path] = []
    failures = 0

    started = datetime.now().isoformat(timespec="seconds")
    print(f"[batch] start={started} runs={num_runs} headless={headless} delay_s={delay_s}")

    for i in range(1, num_runs + 1):
        print(f"[batch] run {i}/{num_runs} starting")
        try:
            run_dir = await agent.main(headless=headless)
            run_path = Path(run_dir)
            successes.append(run_path)
            if PRINT_ANSWER_EVAL:
                eval_text = _format_answer_eval(run_path)
                print(f"[batch] run {i}/{num_runs} ok path={run_dir} {eval_text}")
            else:
                print(f"[batch] run {i}/{num_runs} ok path={run_dir}")
        except Exception as exc:
            failures += 1
            print(f"[batch] run {i}/{num_runs} failed error={exc}")
            if fail_fast:
                break

        if delay_s > 0 and i < num_runs:
            await asyncio.sleep(delay_s)

    finished = datetime.now().isoformat(timespec="seconds")
    print(f"[batch] finish={finished} ok={len(successes)} failed={failures}")
    if successes:
        print("[batch] completed run directories:")
        for path in successes:
            print(f"- {path}")
    return 1 if failures > 0 else 0


def main() -> int:
    if NUM_RUNS <= 0:
        raise SystemExit("NUM_RUNS must be >= 1")
    if DELAY_S < 0:
        raise SystemExit("DELAY_S must be >= 0")

    return asyncio.run(
        _run_many(
            num_runs=NUM_RUNS,
            headless=HEADLESS,
            delay_s=DELAY_S,
            fail_fast=FAIL_FAST,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
