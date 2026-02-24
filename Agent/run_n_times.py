import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# -----------------------------
# Editable in-code configuration
# -----------------------------
NUM_RUNS = 2
HEADLESS = True
DELAY_S = 0.0
FAIL_FAST = False

# Ensure `import agent` works even when launched from workspace root.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
