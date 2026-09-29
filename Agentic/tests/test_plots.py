"""Check plotting denominators and experiment grouping without rendering charts."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from plots.common import attempt_rows, format_rows, load_groups, summary_rows, theme_rows


def example():
    return {"model": "example", "prompt_version": "zero-shot", "status": "submitted",
            "theme_id": "work", "order_id": "order01", "question_count": 2,
            "model_calls": 4, "wall_seconds": 3, "model_seconds": 2,
            "budget_exhausted_questions": 1,
            "question_results": [{"question_id": "a", "kind": "short-text", "status": "answered", "turns": 1},
                                 {"question_id": "b", "kind": "short-text", "status": "unanswered", "turns": 3,
                                  "reason": "turn_budget_exhausted"}],
            "evaluation": {"submitted": True, "valid_count": 1, "skipped_count": 1, "invalid_count": 0,
                           "question_results": [{"questionId": "a", "kind": "short-text", "status": "valid"},
                                                {"questionId": "b", "kind": "short-text", "status": "skipped"}]}}


class PlotDataTests(unittest.TestCase):
    def test_failed_run_stays_in_overall_but_not_format_denominator(self):
        good = example()
        failed = deepcopy(good)
        failed.update(status="error", evaluation={"submitted": False, "valid_count": None})
        row = summary_rows([good, failed])[0]
        self.assertEqual(row["end_to_end_valid_pct"], 25)
        self.assertEqual(row["valid_pct_submitted"], 50)
        self.assertEqual(row["submission_pct"], 50)
        self.assertEqual(row["unsubmitted_questions"], 2)
        self.assertEqual(format_rows([good, failed])[0]["submitted_items"], 2)
        self.assertEqual(theme_rows([good, failed])[0]["valid_pct"], 25)

    def test_format_rates_pool_counts_instead_of_averaging_rates(self):
        first, second = example(), example()
        first["evaluation"]["question_results"] = [{"questionId": "a", "kind": "short-text", "status": "valid"}]
        second["evaluation"]["question_results"] = [{"questionId": str(i), "kind": "short-text", "status": "skipped"} for i in range(3)]
        self.assertEqual(format_rows([first, second])[0]["valid_pct"], 25)

    def test_third_attempt_can_succeed_or_exhaust(self):
        run = example()
        run["question_results"][0]["turns"] = 3
        row = attempt_rows([run])[0]
        self.assertEqual((row["attempts"], row["answered"], row["unanswered"], row["exhausted"]), (3, 1, 1, 1))

    def test_no_submission_is_missing_format_data_not_zero_accuracy(self):
        run = example()
        run["evaluation"] = {"submitted": False}
        self.assertEqual(format_rows([run]), [])
        self.assertIsNone(summary_rows([run])[0]["valid_pct_submitted"])

    def test_raw_runs_group_by_protocol_and_filter_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (prompt, status, batch) in enumerate([("zero-shot", "submitted", "batch-a"),
                    ("few-shot", "submitted", "batch-a"), ("zero-shot", "running", "batch-a"),
                    ("zero-shot", "submitted", "batch-b")]):
                run = example()
                run.update(prompt_version=prompt, status=status)
                path = root / batch / f"run-{index}" / "run_summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(run), encoding="utf-8")
            (root / "comparison.json").write_text(json.dumps([example()] * 20), encoding="utf-8")
            groups = load_groups(root)
            self.assertEqual(len(groups), 2)
            self.assertEqual(sum(len(runs) for _, runs in groups), 3)
            self.assertEqual(sum(len(runs) for _, runs in load_groups(root, batch="batch-a")), 2)
            with self.assertRaisesRegex(ValueError, "No completed"):
                load_groups(root, models=["absent"])


if __name__ == "__main__":
    unittest.main()
