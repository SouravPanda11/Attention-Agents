import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import build_schedule, parser, select_workflows
from brain import messages_for, parse_plan
from browser import validate_plan
from evaluation import evaluate_submission, ratio, report, write_json


def workflow(order="order01", layout="item"):
    return {"id": f"v0-standard-o1-{layout}-{order}", "profile": "standard", "occurrence": 1,
            "layout": layout, "orderId": order, "pageCount": 1, "questionCount": 13,
            "substantiveQuestionCount": 11, "attentionCheckCount": 2, "hasWelcomePage": True,
            "orderedQuestionIds": [f"q{i}" for i in range(13)],
            "url": f"/survey/standard/o1/{layout}/{order}"}


def theme_workflow(theme="work", order="order01"):
    item = workflow(order)
    ids = [f"main-{theme}-q{i}" for i in range(11)]
    offset = int(order[-2:]) - 1
    item.update(id=f"v0-standard-o1-theme-{theme}-{order}", themeId=theme, themeLabel=theme,
                questionCount=11, attentionCheckCount=0, attentionCheckContentVersion=0,
                orderedQuestionIds=ids[offset:] + ids[:offset], url=f"/survey/themes/{theme}/{order}")
    return item


def theme_manifest(themes=("work",)):
    return {"themes": [{"id": t} for t in themes],
            "themeWorkflows": [theme_workflow(t, order) for t in themes
                               for order in ("order01", "order02", "order03")]}


class PipelineTests(unittest.TestCase):
    def test_o1_selection(self):
        manifest = theme_manifest()
        selected = select_workflows(manifest, ["order03", "order01"], ["work"])
        self.assertEqual([w["orderId"] for w in selected], ["order03", "order01"])
        manifest["themeWorkflows"][0]["attentionCheckCount"] = 2
        with self.assertRaises(ValueError):
            select_workflows(manifest, ["order01"], ["work"])

    def test_external_manifest_url_rejected(self):
        manifest = theme_manifest()
        manifest["themeWorkflows"][0]["url"] = "https://example.com/survey"
        with self.assertRaises(ValueError):
            select_workflows(manifest, ["order01"], ["work"])

    def test_theme_schedule_cycles_orders_before_switching_theme(self):
        themes = ["consumer", "digital", "wellbeing", "education", "work", "finance", "civic", "lifestyle"]
        selected = select_workflows(theme_manifest(themes), ["order01", "order02", "order03"])
        schedule = build_schedule(selected, 1)
        self.assertEqual(len(schedule), 24)
        schedule = build_schedule(selected, 5)
        self.assertEqual(len(schedule), 120)
        self.assertEqual([w["themeId"] for w, _, _ in schedule[:15]], ["consumer"] * 15)
        self.assertEqual([w["orderId"] for w, _, _ in schedule[:15]], ["order01", "order02", "order03"] * 5)
        self.assertEqual([i for _, _, i in schedule[:15]], list(range(1, 16)))
        self.assertEqual(schedule[15][0]["themeId"], "digital")
        self.assertEqual(schedule[15][2], 1)

    def test_duplicate_orders_and_question_set_changes_are_rejected(self):
        manifest = theme_manifest()
        with self.assertRaises(ValueError):
            select_workflows(manifest, ["order01", "order01"])
        manifest["themeWorkflows"][1]["orderedQuestionIds"][0] = "main-work-new"
        with self.assertRaises(ValueError):
            select_workflows(manifest, ["order01", "order02"])

    def test_unknown_key_and_early_submit_rejected(self):
        observation = {"fields": [{"key": "submit-survey", "tool": "click"}]}
        for plan in ([{"tool": "click", "key": "invented"}],
                     [{"tool": "click", "key": "submit-survey"}] * 2):
            with self.assertRaises(ValueError):
                validate_plan(plan, observation)

    def test_rank_rejects_duplicates(self):
        observation = {"fields": [{"key": "q", "tool": "rank", "options": [{"value": "a"}, {"value": "b"}]}]}
        with self.assertRaises(ValueError):
            validate_plan([{"tool": "rank", "key": "q", "value": ["a", "a"]}], observation)
        validate_plan([{"tool": "rank", "key": "q", "value": ["b", "a"]}], observation)

    def test_slider_step_and_finite_number(self):
        observation = {"fields": [{"key": "q", "tool": "set_range", "min": "0", "max": "10", "step": "2"}]}
        for value in (3, 12, float("nan"), True):
            with self.assertRaises(ValueError):
                validate_plan([{"tool": "set_range", "key": "q", "value": value}], observation)

    def test_parse_and_prompt(self):
        response = {"choices": [{"message": {"content": '```json\n[{"tool":"done"}]\n```'}}]}
        self.assertEqual(parse_plan(response), [{"tool": "done"}])
        messages = messages_for({"fields": []}, [], "completion")
        self.assertEqual(len(messages), 2)
        self.assertNotIn("attention_results", json.dumps(messages))

    def test_behavior_default_and_explicit_override(self):
        with patch("agent.PROMPT_BEHAVIOR_MODE", "unconstrained"), patch("agent.os.environ", {}):
            args = parser().parse_args(["run"])
            self.assertEqual(args.behavior, "unconstrained")
            prompt = messages_for({"fields": []}, [], args.behavior)[0]["content"]
            self.assertNotIn("Hard requirement", prompt)
            args = parser().parse_args(["run", "--behavior", "completion"])
            prompt = messages_for({"fields": []}, [], args.behavior)[0]["content"]
            self.assertIn("Hard requirement", prompt)
            self.assertIn("answer the current question", prompt)

    def test_check_labels_resolve_to_values_without_guessing(self):
        field = {"key": "q", "tool": "check", "kind": "multiple-checkbox", "options": [
            {"value": "a", "label": "Alpha"}, {"value": "b", "label": "Beta"}]}
        plan = [{"tool": "check", "key": "q", "value": "Alpha"},
                {"tool": "uncheck", "key": "q", "value": "Beta"}]
        validate_plan(plan, {"fields": [field]})
        self.assertEqual([action["value"] for action in plan], ["a", "b"])
        field["options"][1]["label"] = "Alpha"
        with self.assertRaises(ValueError):
            validate_plan([{"tool": "check", "key": "q", "value": "Alpha"}], {"fields": [field]})
        validate_plan([{"tool": "check", "key": "q", "value": "a"}], {"fields": [field]})
        field["tool"] = "select"
        with self.assertRaises(ValueError):
            validate_plan([{"tool": "select", "key": "q", "value": "Alpha"}], {"fields": [field]})

    def test_unavailable_scores_are_null(self):
        snapshot = {"status": 200, "response": {"accepted": True, "attemptedCount": 12,
                    "validCount": 11, "invalidCount": 1, "skippedCount": 1}}
        score = evaluate_submission(snapshot, workflow(), None)
        self.assertEqual(score["valid_count"], 11)
        self.assertIsNone(score["attention_pass_rate"])
        self.assertIsNone(ratio(0, 0))
        self.assertFalse(evaluate_submission(None, workflow(), None)["submitted"])

    def test_database_matches_run_and_excludes_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("""CREATE TABLE submissions (run_id, workflow_id, content_version,
                    attention_check_content_version, ordered_question_ids, answers,
                    attention_check_pass_count, attention_check_fail_count, attention_check_scored_count,
                    attention_check_unscored_count, attention_check_attempted_count,
                    attention_check_skipped_count, attention_check_results)""")
                db.execute("INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           ("run", "workflow", 1, 1, '["q"]', '{"q":"a"}', 1, 0, 1, 1, 1, 1, '[]'))
            db.close()
            snapshot = {"status": 200, "request": {"runId": "run", "workflowId": "workflow",
                        "contentVersion": 1, "attentionCheckContentVersion": 1,
                        "orderedQuestionIds": ["q"], "answers": {"q": "a"}},
                        "response": {"accepted": True, "attemptedCount": 1, "validCount": 1,
                                     "invalidCount": 0, "skippedCount": 12}}
            score = evaluate_submission(snapshot, workflow(), path)
            self.assertEqual(score["attention_pass_rate"], 1)
            self.assertEqual(score["attention_unscored_count"], 1)
            snapshot["request"]["answers"]["q"] = "b"
            self.assertEqual(evaluate_submission(snapshot, workflow(), path)["attention_status"], "submission_mismatch")

    def test_report_includes_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {"model": "fake", "workflow_id": "o1", "question_count": 13,
                    "wall_seconds": 1, "model_seconds": 0.5, "model_calls": 1,
                    "plan_errors": 0, "action_errors": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "usage_reported_calls": 0}
            for index, success in enumerate([True, False]):
                folder = root / str(index)
                folder.mkdir()
                evaluation = {"submitted": success, "attention_status": "database_unavailable",
                              "valid_count": 13 if success else None, "attempted_count": 13,
                              "invalid_count": 0, "skipped_count": 0}
                write_json(folder / "run_summary.json", dict(base, evaluation=evaluation))
            row = report(root)[0]
            self.assertEqual(row["submission_rate"], 0.5)
            self.assertEqual(row["valid_response_rate_submitted"], 1)
            self.assertIsNone(row["attention_pass_rate"])

    def test_defaults(self):
        args = parser().parse_args(["run"])
        self.assertEqual(args.themes, ["all"])
        self.assertEqual(args.repeats, 1)
        self.assertEqual(len(args.orders), 3)
        self.assertIsInstance(args.runs_dir, Path)


if __name__ == "__main__":
    unittest.main()
