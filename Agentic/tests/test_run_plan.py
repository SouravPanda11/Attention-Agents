import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import main, parser, run_one
from plan_runner import completed_plan_ids, fetch_workflow, is_completed, read_plan, run_plan_batch
from brain import PROMPT_VERSION
from questions import EXECUTION_POLICY, QUESTION_TURN_LIMIT


def row(occurrence=2, layout="navigation", sample=1, order="order01", repeat=1):
    sample_id = f"o{occurrence}-s{sample:04d}"
    workflow_id = f"v1-standard-{sample_id}" + ("-item" if layout == "item" else "") + f"-{order}"
    return {"suiteVersion": "v1", "planId": f"{workflow_id}-repeat{repeat:03d}",
            "workflowId": workflow_id, "sampleId": sample_id, "condition": "attention-horizon",
            "occurrence": occurrence, "layout": layout, "orderId": order, "orderSeed": "seed",
            "repeatIndex": repeat, "questionCount": 13 * occurrence + 1,
            "pageCount": 1 if layout == "item" else occurrence,
            "url": f"/survey/samples/{sample_id}/{order}" + ("/item" if layout == "item" else "")}


def live_workflow(plan_row):
    result = dict(plan_row, id=plan_row["workflowId"], contentVersion=1,
                  attentionCheckContentVersion=1, hasWelcomePage=True)
    ids = [f"q{i}" for i in range(result["questionCount"])]
    result["orderedQuestionIds"] = ids
    result["pageQuestionIds"] = [ids] if result["pageCount"] == 1 else [ids[:13], ids[13:]]
    return result


class PlanTests(unittest.TestCase):
    def test_resume_recovers_old_completed_runs_and_saved_submissions(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(runs_dir=Path(temp), observation="dom", behavior="completion")
            rows = [row(1, repeat=i) for i in range(1, 6)]
            for i, planned in enumerate(rows):
                directory = args.runs_dir / "model" / "v1-plans" / "old-batch" / f"run-{i}"
                directory.mkdir(parents=True)
                summary = {"model": "test", "observation": "dom", "behavior": "completion", "temperature": 0,
                           "prompt_version": PROMPT_VERSION, "execution_policy": EXECUTION_POLICY,
                           "question_turn_limit": QUESTION_TURN_LIMIT, "plan_id": planned["planId"],
                           "workflow_id": planned["workflowId"], "evaluation": {"submitted": i == 0}}
                if i == 3:
                    summary["model"] = "different-model"
                    summary["evaluation"]["submitted"] = True
                (directory / "workflow.json").write_text(json.dumps(live_workflow(planned)), encoding="utf-8")
                (directory / "run_summary.json").write_text(json.dumps(summary) if i != 4 else "{partial", encoding="utf-8")
                if i == 1:
                    (directory / "submission_snapshot.json").write_text(json.dumps({"status": 200,
                        "response": {"accepted": True}, "request": {"workflowId": planned["workflowId"]}}), encoding="utf-8")
            completed = completed_plan_ids(args, "test")
            self.assertEqual([is_completed(r, completed) for r in rows], [True, True, False, False, False])
            self.assertFalse(is_completed(dict(rows[0], orderSeed="different-seed"), completed))
            args.behavior = "unconstrained"
            self.assertFalse(completed_plan_ids(args, "test"))

    def test_no_command_starts_default_plan_run(self):
        with patch("agent.run_batch", new_callable=AsyncMock, return_value=0) as batch:
            self.assertEqual(main([]), 0)
            args = batch.call_args.args[0]
            self.assertEqual(args.command, "run")
            self.assertFalse(args.theme_baselines)
            self.assertEqual(args.plan.name, "run-v1.jsonl")

    def test_default_run_uses_plan_and_baselines_are_explicit(self):
        args = parser().parse_args(["run"])
        self.assertFalse(args.theme_baselines)
        self.assertEqual(args.plan.name, "run-v1.jsonl")
        self.assertTrue(parser().parse_args(["run", "--theme-baselines"]).theme_baselines)

    def test_plan_order_and_invalid_rows(self):
        rows = [row(1), row(sample=1, repeat=1), row(sample=1, repeat=2),
                row(sample=2), row(layout="item"), row(3)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(list(read_plan(path)), rows)
            for invalid in ([rows[1], rows[1]], [row(layout="item"), row()],
                            [row(1, "item")], [dict(row(), url="https://example.com")],
                            [dict(row(), repeatIndex=0)], [dict(row(), questionCount=1)]):
                path.write_text("\n".join(map(json.dumps, invalid)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(read_plan(path))

    def test_dry_run_does_not_contact_servers_and_limit_does_not_mask_invalid_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.jsonl"
            path.write_text("\n".join(map(json.dumps, [row(1), row(), row(layout="item")])), encoding="utf-8")
            args = SimpleNamespace(plan=path, suite_version="v1", start_index=2, limit=1,
                                   dry_run=True, models=None, model="test")
            with patch("plan_runner.request_json") as request, patch("builtins.print") as output:
                self.assertEqual(asyncio.run(run_plan_batch(args, None)), 0)
                request.assert_not_called()
                summary = json.loads(output.call_args.args[0])
                self.assertEqual(summary["runs_per_model"], 1)
                self.assertEqual(summary["first_run"]["occurrence"], 2)
            path.write_text(json.dumps(row()) + "\n{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                asyncio.run(run_plan_batch(args, None))

    def test_live_manifest_must_match_plan(self):
        planned = row()
        live = live_workflow(planned)
        manifest = {"suiteVersion": "v1", "pagination": {"total": 1}, "workflows": [live]}
        args = SimpleNamespace(base_url="http://localhost:3001")
        with patch("plan_runner.request_json", return_value=manifest):
            self.assertEqual(asyncio.run(fetch_workflow(args, planned)), live)
            live["orderSeed"] = "different"
            with self.assertRaises(ValueError):
                asyncio.run(fetch_workflow(args, planned))


class PageExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_executes_file_repetitions_in_order_and_retains_schedule(self):
        rows = [row(1, repeat=1), row(1, repeat=2), row(), row(layout="item")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.jsonl"
            plan.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            args = SimpleNamespace(plan=plan, suite_version="v1", start_index=1, limit=None,
                                   dry_run=False, models=None, model="test", model_name="", headed=False,
                                   base_url="http://localhost:3001", lm_base_url="http://localhost:1234/v1",
                                   api_key="local", runs_dir=root / "runs")
            browser = SimpleNamespace(close=AsyncMock())
            playwright = SimpleNamespace(chromium=SimpleNamespace(launch=AsyncMock(return_value=browser)))
            # Special methods are resolved on the class for async context managers.
            class Manager:
                async def __aenter__(self):
                    return playwright

                async def __aexit__(self, *_):
                    return False

            invoked = []

            async def execute(browser, args, workflow, repeat, batch_dir, index):
                invoked.append((workflow["planId"], repeat, index, batch_dir))
                return {"evaluation": {"submitted": index != 3}}

            def request(url, *_):
                return {"data": [{"id": "test"}]} if url.endswith("/models") else {"suiteVersion": "v1"}

            with patch("playwright.async_api.async_playwright", return_value=Manager()), \
                    patch("plan_runner.request_json", side_effect=request), \
                    patch("plan_runner.fetch_workflow", side_effect=lambda args, row: live_workflow(row)) as fetch, \
                    patch("plan_runner.report") as reporting, patch("builtins.print"):
                self.assertEqual(await run_plan_batch(args, execute), 1)
                self.assertEqual(fetch.await_count, 3)  # Cached across adjacent repetitions.
                self.assertEqual(reporting.call_count, 5)
            self.assertEqual([call[:3] for call in invoked], [(row["planId"], row["repeatIndex"], i)
                                                            for i, row in enumerate(rows, 1)])
            schedule = next(args.runs_dir.rglob("schedule.jsonl"))
            saved = [json.loads(line) for line in schedule.read_text().splitlines()]
            self.assertEqual([r["planId"] for r in saved], [r["planId"] for r in rows])
            self.assertNotEqual(invoked[2][3], invoked[3][3])
            browser.close.assert_awaited_once()

            # Resume the same plan; skip submitted repetitions and retry the failed row.
            completed = {(r["planId"], r["orderSeed"], r["questionCount"]) for i, r in enumerate(rows, 1) if i != 3}
            args.resume = True
            invoked.clear()
            with patch("playwright.async_api.async_playwright", return_value=Manager()), \
                    patch("plan_runner.request_json", side_effect=request), \
                    patch("plan_runner.fetch_workflow", side_effect=lambda args, row: live_workflow(row)), \
                    patch("plan_runner.completed_plan_ids", return_value=completed), \
                    patch("plan_runner.report"), patch("builtins.print"):
                self.assertEqual(await run_plan_batch(args, execute), 1)
            self.assertEqual([call[2] for call in invoked], [3])
            self.assertEqual(len(list(args.runs_dir.rglob("experiment.json"))), 2)

            completed.add((rows[2]["planId"], rows[2]["orderSeed"], rows[2]["questionCount"]))
            with patch("plan_runner.completed_plan_ids", return_value=completed), \
                    patch("plan_runner.request_json") as network, patch("builtins.print"):
                self.assertEqual(await run_plan_batch(args, execute), 0)
                network.assert_not_called()

    async def test_navigation_and_item_answer_all_questions_then_submit(self):
        for layout in ("navigation", "item"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as temp:
                workflow = live_workflow(row(layout=layout))
                page = SimpleNamespace(set_default_timeout=lambda _: None,
                                       set_default_navigation_timeout=lambda _: None,
                                       goto=AsyncMock(), locator=lambda _: SimpleNamespace(wait_for=AsyncMock()))
                context = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
                browser = SimpleNamespace(new_context=AsyncMock(return_value=context))
                cursor = -1
                actions = []
                answered = []

                async def observe(_):
                    if cursor == -1:
                        return {"screen": "welcome", "instructions": []}
                    return {"screen": "questions", "instructions": [], "fields": [
                        {"key": key, "kind": "short-text", "prompt": key}
                        for key in workflow["pageQuestionIds"][cursor]]}

                async def execute(_, action):
                    nonlocal cursor
                    actions.append(action["key"])
                    if action["key"] in {"start-survey", "next-page"}:
                        cursor += 1
                    else:
                        self.assertEqual(answered, workflow["orderedQuestionIds"])
                        return {"status": 200, "response": {"accepted": True},
                                "request": {"runId": "test", "answers": {key: "yes" for key in answered}}}

                async def question(_, args, work, field, number, run_dir, summary, trace, history, instructions):
                    self.assertEqual(number, len(answered) + 1)
                    self.assertEqual(len(history), len(answered))
                    answered.append(field["key"])
                    return {"question_id": field["key"], "status": "answered", "reason": "valid_response", "answer": "yes"}

                args = SimpleNamespace(model="test", model_name="", observation="dom", behavior="completion",
                                       lm_base_url="http://localhost:1234/v1", base_url="http://localhost:3001", db_path=None)
                with patch("agent.observe", side_effect=observe), patch("agent.execute", side_effect=execute), \
                        patch("agent.run_question", side_effect=question), \
                        patch("agent.evaluate_submission", side_effect=lambda snapshot, *_: {"submitted": bool(snapshot)}), \
                        patch("builtins.print"):
                    summary = await run_one(browser, args, workflow, 1, Path(temp))
                self.assertEqual(summary["status"], "submitted")
                self.assertEqual(summary["answered_questions"], 27)
                self.assertEqual(actions, ["start-survey"] + (["next-page"] if layout == "navigation" else []) + ["submit-survey"])
                context.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
