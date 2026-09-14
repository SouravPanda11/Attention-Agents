import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from questions import answer_state, run_question
from brain import complete


def response(plan):
    return {"choices": [{"message": {"content": json.dumps(plan)}}]}, 0.01


class QuestionBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = SimpleNamespace(model="unit-fixture", observation="dom", behavior="completion")
        self.field = {"key": "q", "kind": "short-text", "tool": "fill", "current": "",
                      "minlength": "1", "maxlength": "100", "prompt": "Describe your choice."}
        self.workflow = {"id": "workflow", "questionCount": 13, "orderId": "order01"}
        self.summary = dict.fromkeys(["model_calls", "model_seconds", "model_errors", "plan_errors",
                                      "action_errors", "actions_executed", "usage_reported_calls",
                                      "prompt_tokens", "completion_tokens", "invalid_answer_turns"], 0)

    async def run_fixture(self, model, execute_action=None, count=1):
        with tempfile.TemporaryDirectory() as directory:
            with patch("questions.current_field", AsyncMock(side_effect=lambda *a: dict(self.field))), \
                 patch("questions.card_for") as card, \
                 patch("questions.complete", model), \
                 patch("questions.execute", AsyncMock(side_effect=execute_action)), \
                 patch("questions.clear_unanswered", AsyncMock()) as clear, patch("builtins.print"):
                card.return_value.screenshot = AsyncMock()
                results = []
                for number in range(1, count + 1):
                    results.append(await run_question(None, self.args, self.workflow, self.field,
                                                      number, Path(directory), self.summary, [], [], []))
                if self.args.observation == "dom":
                    card.assert_not_called()
                    self.assertEqual(list(Path(directory).glob("*.png")), [])
                return results, clear.await_count

    async def test_thirteen_failed_questions_use_exactly_39_calls(self):
        model = AsyncMock(return_value=response([{"tool": "fill", "key": "other-question", "value": "x"}]))
        results, clears = await self.run_fixture(model, count=13)
        self.assertEqual(model.await_count, 39)
        self.assertEqual(clears, 13)
        self.assertTrue(all(r["turns"] == 3 and r["reason"] == "turn_budget_exhausted" for r in results))
        for call in model.call_args_list:
            observation = json.loads(call.args[1][1]["content"][0]["text"])["ACTION_SPACE"]
            self.assertEqual([f["key"] for f in observation["fields"]], ["q"])

    async def test_recovery_on_third_turn_is_retained(self):
        model = AsyncMock(side_effect=[response([{"tool": "bad", "key": "q"}]),
                                       response([{"tool": "done"}]),
                                       response([{"tool": "fill", "key": "q", "value": "Answer"}])])
        def action(*args):
            self.field["current"] = "Answer"
        results, clears = await self.run_fixture(model, action)
        self.assertEqual(results[0]["status"], "answered")
        self.assertEqual(results[0]["turns"], 3)
        self.assertEqual(clears, 0)

    async def test_eleven_theme_questions_use_at_most_33_calls(self):
        self.workflow["questionCount"] = 11
        model = AsyncMock(return_value=response([{"tool": "bad", "key": "q"}]))
        results, clears = await self.run_fixture(model, count=11)
        self.assertEqual(model.await_count, 33)
        self.assertEqual(clears, 11)
        self.assertTrue(all(r["status"] == "unanswered" for r in results))

    async def test_partial_action_failure_exhausts_and_clears(self):
        model = AsyncMock(return_value=response([{"tool": "fill", "key": "q", "value": "partial"}]))
        def action(*args):
            self.field["current"] = "partial"
            raise RuntimeError("browser failure after changing the control")
        results, clears = await self.run_fixture(model, action)
        self.assertEqual(results[0]["status"], "unanswered")
        self.assertEqual(clears, 1)
        self.assertEqual(self.summary["action_errors"], 3)

    async def test_request_failures_count_against_question_budget(self):
        results, clears = await self.run_fixture(AsyncMock(side_effect=ConnectionError("offline")))
        self.assertEqual(results[0]["turns"], 3)
        self.assertEqual(self.summary["model_errors"], 3)
        self.assertEqual(clears, 1)

    async def test_dom_image_question_never_captures_or_sends_images(self):
        self.field = {"key": "q", "kind": "image-single-select", "tool": "check", "current": [],
                      "options": [{"value": "a", "label": "A", "imageAlt": "A blue circle"}]}
        model = AsyncMock(return_value=response([{"tool": "check", "key": "q", "value": "a"}]))
        def action(*args):
            self.field["current"] = ["a"]
        results, _ = await self.run_fixture(model, action)
        self.assertEqual(results[0]["status"], "answered")
        content = model.call_args.args[1][1]["content"]
        self.assertEqual([part["type"] for part in content], ["text"])

    async def test_unconstrained_done_skips_only_current_question(self):
        self.args.behavior = "unconstrained"
        results, clears = await self.run_fixture(AsyncMock(return_value=response([{"tool": "done"}])) )
        self.assertEqual(results[0]["turns"], 1)
        self.assertEqual(results[0]["reason"], "model_skipped")
        self.assertEqual(clears, 1)

    async def test_model_request_omits_token_limit_and_timeout(self):
        args = SimpleNamespace(model="unit-fixture", lm_base_url="http://localhost:1234/v1", api_key="local")
        with patch("brain.request_json", return_value={}) as request:
            await complete(args, [{"role": "user", "content": "question"}])
        payload = request.call_args.args[1]
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("max_tokens", payload)
        self.assertEqual(len(request.call_args.args), 3)
        self.assertNotIn("timeout", request.call_args.kwargs)

    def test_public_constraints_require_real_interaction(self):
        self.assertFalse(answer_state({"kind": "slider", "current": "5", "interacted": False})[0])
        self.assertFalse(answer_state({"kind": "ranking", "current": ["a", "b"], "interacted": False})[0])
        self.assertFalse(answer_state({"kind": "numeric", "current": "99", "max": "10"})[0])
        self.assertFalse(answer_state({"kind": "multiple-checkbox", "current": ["a"],
                                      "minSelections": 2, "maxSelections": 2})[0])


if __name__ == "__main__":
    unittest.main()
