"""Question-scoped model turns, public form validation, and unanswered rollback."""
import math
import time

from brain import complete, messages_for, parse_plan
from browser import card_for, execute, observe, validate_plan
from evaluation import write_json

QUESTION_TURN_LIMIT = 3
EXECUTION_POLICY = "per-question"


def answer_state(field):
    """Assess public input constraints, never semantic correctness or private keys."""
    value, kind = field.get("current"), field["kind"]
    if kind in {"single-radio", "likert", "image-single-select"}:
        return len(value) == 1, "Choose one option."
    if kind == "single-dropdown":
        return value in [o["value"] for o in field["options"]], "Choose a dropdown option."
    if kind in {"single-checkbox", "multiple-checkbox"}:
        minimum, maximum = field["minSelections"], field["maxSelections"]
        return minimum <= len(value) <= maximum, f"Select between {minimum} and {maximum} options."
    if kind == "ranking":
        return field.get("interacted", False), "Record the full ranking with rank."
    if kind == "slider" and not field.get("interacted"):
        return False, "Record a slider value with set_range."
    if kind in {"numeric", "slider"}:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False, "Enter a number."
        valid = (math.isfinite(number) and
                 ("min" not in field or number >= float(field["min"])) and
                 ("max" not in field or number <= float(field["max"])))
        return valid, "Enter a number within the displayed range."
    if kind in {"short-text", "long-text"}:
        length = len(value.strip())
        minimum, maximum = int(field["minlength"]), int(field["maxlength"])
        return minimum <= length <= maximum, f"Enter between {minimum} and {maximum} characters."
    raise ValueError(f"Unsupported question kind: {kind}")


async def current_field(page, question_id):
    observation = await observe(page)
    return next(field for field in observation["fields"] if field["key"] == question_id)


async def clear_unanswered(page, workflow_id, question_id):
    # Some controls (radio, ranking, slider) have no UI reset. Remove only this
    # answer from the app's persisted progress and let the app restore it normally.
    # This is harness rollback, never a model action or an answer-generation path.
    changed = await page.evaluate("""async ({workflowId, questionId}) => {
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const key = `survey-benchmark:${workflowId}`;
        const progress = JSON.parse(sessionStorage.getItem(key));
        if (!progress?.runId || !progress.started || progress.completed) {
            throw new Error('Cannot clear an answer outside an active survey');
        }
        if (!Object.hasOwn(progress.answers, questionId)) return false;
        delete progress.answers[questionId];
        sessionStorage.setItem(key, JSON.stringify(progress));
        return true;
    }""", {"workflowId": workflow_id, "questionId": question_id})
    if changed:
        await page.reload(wait_until="domcontentloaded")
    field = await current_field(page, question_id)
    if field["kind"] in {"slider", "ranking"}:
        cleared = not field.get("interacted")
    else:
        cleared = field["current"] in ("", [])
    if not cleared:
        raise RuntimeError(f"Failed to clear exhausted question {question_id}")


async def run_question(page, args, workflow, initial, number, run_dir, summary, trace, history, instructions):
    key = initial["key"]
    result = {"question_id": key, "kind": initial["kind"], "turns": 0,
              "status": "unanswered", "reason": "turn_budget_exhausted"}
    feedback = []
    for question_turn in range(1, QUESTION_TURN_LIMIT + 1):
        field = await current_field(page, key)
        observation = {"screen": "question", "current_question_id": key,
                       "question_number": number, "question_count": workflow["questionCount"],
                       "question_turn": question_turn, "question_turn_limit": QUESTION_TURN_LIMIT,
                       "instructions": instructions,
                       "interaction_scope": "Answer only the current question. The runner handles survey navigation.",
                       "previous_responses": history, "fields": [field]}
        prefix = run_dir / f"question-{number:02d}-turn-{question_turn}"
        write_json(prefix.with_suffix(".observation.json"), observation)
        images = []
        if args.observation == "vision" and field["kind"] == "image-single-select":
            screenshot = prefix.with_suffix(".png")
            await card_for(page, key).screenshot(path=str(screenshot))
            images.append(screenshot)
        messages = messages_for(observation, feedback, args.behavior, images)
        write_json(prefix.with_suffix(".request.json"), {"model": args.model,
                   "temperature": 0, "stream": False, "messages": messages})
        summary["model_calls"] += 1
        result["turns"] += 1
        event = {"turn": summary["model_calls"], "question_id": key,
                 "question_turn": question_turn, "actions": []}
        trace.append(event)
        print(f"  {args.model} | {workflow['orderId']} | question {number}/{workflow['questionCount']} "
              f"| turn {question_turn}/{QUESTION_TURN_LIMIT}", flush=True)
        call_started = time.perf_counter()
        try:
            response, elapsed = await complete(args, messages)
        except Exception as exc:
            summary["model_errors"] += 1
            feedback = [f"Model request failed: {type(exc).__name__}: {exc}"]
            event["model_error"] = feedback[0]
            write_json(run_dir / "trace.json", trace)
            continue
        finally:
            elapsed = time.perf_counter() - call_started
            summary["model_seconds"] += elapsed
            event["model_seconds"] = elapsed
        write_json(prefix.with_suffix(".response.json"), response)
        usage = response.get("usage") or {}
        event.update(usage=usage, returned_model=response.get("model"))
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            summary["usage_reported_calls"] += 1
            summary["prompt_tokens"] += usage["prompt_tokens"]
            summary["completion_tokens"] += usage["completion_tokens"]
        try:
            plan = parse_plan(response)
            validate_plan(plan, observation)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            summary["plan_errors"] += 1
            feedback = [f"Plan rejected before execution: {exc}"]
            event["plan_error"] = str(exc)
            write_json(run_dir / "trace.json", trace)
            continue
        write_json(prefix.with_suffix(".plan.json"), plan)
        feedback = []
        for action in plan:
            if action["tool"] == "done":
                break
            action_started = time.perf_counter()
            action_event = {"action": action}
            event["actions"].append(action_event)
            try:
                await execute(page, action)
                summary["actions_executed"] += 1
                action_event["ok"] = True
            except Exception as exc:
                summary["action_errors"] += 1
                action_event.update(ok=False, error=str(exc))
                feedback = [f"Action failed: {action}: {exc}. Earlier actions on this question are retained."]
                break
            finally:
                action_event["seconds"] = time.perf_counter() - action_started
        updated = await current_field(page, key)
        valid, requirement = answer_state(updated)
        event["answer_valid"] = valid
        write_json(run_dir / "trace.json", trace)
        if valid and not feedback:
            result.update(status="answered", reason="valid_response", answer=updated["current"])
            break
        if plan[0]["tool"] == "done" and args.behavior == "unconstrained":
            result["reason"] = "model_skipped"
            break
        feedback = feedback or [f"Question is still unanswered or invalid: {requirement}"]
        summary["invalid_answer_turns"] += 1
    if result["status"] == "unanswered":
        await clear_unanswered(page, workflow["id"], key)
        result["answer"] = None
    write_json(run_dir / f"question-{number:02d}.json", result)
    return result
