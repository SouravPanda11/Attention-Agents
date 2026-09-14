"""Read authoritative scores only after a run; never pass these to the planner."""
import csv
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from statistics import median, stdev


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def attempted(value):
    if isinstance(value, str):
        return bool(value.strip())
    return isinstance(value, (int, float, list)) and not isinstance(value, bool) and value != []


def evaluate_submission(snapshot, workflow, db_path):
    body = snapshot.get("response", {}) if snapshot else {}
    accepted = bool(snapshot and snapshot.get("status") == 200 and body.get("accepted") is True)
    result = {"submitted": accepted, "attention_status": "not_submitted",
              "attempted_count": None, "valid_count": None, "invalid_count": None,
              "skipped_count": None, "valid_response_rate": None,
              "attention_pass_count": None, "attention_scored_count": None,
              "attention_unscored_count": None, "attention_pass_rate": None}
    if workflow.get("attentionCheckCount") == 0:
        result.update(attention_status="not_applicable", attention_pass_count=0,
                      attention_scored_count=0, attention_unscored_count=0)
    if not accepted:
        return result
    for name in ("attempted", "valid", "invalid", "skipped"):
        result[f"{name}_count"] = body[f"{name}Count"]
    result["valid_response_rate"] = ratio(result["valid_count"], workflow["questionCount"])
    result["question_results"] = body.get("questionResults", [])
    result["all_questions_valid"] = result["valid_count"] == workflow["questionCount"]
    if workflow.get("attentionCheckCount") == 0:
        return result
    result["attention_status"] = "database_unavailable"
    if not db_path or not Path(db_path).is_file():
        return result
    request = snapshot["request"]
    try:
        with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute("SELECT * FROM submissions WHERE run_id = ? AND workflow_id = ?",
                             (request["runId"], request["workflowId"])).fetchone()
        if row is None:
            result["attention_status"] = "submission_not_found"
            return result
        if (row["content_version"] != request["contentVersion"] or
                row["attention_check_content_version"] != request["attentionCheckContentVersion"] or
                json.loads(row["ordered_question_ids"]) != request["orderedQuestionIds"] or
                json.loads(row["answers"]) != request["answers"]):
            result["attention_status"] = "submission_mismatch"
            return result
        result["attention_status"] = "scored"
        for name in ("pass", "fail", "scored", "unscored", "attempted", "skipped"):
            result[f"attention_{name}_count"] = row[f"attention_check_{name}_count"]
        result["attention_results"] = json.loads(row["attention_check_results"])
        result["attention_pass_rate"] = ratio(result["attention_pass_count"], result["attention_scored_count"])
    except (sqlite3.Error, KeyError, IndexError, ValueError) as exc:
        result["attention_status"] = "database_error"
        result["attention_error"] = str(exc)
    return result


def format_attempts(fields, snapshot):
    answers = snapshot.get("request", {}).get("answers", {}) if snapshot else {}
    return [{"question_id": field["key"], "kind": field["kind"],
             "attempted": attempted(answers.get(field["key"]))} for field in fields if "kind" in field]


PROTOCOL_KEYS = ("model", "lm_base_url", "observation", "behavior", "temperature", "execution_policy", "question_turn_limit",
                 "prompt_version", "condition", "content_version", "attention_check_content_version")
GROUP_KEYS = PROTOCOL_KEYS + ("theme_id", "order_id", "workflow_id")


def aggregate(summaries, group_keys):
    groups = defaultdict(list)
    for run in summaries:
        groups[tuple(run.get(key) for key in group_keys)].append(run)
    rows = []
    for key, runs in groups.items():
        submitted = [run for run in runs if run["evaluation"]["submitted"]]
        scored = [run for run in runs if run["evaluation"]["attention_status"] == "scored"]
        evaluated_questions = sum(run["question_count"] for run in submitted)
        valid = sum(run["evaluation"]["valid_count"] for run in submitted)
        passes = sum(run["evaluation"]["attention_pass_count"] for run in scored)
        checks = sum(run["evaluation"]["attention_scored_count"] for run in scored)
        row = dict(zip(group_keys, key))
        row.update(runs=len(runs), submitted=len(submitted), submission_rate=ratio(len(submitted), len(runs)),
                   valid_responses=valid, submitted_questions=evaluated_questions,
                   valid_response_rate_submitted=ratio(valid, evaluated_questions),
                   attempted_responses=sum(r["evaluation"]["attempted_count"] for r in submitted),
                   invalid_responses=sum(r["evaluation"]["invalid_count"] for r in submitted),
                   skipped_responses=sum(r["evaluation"]["skipped_count"] for r in submitted),
                   all_questions_valid_runs=sum(r["evaluation"]["valid_count"] == r["question_count"] for r in submitted),
                   all_questions_valid_rate=sum(r["evaluation"]["valid_count"] == r["question_count"] for r in submitted) / len(runs),
                   end_to_end_valid_rate=ratio(valid, sum(r["question_count"] for r in runs)),
                   attention_evaluated_runs=len(scored), attention_passes=passes, attention_scored=checks,
                   attention_pass_rate=ratio(passes, checks),
                   mean_wall_seconds=sum(r["wall_seconds"] for r in runs) / len(runs),
                   median_wall_seconds=median(r["wall_seconds"] for r in runs),
                   stdev_wall_seconds=stdev(r["wall_seconds"] for r in runs) if len(runs) > 1 else None,
                   mean_model_seconds=sum(r["model_seconds"] for r in runs) / len(runs),
                   model_calls=sum(r["model_calls"] for r in runs),
                   model_errors=sum(r.get("model_errors", 0) for r in runs),
                   budget_exhausted_questions=sum(r.get("budget_exhausted_questions", 0) for r in runs),
                   invalid_answer_turns=sum(r.get("invalid_answer_turns", 0) for r in runs),
                   plan_errors=sum(r["plan_errors"] for r in runs),
                   action_errors=sum(r["action_errors"] for r in runs),
                   prompt_tokens=sum(r["prompt_tokens"] for r in runs),
                   completion_tokens=sum(r["completion_tokens"] for r in runs),
                   usage_reported_calls=sum(r["usage_reported_calls"] for r in runs))
        rows.append(row)
    return rows


def write_table(root, name, rows):
    write_json(root / f"{name}.json", rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
        if columns:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def report(root):
    paths = sorted(root.rglob("run_summary.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    rows = aggregate(summaries, GROUP_KEYS)
    root.mkdir(parents=True, exist_ok=True)
    write_table(root, "comparison", rows)
    write_table(root, "theme_comparison", aggregate(summaries, PROTOCOL_KEYS + ("theme_id",)))
    write_table(root, "model_comparison", aggregate(summaries, PROTOCOL_KEYS))
    run_rows, question_rows = [], []
    for path, run in zip(paths, summaries):
        scalar = {key: value for key, value in run.items() if not isinstance(value, (dict, list))}
        scalar.update({key: value for key, value in run["evaluation"].items() if not isinstance(value, (dict, list))})
        scalar["run_dir"] = str(path.parent.relative_to(root))
        run_rows.append(scalar)
        attempts = {q["question_id"]: q for q in run.get("question_results", [])}
        for item in run["evaluation"].get("question_results", []):
            question_rows.append({**{key: run.get(key) for key in PROTOCOL_KEYS},
                                  "theme_id": run.get("theme_id"), "order_id": run.get("order_id"),
                                  "run_dir": scalar["run_dir"], "repeat": run.get("repeat"),
                                  "question_id": item["questionId"], "kind": item["kind"],
                                  "status": item["status"], "turns": attempts.get(item["questionId"], {}).get("turns")})
    write_table(root, "run_metrics", run_rows)
    write_table(root, "question_metrics", question_rows)
    by_format = defaultdict(list)
    format_keys = PROTOCOL_KEYS + ("theme_id", "kind")
    for question in question_rows:
        by_format[tuple(question.get(key) for key in format_keys)].append(question)
    format_rows = []
    for keys, questions in by_format.items():
        turns = [q["turns"] for q in questions if q["turns"] is not None]
        format_rows.append({**dict(zip(format_keys, keys)), "submitted_items": len(questions),
                            **{f"{status}_count": sum(q["status"] == status for q in questions)
                               for status in ("valid", "invalid", "skipped")},
                            "valid_rate": sum(q["status"] == "valid" for q in questions) / len(questions),
                            "mean_turns": sum(turns) / len(turns) if turns else None})
    write_table(root, "format_comparison", format_rows)
    return rows


def reevaluate(root, db_path):
    count = 0
    for path in root.rglob("run_summary.json"):
        snapshot_path = path.parent / "submission_snapshot.json"
        if not snapshot_path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        workflow = json.loads((path.parent / "workflow.json").read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        run["evaluation"] = evaluate_submission(snapshot, workflow, db_path)
        write_json(path.parent / "evaluation.json", run["evaluation"])
        write_json(path, run)
        count += 1
    return count
