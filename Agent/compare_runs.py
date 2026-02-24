import json
import csv
import sqlite3
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_stack import ensure_env_loaded, get_model_name_for_path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------------
# Editable in-code configuration
# -----------------------------
ensure_env_loaded()
RUNS_ROOT = str(Path(__file__).resolve().parent / "runs")
SURVEY_VERSION = os.getenv("SURVEY_VERSION", "").strip()
MODEL_NAME = get_model_name_for_path()
# "all" | "completion" | "unconstrained"
MODE_FILTER = "unconstrained"
# Optional JSON export path. Keep empty to disable.
JSON_OUT = ""
# CSV export path. Keep empty to disable.
CSV_OUT = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_compare_runs.csv")
# Plot output controls
WRITE_PLOTS = True
PLOTS_DIR = str(Path(__file__).resolve().parent / "runs" / SURVEY_VERSION / MODEL_NAME / "_compare_plots")
# Backend-validation controls
DB_PATH = str(Path(__file__).resolve().parent.parent / "survey-site" / "data.sqlite")
# External answer-key file used only during offline evaluation.
ANSWER_KEY_PATH = str(Path(__file__).resolve().parent.parent / "evaluation" / "answer_key.json")

TEXT_QUESTION_IDS = [
    "age",
    "follow_by_age",
    "employment_status",
    "watch_time",
    "trust_coverage",
    "country_satisfaction",
    "future_interest",
    "favorite_moment",
    "mood_impact",
]
IMAGE_QUESTION_IDS = ["image_q1", "image_q2", "image_q3", "image_q4"]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _common_prefix_len(a: List[str], b: List[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _sequence_ratio(a: List[str], b: List[str]) -> float:
    return SequenceMatcher(a=a, b=b).ratio() if (a or b) else 1.0


def _tri(v: Any) -> str:
    if v is True:
        return "Y"
    if v is False:
        return "N"
    return "-"


@dataclass
class RunRecord:
    mode: str
    run_name: str
    run_dir: Path
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    run_start_utc: Optional[datetime] = None
    run_end_utc: Optional[datetime] = None
    backend_validation: Dict[str, Any] = field(default_factory=dict)


def _default_csv_out(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_compare_runs.csv")


def _default_plots_dir(survey_version: str, model_name: str) -> str:
    return str(Path(__file__).resolve().parent / "runs" / survey_version / model_name / "_compare_plots")


def _count_mode_runs(model_dir: Path, mode: str) -> int:
    mode_dir = model_dir / mode
    if not mode_dir.exists():
        return 0
    count = 0
    for run_dir in mode_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if (run_dir / "run_summary.json").exists():
            count += 1
    return count


def _resolve_model_name(
    root: Path,
    survey_version: str,
    configured_model_name: str,
    modes: List[str],
) -> str:
    survey_dir = root / survey_version
    if not survey_dir.exists():
        return configured_model_name

    candidates = [p for p in survey_dir.iterdir() if p.is_dir()]
    if not candidates:
        return configured_model_name

    def _score(model_dir: Path) -> int:
        return sum(_count_mode_runs(model_dir, mode) for mode in modes)

    configured_dir = survey_dir / configured_model_name
    configured_score = _score(configured_dir) if configured_model_name and configured_dir.exists() else -1
    if configured_score > 0:
        return configured_model_name

    scored = [(p, _score(p)) for p in candidates]
    scored_with_runs = [item for item in scored if item[1] > 0]
    if scored_with_runs:
        # Prefer the model with the most valid runs; tiebreak by latest update time.
        chosen = sorted(
            scored_with_runs,
            key=lambda item: (item[1], item[0].stat().st_mtime),
            reverse=True,
        )[0][0]
        return chosen.name

    # No model has valid runs yet; fall back to latest folder.
    chosen = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return chosen.name


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _load_answer_key(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {}
    surveys = payload.get("surveys")
    if not isinstance(surveys, dict):
        return {}
    return payload


def _get_survey_answer_key(answer_key: Dict[str, Any], survey_version: str) -> Dict[str, Any]:
    surveys = answer_key.get("surveys")
    if not isinstance(surveys, dict):
        return {}
    survey_payload = surveys.get(survey_version)
    return survey_payload if isinstance(survey_payload, dict) else {}


def _parse_json_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _is_answered_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _extract_run_window(run_dir: Path) -> tuple[Optional[datetime], Optional[datetime]]:
    trace = _load_json(run_dir / "trace.json")
    if not isinstance(trace, list):
        return None, None
    ts_values: List[datetime] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        parsed = _parse_iso_utc(item.get("ts_utc"))
        if parsed:
            ts_values.append(parsed)
    if not ts_values:
        return None, None
    return min(ts_values), max(ts_values)


def _load_backend_submissions(db_path: Path, survey_version: str) -> List[Dict[str, Any]]:
    if not db_path.exists():
        return []
    event_type = "submit_v1" if survey_version == "survey_v1" else "submit"
    rows: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.ts,
                s.session_id,
                s.text_answers,
                s.text_attention_value,
                s.image_answers,
                s.captcha_input,
                s.image_attention_choice
            FROM submissions s
            WHERE EXISTS (
                SELECT 1
                FROM events e
                WHERE e.session_id = s.session_id
                  AND e.event_type = ?
            )
            ORDER BY s.ts ASC
            """,
            (event_type,),
        )
        for ts, sid, text_answers, text_attention_value, image_answers, captcha_input, image_attention_choice in cur.fetchall():
            parsed_ts = _parse_iso_utc(ts)
            if not parsed_ts:
                continue
            rows.append(
                {
                    "ts_utc": parsed_ts,
                    "session_id": str(sid or ""),
                    "text_answers": _parse_json_dict(text_answers),
                    "text_attention_value": str(text_attention_value or ""),
                    "image_answers": _parse_json_dict(image_answers),
                    "captcha_input": str(captcha_input or ""),
                    "image_attention_choice": str(image_attention_choice or ""),
                }
            )
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return rows


def _load_captcha_by_session(db_path: Path, survey_version: str) -> Dict[str, str]:
    if not db_path.exists():
        return {}
    event_type = "captcha_issued_v1" if survey_version == "survey_v1" else "captcha_issued"
    by_session: Dict[str, tuple[datetime, str]] = {}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts, session_id, payload
            FROM events
            WHERE event_type = ?
            ORDER BY ts ASC
            """,
            (event_type,),
        )
        for ts, sid, payload in cur.fetchall():
            sid_str = str(sid or "").strip()
            if not sid_str:
                continue
            ts_utc = _parse_iso_utc(ts)
            if ts_utc is None:
                continue
            parsed_payload = _parse_json_dict(payload)
            captcha_code = str(parsed_payload.get("captcha_code") or "").strip().upper()
            if not captcha_code:
                continue
            current = by_session.get(sid_str)
            if current is None or ts_utc >= current[0]:
                by_session[sid_str] = (ts_utc, captcha_code)
    except Exception:
        return {}
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return {sid: code for sid, (_, code) in by_session.items()}


def _attach_backend_validation(
    records: List[RunRecord],
    db_path: Path,
    survey_version: str,
    answer_key: Dict[str, Any],
) -> None:
    submissions = _load_backend_submissions(db_path=db_path, survey_version=survey_version)
    if not submissions:
        return

    survey_key = _get_survey_answer_key(answer_key=answer_key, survey_version=survey_version)
    text_attention_expected = str(
        ((survey_key.get("text_attention") or {}).get("expected_value") or "")
    ).strip()
    image_attention_expected = str(
        ((survey_key.get("image_attention") or {}).get("expected_option_id") or "")
    ).strip()
    image_expected = (survey_key.get("image_questions") or {})
    image_expected = image_expected if isinstance(image_expected, dict) else {}
    captcha_by_session = _load_captcha_by_session(db_path=db_path, survey_version=survey_version)

    def _normalized(value: Any) -> str:
        return str(value or "").strip().lower()

    def _eq_expected(actual: Any, expected: str) -> Optional[bool]:
        if not expected:
            return None
        return _normalized(actual) == _normalized(expected)

    submissions_by_session: Dict[str, Dict[str, Any]] = {}
    for sub in submissions:
        sid = str(sub.get("session_id") or "").strip()
        if not sid:
            continue
        current = submissions_by_session.get(sid)
        if current is None or sub["ts_utc"] >= current["ts_utc"]:
            submissions_by_session[sid] = sub

    for rec in records:
        run_session_id = str(rec.summary.get("session_id") or "").strip()
        if not run_session_id:
            continue
        matched = submissions_by_session.get(run_session_id)
        if matched is None:
            continue

        text_answers = matched.get("text_answers") if isinstance(matched.get("text_answers"), dict) else {}
        image_answers = matched.get("image_answers") if isinstance(matched.get("image_answers"), dict) else {}
        matched_session = run_session_id
        issued_captcha = str(captcha_by_session.get(matched_session) or "").strip().upper()
        submitted_captcha = str(matched.get("captcha_input") or "").strip().upper()

        text_attention_ok = _eq_expected(matched.get("text_attention_value"), text_attention_expected)
        image_attention_ok = _eq_expected(matched.get("image_attention_choice"), image_attention_expected)
        captcha_ok: Optional[bool] = None
        if issued_captcha:
            captcha_ok = submitted_captcha == issued_captcha

        text_answered_count = sum(1 for qid in TEXT_QUESTION_IDS if _is_answered_value(text_answers.get(qid)))
        text_total_count = len(TEXT_QUESTION_IDS)

        image_answered_count = sum(1 for qid in IMAGE_QUESTION_IDS if _is_answered_value(image_answers.get(qid)))
        image_total_count = len(IMAGE_QUESTION_IDS)
        image_correct_count = sum(
            1
            for qid, expected in image_expected.items()
            if str(image_answers.get(qid) or "").strip() == expected
        )
        image_known_count = len(image_expected)
        image_accuracy_known = (image_correct_count / image_known_count) if image_known_count else None
        image_questions_all_correct: Optional[bool]
        if image_known_count <= 0:
            image_questions_all_correct = None
        else:
            image_questions_all_correct = image_correct_count == image_known_count

        overall_ok: Optional[bool]
        if (
            text_attention_ok is None
            or image_attention_ok is None
            or captcha_ok is None
            or image_questions_all_correct is None
        ):
            overall_ok = None
        else:
            overall_ok = bool(
                text_attention_ok
                and image_attention_ok
                and captcha_ok
                and image_questions_all_correct
            )

        rec.backend_validation = {
            "session_id": matched_session,
            "submission_ts_utc": matched["ts_utc"].isoformat(),
            "match_delta_s": None,
            "text_attention_value": matched.get("text_attention_value"),
            "text_attention_ok": text_attention_ok,
            "captcha_input": matched.get("captcha_input"),
            "captcha_expected": issued_captcha,
            "captcha_ok": captcha_ok,
            "image_attention_choice": matched.get("image_attention_choice"),
            "image_attention_ok": image_attention_ok,
            "overall_ok": overall_ok,
            "text_completion_answered": text_answered_count,
            "text_completion_total": text_total_count,
            "text_completion_rate": (text_answered_count / text_total_count) if text_total_count else None,
            "image_completion_answered": image_answered_count,
            "image_completion_total": image_total_count,
            "image_completion_rate": (image_answered_count / image_total_count) if image_total_count else None,
            "image_accuracy_correct": image_correct_count,
            "image_accuracy_known_total": image_known_count,
            "image_accuracy_rate": image_accuracy_known,
            "image_questions_all_correct": image_questions_all_correct,
        }


def _derive_metrics_from_summary(summary: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    exec_steps = summary.get("exec_steps_in_order") or []
    answer_order: List[str] = []
    seen = set()
    next_click_count = 0
    submit_click_count = 0
    for step in exec_steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("tool") or "") == "click":
            selector = str(step.get("selector") or "").lower()
            if "next" in selector:
                next_click_count += 1
            if "submit" in selector:
                submit_click_count += 1
        key = str(step.get("field_key") or step.get("question_text") or step.get("field_label") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        answer_order.append(key)

    reached_thank_you = None
    submitted = None
    trace = _load_json(run_dir / "trace.json")
    if isinstance(trace, list):
        urls: List[str] = []
        for event in trace:
            if not isinstance(event, dict):
                continue
            for k in ("url", "new_url", "current_url"):
                val = event.get(k)
                if isinstance(val, str) and val:
                    urls.append(val)
        lowered = [u.lower() for u in urls]
        reached_thank_you = any("/thank-you" in u for u in lowered)
        reached_done = any("/done" in u for u in lowered)
        submitted = reached_done
    return {
        "total_exec_steps": len(exec_steps),
        "unique_answer_items_touched": len(answer_order),
        "answer_order": answer_order,
        "next_click_count": next_click_count,
        "submit_click_count": submit_click_count,
        "reached_thank_you": reached_thank_you,
        "submitted": submitted,
    }


def _run_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(r"run_(\d+)", path.name)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def _collect_runs(root: Path, survey_version: str, model_name: str, mode: str) -> List[RunRecord]:
    mode_dir = root / survey_version / model_name / mode
    if not mode_dir.exists():
        return []

    records: List[RunRecord] = []
    for run_dir in sorted([p for p in mode_dir.iterdir() if p.is_dir()], key=_run_sort_key):
        summary = _load_json(run_dir / "run_summary.json")
        if not summary:
            continue

        metrics = _load_json(run_dir / "run_metrics.json") or _derive_metrics_from_summary(summary, run_dir=run_dir)
        run_start_utc, run_end_utc = _extract_run_window(run_dir)
        records.append(
            RunRecord(
                mode=mode,
                run_name=run_dir.name,
                run_dir=run_dir,
                metrics=metrics,
                summary=summary,
                run_start_utc=run_start_utc,
                run_end_utc=run_end_utc,
            )
        )
    return records

def _image_acc_text_from_backend(backend: Dict[str, Any]) -> str:
    known = int(backend.get("image_accuracy_known_total") or 0)
    correct = int(backend.get("image_accuracy_correct") or 0)
    acc = backend.get("image_accuracy_rate")
    if known <= 0 and acc is None:
        return "-"
    if not isinstance(acc, (int, float)):
        return f"{correct}/{known}"
    return f"{correct}/{known}({float(acc):.2f})"


def _completion_text_from_backend(backend: Dict[str, Any], prefix: str) -> str:
    answered = int(backend.get(f"{prefix}_completion_answered") or 0)
    total = int(backend.get(f"{prefix}_completion_total") or 0)
    rate = backend.get(f"{prefix}_completion_rate")
    if total <= 0:
        return "-"
    if not isinstance(rate, (int, float)):
        return f"{answered}/{total}"
    return f"{answered}/{total}({float(rate):.2f})"


def _print_mode_report(mode: str, records: List[RunRecord]):
    print(f"\n=== Mode: {mode} ===")
    if not records:
        print("No runs found.")
        return

    baseline_order = records[0].metrics.get("answer_order", []) or []
    header = (
        "run",
        "text_comp",
        "text_attn",
        "img_comp",
        "img_acc",
        "img_all",
        "img_attn",
        "captcha",
        "overall",
        "thankyou",
        "submitted",
        "match_s",
        "exec",
        "uniq_ans",
        "next",
        "submit",
        "ord_same",
        "ord_pref",
        "ord_jacc",
        "ord_seq",
    )
    print(" | ".join(header))
    print("-" * 170)

    for rec in records:
        order = rec.metrics.get("answer_order", []) or []
        same = order == baseline_order
        pref = _common_prefix_len(order, baseline_order)
        jacc = _jaccard(order, baseline_order)
        seq = _sequence_ratio(order, baseline_order)

        backend = rec.backend_validation or {}
        img_attn_ok = backend.get("image_attention_ok")
        txt_attn_ok = backend.get("text_attention_ok")
        captcha_ok = backend.get("captcha_ok")
        overall_ok = backend.get("overall_ok")
        image_all_ok = backend.get("image_questions_all_correct")
        submitted_flag = rec.metrics.get("submitted")
        if backend.get("session_id"):
            submitted_flag = True
        elif submitted_flag is None and isinstance(overall_ok, bool):
            submitted_flag = True
        match_delta = backend.get("match_delta_s")
        match_text = "-" if not isinstance(match_delta, (int, float)) else f"{float(match_delta):.1f}"
        text_comp = _completion_text_from_backend(backend, "text")
        image_comp = _completion_text_from_backend(backend, "image")
        img_acc_text = _image_acc_text_from_backend(backend)

        row = (
            rec.run_name,
            text_comp,
            _tri(txt_attn_ok),
            image_comp,
            img_acc_text,
            _tri(image_all_ok),
            _tri(img_attn_ok),
            _tri(captcha_ok),
            _tri(overall_ok),
            _tri(rec.metrics.get("reached_thank_you")),
            _tri(submitted_flag),
            match_text,
            str(rec.metrics.get("total_exec_steps", 0)),
            str(rec.metrics.get("unique_answer_items_touched", 0)),
            str(rec.metrics.get("next_click_count", 0)),
            str(rec.metrics.get("submit_click_count", 0)),
            _tri(same),
            str(pref),
            f"{jacc:.2f}",
            f"{seq:.2f}",
        )
        print(" | ".join(row))


def _build_json_report(records_by_mode: Dict[str, List[RunRecord]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"modes": {}}
    for mode, records in records_by_mode.items():
        if not records:
            out["modes"][mode] = {"runs": []}
            continue
        baseline_order = records[0].metrics.get("answer_order", []) or []
        runs_payload = []
        for rec in records:
            order = rec.metrics.get("answer_order", []) or []
            runs_payload.append(
                {
                    "run_name": rec.run_name,
                    "run_dir": str(rec.run_dir),
                    "metrics": rec.metrics,
                    "backend_validation": rec.backend_validation,
                    "order_compare_to_baseline": {
                        "same_order": order == baseline_order,
                        "common_prefix_len": _common_prefix_len(order, baseline_order),
                        "jaccard_overlap": _jaccard(order, baseline_order),
                        "sequence_ratio": _sequence_ratio(order, baseline_order),
                    },
                }
            )
        out["modes"][mode] = {"baseline_run": records[0].run_name, "runs": runs_payload}
    return out


def _build_csv_rows(records_by_mode: Dict[str, List[RunRecord]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode, records in records_by_mode.items():
        baseline_order = records[0].metrics.get("answer_order", []) or [] if records else []
        for rec in records:
            order = rec.metrics.get("answer_order", []) or []
            backend = rec.backend_validation or {}
            img_attn_ok = backend.get("image_attention_ok")
            txt_attn_ok = backend.get("text_attention_ok")
            captcha_ok = backend.get("captcha_ok")
            overall_ok = backend.get("overall_ok")
            text_answered = backend.get("text_completion_answered")
            text_total = backend.get("text_completion_total")
            text_rate = backend.get("text_completion_rate")

            image_answered = backend.get("image_completion_answered")
            image_total = backend.get("image_completion_total")
            image_rate = backend.get("image_completion_rate")

            img_known = backend.get("image_accuracy_known_total")
            img_correct = backend.get("image_accuracy_correct")
            img_rate = backend.get("image_accuracy_rate")
            submitted_flag = rec.metrics.get("submitted")
            if backend.get("session_id"):
                submitted_flag = True
            elif submitted_flag is None and isinstance(overall_ok, bool):
                submitted_flag = True
            rows.append(
                {
                    "mode": mode,
                    "run_name": rec.run_name,
                    "run_dir": str(rec.run_dir),
                    "text_completion_answered": text_answered,
                    "text_completion_total": text_total,
                    "text_completion_rate": text_rate,
                    "text_attention_ok": txt_attn_ok,
                    "image_completion_answered": image_answered,
                    "image_completion_total": image_total,
                    "image_completion_rate": image_rate,
                    "image_accuracy_correct": img_correct,
                    "image_accuracy_known_total": img_known,
                    "image_accuracy_rate": img_rate,
                    "image_questions_all_correct": backend.get("image_questions_all_correct"),
                    "image_attention_ok": img_attn_ok,
                    "captcha_ok": captcha_ok,
                    "overall_ok": overall_ok,
                    "reached_thank_you": rec.metrics.get("reached_thank_you"),
                    "submitted": submitted_flag,
                    "validation_session_id": backend.get("session_id"),
                    "validation_submission_ts_utc": backend.get("submission_ts_utc"),
                    "validation_match_delta_s": backend.get("match_delta_s"),
                    "total_exec_steps": rec.metrics.get("total_exec_steps", 0),
                    "unique_answer_items_touched": rec.metrics.get("unique_answer_items_touched", 0),
                    "next_click_count": rec.metrics.get("next_click_count", 0),
                    "submit_click_count": rec.metrics.get("submit_click_count", 0),
                    "order_same_as_baseline": order == baseline_order,
                    "order_common_prefix_len": _common_prefix_len(order, baseline_order),
                    "order_jaccard_overlap": _jaccard(order, baseline_order),
                    "order_sequence_ratio": _sequence_ratio(order, baseline_order),
                }
            )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "run_name",
        "run_dir",
        "text_completion_answered",
        "text_completion_total",
        "text_completion_rate",
        "text_attention_ok",
        "image_completion_answered",
        "image_completion_total",
        "image_completion_rate",
        "image_accuracy_correct",
        "image_accuracy_known_total",
        "image_accuracy_rate",
        "image_questions_all_correct",
        "image_attention_ok",
        "captcha_ok",
        "overall_ok",
        "reached_thank_you",
        "submitted",
        "validation_session_id",
        "validation_submission_ts_utc",
        "validation_match_delta_s",
        "total_exec_steps",
        "unique_answer_items_touched",
        "next_click_count",
        "submit_click_count",
        "order_same_as_baseline",
        "order_common_prefix_len",
        "order_jaccard_overlap",
        "order_sequence_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _csv_filename_for_mode(mode: str) -> str:
    if mode == "completion":
        return "completed_compared_runs.csv"
    if mode == "unconstrained":
        return "unconstrained_compare_runs.csv"
    return f"{mode}_compare_runs.csv"


def _csv_path_for_mode(base_path: Path, mode: str) -> Path:
    if base_path.name == "_compare_runs.csv":
        return base_path.with_name(_csv_filename_for_mode(mode))
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_{mode}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}_{mode}")


def _plot_mode_table(
    mode: str,
    records: List[RunRecord],
    out_dir: Path,
    survey_version: str,
    model_name: str,
) -> Optional[Path]:
    if plt is None:
        return None
    if not records:
        return None

    col_labels = [
        "text_completion",
        "text_attention_ok",
        "image_completion",
        "image_accuracy",
        "image_all_correct",
        "image_attention_ok",
        "captcha_ok",
        "reached_thank_you",
        "submitted",
    ]
    row_labels = [r.run_name for r in records]
    rows: List[List[str]] = []
    for rec in records:
        backend = rec.backend_validation or {}
        img_acc_text = "-"
        backend_img_acc = backend.get("image_accuracy_rate")
        if isinstance(backend_img_acc, (int, float)):
            img_acc_text = f"{float(backend_img_acc):.2f}"
        text_comp = _completion_text_from_backend(backend, "text")
        image_comp = _completion_text_from_backend(backend, "image")
        img_attn_ok = backend.get("image_attention_ok")
        txt_attn_ok = backend.get("text_attention_ok")
        captcha_ok = backend.get("captcha_ok")
        reached = rec.metrics.get("reached_thank_you")
        submitted = rec.metrics.get("submitted")
        if backend.get("session_id"):
            submitted = True
        elif submitted is None and backend.get("overall_ok") is not None:
            submitted = True
        rows.append(
            [
                text_comp,
                _tri(txt_attn_ok),
                image_comp,
                img_acc_text,
                _tri(backend.get("image_questions_all_correct")),
                _tri(img_attn_ok),
                _tri(captcha_ok),
                _tri(reached),
                _tri(submitted),
            ]
        )

    # Height scales with number of runs to keep table readable.
    fig_h = max(4.0, 1.2 + 0.45 * len(records))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    ax.set_title(
        f"Compare Runs - {survey_version} - {model_name} - {mode}",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_outcomes_table.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    if MODE_FILTER not in {"all", "completion", "unconstrained"}:
        raise SystemExit("MODE_FILTER must be one of: all, completion, unconstrained")
    if not SURVEY_VERSION:
        raise SystemExit("SURVEY_VERSION must be set in environment (for example in Agent/.env).")

    root = Path(RUNS_ROOT)
    modes = ["completion", "unconstrained"] if MODE_FILTER == "all" else [MODE_FILTER]
    resolved_model_name = _resolve_model_name(
        root=root,
        survey_version=SURVEY_VERSION,
        configured_model_name=MODEL_NAME,
        modes=modes,
    )
    records_by_mode: Dict[str, List[RunRecord]] = {
        mode: _collect_runs(
            root=root,
            survey_version=SURVEY_VERSION,
            model_name=resolved_model_name,
            mode=mode,
        )
        for mode in modes
    }
    answer_key = _load_answer_key(Path(ANSWER_KEY_PATH))
    db_file = Path(DB_PATH)
    for mode in modes:
        _attach_backend_validation(
            records=records_by_mode[mode],
            db_path=db_file,
            survey_version=SURVEY_VERSION,
            answer_key=answer_key,
        )

    print(f"runs_root={root}")
    print(f"survey_version={SURVEY_VERSION}")
    print(f"model_name_configured={MODEL_NAME}")
    print(f"model_name_resolved={resolved_model_name}")
    print(f"backend_validation_db={Path(DB_PATH)}")
    print(f"answer_key_path={Path(ANSWER_KEY_PATH)}")
    if not answer_key:
        print("warning: answer key not found/invalid; attention and accuracy checks may be incomplete.")
    if resolved_model_name != MODEL_NAME:
        print("note: configured model has no valid runs for this survey_version/mode; using resolved model folder.")
    if WRITE_PLOTS and plt is None:
        print("Plotting disabled: matplotlib is not available in this Python environment.")
    plot_paths: List[Path] = []
    configured_plot_default = _default_plots_dir(SURVEY_VERSION, MODEL_NAME)
    effective_plot_dir = _default_plots_dir(SURVEY_VERSION, resolved_model_name) if PLOTS_DIR == configured_plot_default else PLOTS_DIR
    plot_dir = Path(effective_plot_dir)
    for mode in modes:
        _print_mode_report(mode, records_by_mode[mode])
        if WRITE_PLOTS:
            p = _plot_mode_table(
                mode=mode,
                records=records_by_mode[mode],
                out_dir=plot_dir,
                survey_version=SURVEY_VERSION,
                model_name=resolved_model_name,
            )
            if p:
                plot_paths.append(p)

    if WRITE_PLOTS and plt is not None:
        if plot_paths:
            print(f"\nWrote comparison plots to: {plot_dir.resolve()}")
            for p in plot_paths:
                print(f"- {p.resolve()}")
        else:
            print("\nNo plots were written (no runs found).")

    if JSON_OUT:
        report = _build_json_report(records_by_mode)
        out_path = Path(JSON_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {out_path.resolve()}")

    configured_csv_default = _default_csv_out(SURVEY_VERSION, MODEL_NAME)
    effective_csv_out = _default_csv_out(SURVEY_VERSION, resolved_model_name) if CSV_OUT == configured_csv_default else CSV_OUT
    if effective_csv_out:
        csv_base_path = Path(effective_csv_out)
        if len(modes) > 1:
            print("\nWrote CSV reports:")
            for mode in modes:
                mode_rows = _build_csv_rows({mode: records_by_mode[mode]})
                mode_csv_path = _csv_path_for_mode(csv_base_path, mode)
                _write_csv(mode_csv_path, mode_rows)
                print(f"- {mode}: {mode_csv_path.resolve()} ({len(mode_rows)} rows)")
        else:
            csv_rows = _build_csv_rows(records_by_mode)
            csv_path = _csv_path_for_mode(csv_base_path, modes[0]) if csv_base_path.name == "_compare_runs.csv" else csv_base_path
            _write_csv(csv_path, csv_rows)
            print(f"\nWrote CSV report to: {csv_path.resolve()} ({len(csv_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
