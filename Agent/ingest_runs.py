import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _to_clean_str(value: Any) -> str:
    return str(value or "").strip()


def _sanitize_answer_dict(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, raw in value.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        if raw is None:
            continue
        if isinstance(raw, str):
            cleaned = raw.strip()
            if not cleaned:
                continue
            out[key_str] = cleaned
            continue
        out[key_str] = raw
    return out


def _normalize_payload(value: Dict[str, Any]) -> Dict[str, Any]:
    text_answers = _sanitize_answer_dict(value.get("text_answers"))
    image_answers = _sanitize_answer_dict(value.get("image_answers"))
    text_attention_value = _to_clean_str(value.get("text_attention_value"))
    captcha_input = _to_clean_str(value.get("captcha_input"))
    image_attention_choice = _to_clean_str(value.get("image_attention_choice"))

    has_any_values = bool(
        text_answers
        or image_answers
        or text_attention_value
        or captcha_input
        or image_attention_choice
    )
    return {
        "text_answers": text_answers,
        "text_attention_value": text_attention_value,
        "image_answers": image_answers,
        "captcha_input": captcha_input,
        "image_attention_choice": image_attention_choice,
        "text_answer_count": len(text_answers),
        "image_answer_count": len(image_answers),
        "has_any_values": has_any_values,
    }


def _payload_from_exec_steps(summary: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "text_answers": {},
        "text_attention_value": "",
        "image_answers": {},
        "captcha_input": "",
        "image_attention_choice": "",
    }
    exec_steps = summary.get("exec_steps_in_order")
    if not isinstance(exec_steps, list):
        return _normalize_payload(payload)

    for step in exec_steps:
        if not isinstance(step, dict):
            continue
        key = _to_clean_str(step.get("field_key"))
        tool = _to_clean_str(step.get("tool")).lower()
        selector = _to_clean_str(step.get("selector")).lower()
        value = step.get("value")
        value_str = _to_clean_str(value)

        if key in TEXT_QUESTION_IDS and value is not None:
            payload["text_answers"][key] = value
        elif key == "attention_text_mid":
            payload["text_attention_value"] = value_str
        elif key in IMAGE_QUESTION_IDS and value_str:
            payload["image_answers"][key] = value_str
        elif key == "attention_image_mid":
            payload["image_attention_choice"] = value_str
        elif (
            tool == "fill"
            and value_str
            and not payload["captcha_input"]
            and ("captcha" in key.lower() or "captcha" in selector)
        ):
            payload["captcha_input"] = value_str

    return _normalize_payload(payload)


def _read_submission_payload(run_dir: Path, summary: Dict[str, Any]) -> Tuple[Dict[str, Any], str, bool, str]:
    snapshot_path = run_dir / "submission_snapshot.json"
    if snapshot_path.exists():
        snapshot = _load_json(snapshot_path)
        best_capture = snapshot.get("best_capture")
        if isinstance(best_capture, dict):
            normalized = best_capture.get("normalized_payload")
            if isinstance(normalized, dict):
                return (
                    _normalize_payload(normalized),
                    "submission_snapshot",
                    True,
                    str(snapshot_path),
                )

    fallback = _payload_from_exec_steps(summary)
    return (fallback, "run_summary_fallback", False, str(snapshot_path))


def _iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    for run_dir in sorted(runs_root.glob("*/*/*/run_*")):
        if run_dir.is_dir():
            yield run_dir


def _parse_run_identity(runs_root: Path, run_dir: Path) -> Tuple[str, str, str, str]:
    rel = run_dir.relative_to(runs_root)
    parts = rel.parts
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2], parts[3]
    return ("unknown_survey", "unknown_model", "unknown_mode", run_dir.name)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_index (
          survey_version TEXT NOT NULL,
          model_name TEXT NOT NULL,
          mode TEXT NOT NULL,
          run_name TEXT NOT NULL,
          run_dir TEXT NOT NULL,
          session_id TEXT NOT NULL,
          machine_name TEXT NOT NULL,
          has_submission_snapshot INTEGER NOT NULL,
          snapshot_path TEXT NOT NULL,
          snapshot_has_any_values INTEGER NOT NULL,
          payload_source TEXT NOT NULL,
          imported_at_utc TEXT NOT NULL,
          PRIMARY KEY (survey_version, model_name, mode, run_name)
        );

        CREATE TABLE IF NOT EXISTS run_submissions (
          survey_version TEXT NOT NULL,
          model_name TEXT NOT NULL,
          mode TEXT NOT NULL,
          run_name TEXT NOT NULL,
          run_dir TEXT NOT NULL,
          session_id TEXT NOT NULL,
          text_answers_json TEXT NOT NULL,
          text_attention_value TEXT NOT NULL,
          image_answers_json TEXT NOT NULL,
          captcha_input TEXT NOT NULL,
          image_attention_choice TEXT NOT NULL,
          text_answer_count INTEGER NOT NULL,
          image_answer_count INTEGER NOT NULL,
          has_any_values INTEGER NOT NULL,
          payload_source TEXT NOT NULL,
          imported_at_utc TEXT NOT NULL,
          PRIMARY KEY (survey_version, model_name, mode, run_name)
        );

        CREATE INDEX IF NOT EXISTS idx_run_index_model_mode ON run_index(model_name, mode);
        CREATE INDEX IF NOT EXISTS idx_run_submissions_model_mode ON run_submissions(model_name, mode);
        """
    )


def _upsert_run(
    conn: sqlite3.Connection,
    survey_version: str,
    model_name: str,
    mode: str,
    run_name: str,
    run_dir: str,
    session_id: str,
    machine_name: str,
    has_submission_snapshot: bool,
    snapshot_path: str,
    payload_source: str,
    payload: Dict[str, Any],
    imported_at_utc: str,
) -> None:
    conn.execute(
        """
        INSERT INTO run_index (
          survey_version, model_name, mode, run_name, run_dir, session_id, machine_name,
          has_submission_snapshot, snapshot_path, snapshot_has_any_values, payload_source, imported_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(survey_version, model_name, mode, run_name) DO UPDATE SET
          run_dir=excluded.run_dir,
          session_id=excluded.session_id,
          machine_name=excluded.machine_name,
          has_submission_snapshot=excluded.has_submission_snapshot,
          snapshot_path=excluded.snapshot_path,
          snapshot_has_any_values=excluded.snapshot_has_any_values,
          payload_source=excluded.payload_source,
          imported_at_utc=excluded.imported_at_utc
        """,
        (
            survey_version,
            model_name,
            mode,
            run_name,
            run_dir,
            session_id,
            machine_name,
            int(has_submission_snapshot),
            snapshot_path,
            int(bool(payload.get("has_any_values"))),
            payload_source,
            imported_at_utc,
        ),
    )
    conn.execute(
        """
        INSERT INTO run_submissions (
          survey_version, model_name, mode, run_name, run_dir, session_id,
          text_answers_json, text_attention_value, image_answers_json,
          captcha_input, image_attention_choice,
          text_answer_count, image_answer_count, has_any_values,
          payload_source, imported_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(survey_version, model_name, mode, run_name) DO UPDATE SET
          run_dir=excluded.run_dir,
          session_id=excluded.session_id,
          text_answers_json=excluded.text_answers_json,
          text_attention_value=excluded.text_attention_value,
          image_answers_json=excluded.image_answers_json,
          captcha_input=excluded.captcha_input,
          image_attention_choice=excluded.image_attention_choice,
          text_answer_count=excluded.text_answer_count,
          image_answer_count=excluded.image_answer_count,
          has_any_values=excluded.has_any_values,
          payload_source=excluded.payload_source,
          imported_at_utc=excluded.imported_at_utc
        """,
        (
            survey_version,
            model_name,
            mode,
            run_name,
            run_dir,
            session_id,
            json.dumps(payload.get("text_answers") or {}, ensure_ascii=True, sort_keys=True),
            _to_clean_str(payload.get("text_attention_value")),
            json.dumps(payload.get("image_answers") or {}, ensure_ascii=True, sort_keys=True),
            _to_clean_str(payload.get("captcha_input")),
            _to_clean_str(payload.get("image_attention_choice")),
            int(payload.get("text_answer_count") or 0),
            int(payload.get("image_answer_count") or 0),
            int(bool(payload.get("has_any_values"))),
            payload_source,
            imported_at_utc,
        ),
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Ingest per-run JSON artifacts into an aggregated SQLite database."
    )
    parser.add_argument(
        "--runs-root",
        default=str(script_dir / "runs"),
        help="Root directory containing survey/model/mode/run_* folders.",
    )
    parser.add_argument(
        "--db-path",
        default=str(script_dir / "runs_aggregate.sqlite"),
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Skip runs that do not contain submission_snapshot.json payloads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and summarize runs without writing to the database.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    db_path = Path(args.db_path).resolve()

    if not runs_root.exists():
        print(f"Runs root not found: {runs_root}")
        return 1

    discovered = 0
    ingested = 0
    skipped = 0
    with_payload = 0

    conn: sqlite3.Connection | None = None
    if not args.dry_run:
        conn = sqlite3.connect(db_path)
        _init_db(conn)

    try:
        for run_dir in _iter_run_dirs(runs_root):
            discovered += 1
            survey_version, model_name, mode, run_name = _parse_run_identity(runs_root, run_dir)
            summary = _load_json(run_dir / "run_summary.json")
            session_id = _to_clean_str(summary.get("session_id"))
            machine_name = _to_clean_str(summary.get("machine_name"))
            payload, payload_source, has_snapshot, snapshot_path = _read_submission_payload(run_dir, summary)

            if args.snapshot_only and payload_source != "submission_snapshot":
                skipped += 1
                continue

            if payload.get("has_any_values"):
                with_payload += 1

            imported_at_utc = _utc_now_iso()
            if conn is not None:
                _upsert_run(
                    conn=conn,
                    survey_version=survey_version,
                    model_name=model_name,
                    mode=mode,
                    run_name=run_name,
                    run_dir=str(run_dir),
                    session_id=session_id,
                    machine_name=machine_name,
                    has_submission_snapshot=has_snapshot,
                    snapshot_path=snapshot_path,
                    payload_source=payload_source,
                    payload=payload,
                    imported_at_utc=imported_at_utc,
                )
            ingested += 1
    finally:
        if conn is not None:
            conn.commit()
            conn.close()

    print(f"runs_root={runs_root}")
    print(f"db_path={db_path}")
    print(f"discovered={discovered}")
    print(f"ingested={ingested}")
    print(f"skipped={skipped}")
    print(f"with_payload={with_payload}")
    print(f"dry_run={bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
