import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError


class TraceLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.steps: List[Dict[str, Any]] = []
        self._seq = 0
        self.trace_path = self.out_dir / "trace.json"

    def log(self, kind: str, payload: Dict[str, Any]):
        self._seq += 1
        self.steps.append(
            {
                "seq": self._seq,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "kind": kind,
                **payload,
            }
        )
        self._flush_trace()

    def _flush_trace(self):
        try:
            self.trace_path.write_text(json.dumps(self.steps, indent=2), encoding="utf-8")
        except Exception:
            # Best-effort debug persistence: never block agent execution.
            pass

    def save(self):
        self._flush_trace()

        counts = Counter(step.get("kind", "") for step in self.steps)
        exec_steps = [
            {
                "seq": step.get("seq"),
                "step_index": step.get("index"),
                "tool": step.get("tool"),
                "selector": step.get("selector"),
                "value": step.get("value"),
                "field_key": step.get("field_key"),
                "field_label": step.get("field_label"),
                "field_interaction": step.get("field_interaction"),
                "field_kind": step.get("field_kind"),
                "question_text": step.get("question_text"),
                "alt": step.get("alt"),
                "option_label": step.get("option_label"),
                "option_value": step.get("option_value"),
            }
            for step in self.steps
            if step.get("kind") == "exec_step"
        ]
        plan_step_counts = [
            {
                "seq": step.get("seq"),
                "plan_size": len(step.get("steps") or []),
            }
            for step in self.steps
            if step.get("kind") == "plan"
        ]

        summary = {
            "total_events": len(self.steps),
            "event_counts": dict(sorted(counts.items())),
            "plan_step_counts": plan_step_counts,
            "exec_steps_in_order": exec_steps,
        }
        (self.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        answer_eval = self._build_answer_eval(exec_steps=exec_steps)
        (self.out_dir / "answer_eval.json").write_text(json.dumps(answer_eval, indent=2), encoding="utf-8")

        metrics = self._build_metrics(exec_steps=exec_steps, counts=counts, plan_step_counts=plan_step_counts)
        (self.out_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _build_answer_eval(self, exec_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a per-run answer-vs-expected artifact.
        Expected values are sourced from strict UI/server ground truth:
        - ui_layout_trace.json (question correct_option_id / expected_option_id)
        - latest action_space meta captcha.code_text
        - explicit text attention instruction ("Please select Somewhat...")
        """
        latest_by_key: Dict[str, Dict[str, Any]] = {}
        for step in exec_steps:
            key = str(step.get("field_key") or "").strip()
            if not key:
                continue
            tool = str(step.get("tool") or "").strip()
            if tool not in {"fill", "select", "check", "set_range"}:
                continue
            latest_by_key[key] = step

        def _is_answered(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            return True

        ui_trace: Dict[str, Any] = {}
        ui_trace_path = self.out_dir / "ui_layout_trace.json"
        if ui_trace_path.exists():
            try:
                ui_trace = json.loads(ui_trace_path.read_text(encoding="utf-8"))
            except Exception:
                ui_trace = {}

        image_eval: List[Dict[str, Any]] = []
        for item in (ui_trace.get("questions") or []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("question_id") or "").strip()
            if not key:
                continue
            selected = latest_by_key.get(key) or {}
            selected_value = str(selected.get("value") or "")
            expected_value = str(item.get("correct_option_id") or "")
            is_correct: Optional[bool] = None
            if expected_value:
                is_correct = selected_value == expected_value
            image_eval.append(
                {
                    "key": key,
                    "question_text": None,
                    "selected_value": selected_value,
                    "selected_option_label": selected.get("option_label"),
                    "expected_value": expected_value or None,
                    "is_correct": is_correct,
                }
            )

        text_question_ids = [
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
        text_answered_count = sum(
            1 for qid in text_question_ids if _is_answered((latest_by_key.get(qid) or {}).get("value"))
        )
        text_known_count = len(text_question_ids)

        image_question_ids = [str(item.get("question_id") or "").strip() for item in (ui_trace.get("questions") or [])]
        image_question_ids = [qid for qid in image_question_ids if qid]
        image_answered_count = sum(
            1 for qid in image_question_ids if _is_answered((latest_by_key.get(qid) or {}).get("value"))
        )
        image_question_count = len(image_question_ids)

        image_attention_eval: Optional[Dict[str, Any]] = None
        attention_item = ui_trace.get("image_attention") if isinstance(ui_trace.get("image_attention"), dict) else None
        if attention_item:
            attention_key = str(attention_item.get("question_id") or "").strip()
            selected = latest_by_key.get(attention_key) or {}
            selected_value = str(selected.get("value") or "")
            expected_value = str(attention_item.get("expected_option_id") or "")
            image_attention_eval = {
                "key": attention_key,
                "selected_value": selected_value,
                "selected_option_label": selected.get("option_label"),
                "expected_value": expected_value or None,
                "is_correct": (selected_value == expected_value) if expected_value else None,
            }

        text_attention_eval: Optional[Dict[str, Any]] = None
        attention_candidates = [
            step
            for step in exec_steps
            if str(step.get("field_key") or "").strip() == "attention_text_mid"
            or "please select somewhat" in str(step.get("field_label") or "").strip().lower()
        ]
        if attention_candidates:
            chosen = attention_candidates[-1]
            selected_value = str(chosen.get("value") or "")
            text_attention_eval = {
                "key": str(chosen.get("field_key") or "attention_text_mid"),
                "selected_value": selected_value,
                "expected_value": "somewhat",
                "is_correct": selected_value.lower() == "somewhat",
            }

        latest_action_space: Dict[str, Any] = {}
        latest_action_space_with_captcha: Optional[Path] = None
        action_space_candidates = sorted(self.out_dir.glob("action_space_*.json"))
        for candidate in reversed(action_space_candidates):
            try:
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                parsed = None
            if not isinstance(parsed, dict):
                continue
            if not latest_action_space:
                latest_action_space = parsed
            code_text = str((((parsed.get("meta") or {}).get("captcha") or {}).get("code_text") or "")).strip()
            if code_text:
                latest_action_space = parsed
                latest_action_space_with_captcha = candidate
                break
        captcha_expected = str(
            (((latest_action_space.get("meta") or {}).get("captcha") or {}).get("code_text") or "")
        )
        captcha_input_selector = str(
            (((latest_action_space.get("meta") or {}).get("captcha") or {}).get("input_selector") or "")
        ).strip()
        captcha_field_key = ""
        for f in (latest_action_space.get("fields") or []):
            if not isinstance(f, dict):
                continue
            if str(f.get("kind") or "").strip() != "captcha_input":
                continue
            captcha_field_key = str(f.get("key") or "").strip()
            if not captcha_input_selector:
                captcha_input_selector = str(f.get("selector") or "").strip()
            break

        captcha_eval: Optional[Dict[str, Any]] = None
        captcha_candidates: List[Dict[str, Any]] = []
        for step in exec_steps:
            if str(step.get("tool") or "").strip() != "fill":
                continue
            step_kind = str(step.get("field_kind") or "").strip()
            step_selector = str(step.get("selector") or "").strip()
            step_field_key = str(step.get("field_key") or "").strip()
            if step_kind == "captcha_input":
                captcha_candidates.append(step)
                continue
            if captcha_input_selector and step_selector == captcha_input_selector:
                captcha_candidates.append(step)
                continue
            if captcha_field_key and step_field_key == captcha_field_key:
                captcha_candidates.append(step)
        if captcha_candidates:
            chosen = captcha_candidates[-1]
            selected_value = str(chosen.get("value") or "")
            captcha_eval = {
                "key": str(chosen.get("field_key") or ""),
                "selected_value": selected_value,
                "expected_value": captcha_expected or None,
                "is_correct": (selected_value == captcha_expected) if captcha_expected else None,
            }

        known_image = [x for x in image_eval if isinstance(x.get("is_correct"), bool)]
        image_correct_count = sum(1 for x in known_image if x.get("is_correct") is True)

        return {
            "image_items": image_eval,
            "image_known_count": len(known_image),
            "image_correct_count": image_correct_count,
            "image_accuracy_known": (image_correct_count / len(known_image)) if known_image else None,
            "text_page_completion": {
                "answered_count": text_answered_count,
                "total_count": text_known_count,
                "completion_rate": (text_answered_count / text_known_count) if text_known_count else None,
            },
            "image_page_completion": {
                "answered_count": image_answered_count,
                "total_count": image_question_count,
                "completion_rate": (image_answered_count / image_question_count) if image_question_count else None,
            },
            "image_attention": image_attention_eval,
            "text_attention": text_attention_eval,
            "captcha": captcha_eval,
            "ground_truth_source": {
                "ui_layout_trace_path": str(ui_trace_path) if ui_trace_path.exists() else None,
                "latest_action_space_path": str(action_space_candidates[-1]) if action_space_candidates else None,
                "captcha_action_space_path": str(latest_action_space_with_captcha) if latest_action_space_with_captcha else None,
            },
        }

    def _build_metrics(
        self,
        exec_steps: List[Dict[str, Any]],
        counts: Counter,
        plan_step_counts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _lower(v: Any) -> str:
            return str(v or "").strip().lower()

        def _looks_like_answer_step(step: Dict[str, Any]) -> bool:
            tool = step.get("tool")
            if tool in {"fill", "select", "check", "set_range"}:
                return True
            if tool == "click" and step.get("field_kind") == "image_option":
                return True
            return False

        def _field_identity(step: Dict[str, Any]) -> str:
            return (
                str(step.get("field_key") or "").strip()
                or str(step.get("question_text") or "").strip()
                or str(step.get("field_label") or "").strip()
                or str(step.get("selector") or "").strip()
            )

        def _extract_latest_url() -> str:
            for event in reversed(self.steps):
                for key in ("url", "new_url", "current_url"):
                    value = event.get(key)
                    if isinstance(value, str) and value:
                        return value
            return ""

        def _extract_route_prefix(target_url: Any) -> str:
            target_text = str(target_url or "").strip()
            if not target_text:
                return "/survey"
            path = urlparse(target_text).path or "/survey"
            path = path.rstrip("/")
            return path if path else "/survey"

        def _extract_path(url_or_path: Any) -> str:
            text = str(url_or_path or "").strip()
            if not text:
                return ""
            parsed = urlparse(text)
            if parsed.scheme or parsed.netloc:
                return parsed.path or ""
            return text

        answer_steps = [s for s in exec_steps if _looks_like_answer_step(s)]
        unique_answer_ids: List[str] = []
        seen_answer_ids = set()
        for step in answer_steps:
            key = _field_identity(step)
            if not key or key in seen_answer_ids:
                continue
            seen_answer_ids.add(key)
            unique_answer_ids.append(key)

        text_attention_steps = [
            s
            for s in exec_steps
            if "please select somewhat" in _lower(s.get("field_label"))
        ]
        image_attention_steps = [
            s
            for s in exec_steps
            if "select the soccer ball" in _lower(s.get("question_text"))
            or "select the soccer ball" in _lower(s.get("field_label"))
        ]
        run_context = next((e for e in self.steps if e.get("kind") == "run_context"), {})
        page_markers = [e for e in self.steps if e.get("kind") == "page_marker"]
        route_prefix = _extract_route_prefix(run_context.get("target"))
        thank_you_path = f"{route_prefix}/thank-you"
        done_path = f"{route_prefix}/done"
        reached_urls = {
            str(e.get("url"))
            for e in self.steps
            if isinstance(e.get("url"), str) and e.get("url")
        }
        reached_urls.update(
            str(e.get("new_url"))
            for e in self.steps
            if isinstance(e.get("new_url"), str) and e.get("new_url")
        )
        reached_urls = {u for u in reached_urls if u}
        reached_paths = {_extract_path(u) for u in reached_urls}
        reached_paths = {p for p in reached_paths if p}

        reached_thank_you = any(p == thank_you_path for p in reached_paths)
        reached_done = any(p == done_path for p in reached_paths)
        next_click_count = sum(
            1 for s in exec_steps if s.get("tool") == "click" and "next" in _lower(s.get("selector"))
        )
        submit_click_count = sum(
            1 for s in exec_steps if s.get("tool") == "click" and "submit" in _lower(s.get("selector"))
        )
        model_raw_events = [e for e in self.steps if e.get("kind") == "model_raw"]
        model_raw_by_source = {
            "llm": sum(1 for e in model_raw_events if e.get("source") == "llm"),
            "vlm": sum(1 for e in model_raw_events if e.get("source") == "vlm"),
        }
        submission_cleared_session = any(
            _extract_path(e.get("url")) == thank_you_path
            and e.get("has_text_answers_store") is False
            and e.get("has_image_answers_store") is False
            for e in page_markers
        )
        submitted = bool(submission_cleared_session or reached_done)

        return {
            "survey_version": run_context.get("survey_version"),
            "prompt_behavior_mode": run_context.get("prompt_behavior_mode"),
            "planner_mode": run_context.get("planner_mode"),
            "target": run_context.get("target"),
            "total_events": len(self.steps),
            "total_exec_steps": len(exec_steps),
            "answer_step_count": len(answer_steps),
            "unique_answer_items_touched": len(unique_answer_ids),
            "answer_order": unique_answer_ids,
            "plan_iterations": len(plan_step_counts),
            "plan_step_counts": [p.get("plan_size", 0) for p in plan_step_counts],
            "event_counts": dict(sorted(counts.items())),
            "plan_fallback_count": int(counts.get("plan_fallback", 0)),
            "validation_blocked_count": int(counts.get("validation_blocked", 0)),
            "model_raw_count": len(model_raw_events),
            "model_raw_by_source": model_raw_by_source,
            "next_click_count": next_click_count,
            "submit_click_count": submit_click_count,
            "text_attention_action_count": len(text_attention_steps),
            "image_attention_action_count": len(image_attention_steps),
            "reached_thank_you": reached_thank_you,
            "reached_done": reached_done,
            "submission_cleared_session": submission_cleared_session,
            "submitted": submitted,
            "final_url": _extract_latest_url(),
        }


async def tool_goto(page: Page, logger: TraceLogger, url: str):
    # domcontentloaded is often too early for Next.js/SPA hydration.
    await page.goto(url, wait_until="domcontentloaded")
    # Give the client a chance to render (cheap + helps a lot).
    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except PlaywrightTimeoutError:
        # Not fatal; some apps keep network busy. We'll rely on selector waits later.
        pass

    logger.log("goto", {"url": url, "current_url": page.url})


async def tool_wait_for_selector(
    page: Page,
    logger: TraceLogger,
    selector: str,
    timeout_ms: int = 20_000,
    state: str = "attached",
):
    """
    Wait until a selector appears in the DOM.
    state: "attached" | "visible" | "detached" | "hidden"
    """
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms, state=state)
        logger.log(
            "wait_for_selector",
            {"selector": selector, "timeout_ms": timeout_ms, "state": state},
        )
    except PlaywrightTimeoutError:
        logger.log(
            "wait_for_selector_timeout",
            {"selector": selector, "timeout_ms": timeout_ms, "state": state, "url": page.url},
        )
        raise


async def tool_wait_for_any(
    page: Page,
    logger: TraceLogger,
    selectors: List[str],
    timeout_ms: int = 20_000,
    poll_ms: int = 250,
):
    """
    Wait until ANY selector in the list exists.
    Useful when you don't know exact IDs yet.
    """
    deadline = page.context._loop.time() + (timeout_ms / 1000.0)

    while page.context._loop.time() < deadline:
        for sel in selectors:
            try:
                if await page.locator(sel).count() > 0:
                    logger.log("wait_for_any", {"hit": sel, "selectors": selectors, "timeout_ms": timeout_ms})
                    return sel
            except Exception:
                # ignore transient DOM errors during hydration
                pass
        await page.wait_for_timeout(poll_ms)

    logger.log("wait_for_any_timeout", {"selectors": selectors, "timeout_ms": timeout_ms, "url": page.url})
    raise TimeoutError(f"None of these selectors appeared within {timeout_ms}ms: {selectors}")


async def tool_wait_for_not_loading(
    page: Page,
    logger: TraceLogger,
    timeout_ms: int = 20_000,
):
    """
    Next.js apps often show a 'Loading...' shell. This waits until it disappears.
    If your page doesn't use 'Loading...', this is harmless (will return quickly).
    """
    js = "() => !document.body || !document.body.innerText || !document.body.innerText.includes('Loading...')"
    try:
        await page.wait_for_function(js, timeout=timeout_ms)
        logger.log("wait_for_not_loading", {"timeout_ms": timeout_ms})
    except PlaywrightTimeoutError:
        logger.log("wait_for_not_loading_timeout", {"timeout_ms": timeout_ms, "url": page.url})
        raise


async def tool_dom_snapshot(page: Page, logger: TraceLogger, path: Path):
    html = await page.content()
    path.write_text(html, encoding="utf-8")
    logger.log("dom_snapshot", {"path": str(path)})


async def tool_screenshot(page: Page, logger: TraceLogger, path: Path, full_page: bool = True):
    await page.screenshot(path=str(path), full_page=full_page)
    logger.log("screenshot", {"path": str(path), "full_page": full_page})


async def tool_fill(page: Page, logger: TraceLogger, selector: str, value: Any):
    """
    Fill text-like inputs and textarea fields.

    Notes:
    - Number inputs are normalized to numeric text when possible.
    - Dispatches input/change events to keep reactive forms in sync.
    - Uses layered fallbacks (fill -> type -> eval) for robustness.
    """
    text_value = str(value)
    locator = page.locator(selector).first

    try:
        input_type = (await locator.get_attribute("type") or "").lower()
    except Exception:
        input_type = ""

    if input_type == "number":
        try:
            text_value = str(int(float(text_value)))
        except Exception:
            digits_only = "".join(ch for ch in text_value if ch.isdigit())
            text_value = digits_only if digits_only else "25"
        await locator.click()
        await locator.fill("")
        await locator.type(text_value)
        await locator.evaluate(
            """(el) => {
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        logger.log("fill", {"selector": selector, "value": text_value, "element_type": "number"})
        return

    try:
        await page.fill(selector, text_value)
        logger.log("fill", {"selector": selector, "value": text_value, "element_type": input_type or "text"})
        return
    except Exception:
        pass

    try:
        await locator.click()
        await locator.press("Control+A")
        await locator.type(text_value)
        logger.log(
            "fill",
            {"selector": selector, "value": text_value, "via": "fallback_type", "element_type": input_type or "text"},
        )
        return
    except Exception:
        pass

    await locator.evaluate(
        """(el, v) => {
            el.value = String(v);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        text_value,
    )
    logger.log(
        "fill",
        {"selector": selector, "value": text_value, "via": "fallback_eval", "element_type": input_type or "text"},
    )


async def tool_wait_for_url_change(page: Page, logger: TraceLogger, old_url: str, timeout_ms: int = 8000) -> bool:
    try:
        await page.wait_for_function("old => location.href !== old", arg=old_url, timeout=timeout_ms)
        logger.log("url_changed", {"old_url": old_url, "new_url": page.url})
        return True
    except PlaywrightTimeoutError:
        logger.log("url_not_changed", {"old_url": old_url, "current_url": page.url})
        return False


async def tool_select(page: Page, logger: TraceLogger, selector: str, value: str):
    """
    Select an option in a native <select> element by option value.
    """
    await page.select_option(selector, value=value)
    logger.log("select", {"selector": selector, "value": value})


async def tool_click(page: Page, logger: TraceLogger, selector: str):
    """
    Click a clickable target (button, link, image option, or control).
    """
    await page.click(selector)
    logger.log("click", {"selector": selector})


async def tool_set_range(page: Page, logger: TraceLogger, selector: str, value: int):
    """
    Set a slider/range input to a numeric value.

    Notes:
    - Value is clamped to [min, max] and aligned to step.
    - Prefers keyboard nudging; falls back to direct value assignment + events.
    """
    locator = page.locator(selector).first

    def _to_float(raw: Optional[str], default: float) -> float:
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except Exception:
            return default

    min_v = _to_float(await locator.get_attribute("min"), 0.0)
    max_v = _to_float(await locator.get_attribute("max"), 100.0)
    step_v = _to_float(await locator.get_attribute("step"), 1.0)
    current_v = _to_float(await locator.get_attribute("value"), min_v)

    target = float(value)
    clamped = min(max(target, min_v), max_v)
    if step_v > 0:
        clamped = min_v + round((clamped - min_v) / step_v) * step_v
        clamped = min(max(clamped, min_v), max_v)

    try:
        delta_steps = int(round((clamped - current_v) / step_v)) if step_v > 0 else 0
        await locator.focus()
        if delta_steps > 0:
            for _ in range(delta_steps):
                await locator.press("ArrowRight")
        elif delta_steps < 0:
            for _ in range(abs(delta_steps)):
                await locator.press("ArrowLeft")

        await locator.evaluate(
            """(el) => {
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }"""
        )
        actual_value = await locator.evaluate("el => el.value")
        logger.log(
            "set_range",
            {
                "selector": selector,
                "requested": value,
                "clamped_value": clamped,
                "actual_value": actual_value,
                "min": min_v,
                "max": max_v,
                "step": step_v,
                "via": "keyboard",
            },
        )
        return
    except Exception:
        pass

    await locator.evaluate(
        """(el, v) => {
            el.value = String(v);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        clamped,
    )
    actual_value = await locator.evaluate("el => el.value")
    logger.log(
        "set_range",
        {
            "selector": selector,
            "requested": value,
            "clamped_value": clamped,
            "actual_value": actual_value,
            "min": min_v,
            "max": max_v,
            "step": step_v,
            "via": "eval_fallback",
        },
    )


async def tool_check(page: Page, logger: TraceLogger, selector: str):
    """
    Check/select a radio or checkbox option.

    Notes:
    - Tries native Playwright check first.
    - Falls back to first visible checkable candidate or click.
    - Supports degraded group-selector recovery if option selector misses.
    """
    try:
        await page.check(selector)
        logger.log("check", {"selector": selector})
        return
    except Exception:
        pass

    locator = page.locator(selector)
    count = await locator.count()
    if count == 0:
        # If a strict option selector misses (e.g., missing value attr), degrade to group selector.
        name_match = re.search(r'name="([^"]+)"', selector)
        type_match = re.search(r'type="([^"]+)"', selector)
        if name_match:
            group_type = type_match.group(1) if type_match else "radio"
            group_selector = f'input[type="{group_type}"][name="{name_match.group(1)}"]'
            locator = page.locator(group_selector)
            count = await locator.count()
            selector = group_selector

    for i in range(count):
        candidate = locator.nth(i)
        try:
            if not await candidate.is_visible():
                continue
            input_type = (await candidate.get_attribute("type") or "").lower()
            tag_name = await candidate.evaluate("el => el.tagName.toLowerCase()")
            if tag_name == "input" and input_type in {"radio", "checkbox"}:
                await candidate.check()
                logger.log("check", {"selector": selector, "via": "fallback_first"})
                return
        except Exception:
            continue

    for i in range(count):
        candidate = locator.nth(i)
        try:
            if await candidate.is_visible():
                await candidate.click()
                logger.log("check", {"selector": selector, "via": "fallback_first"})
                return
        except Exception:
            continue

    raise RuntimeError(f"tool_check could not act on selector: {selector}")
