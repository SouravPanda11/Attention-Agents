import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError


class TraceLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.steps: List[Dict[str, Any]] = []

    def log(self, kind: str, payload: Dict[str, Any]):
        self.steps.append({"kind": kind, **payload})

    def save(self):
        (self.out_dir / "trace.json").write_text(
            json.dumps(self.steps, indent=2), encoding="utf-8"
        )


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
    await page.select_option(selector, value=value)
    logger.log("select", {"selector": selector, "value": value})


async def tool_click(page: Page, logger: TraceLogger, selector: str):
    await page.click(selector)
    logger.log("click", {"selector": selector})


async def tool_set_range(page: Page, logger: TraceLogger, selector: str, value: int):
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
