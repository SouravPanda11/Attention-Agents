"""Validated action intents executed through the benchmark's visible controls."""
import json
import math
from pathlib import Path

OBSERVE_JS = Path(__file__).with_name("observe.js").read_text(encoding="utf-8")


def validate_plan(plan, observation):
    fields = {field["key"]: field for field in observation["fields"]}
    for index, action in enumerate(plan):
        if not isinstance(action, dict):
            raise ValueError(f"Action {index + 1} must be an object")
        tool = action.get("tool")
        if tool == "done":
            if len(plan) != 1 or set(action) != {"tool"}:
                raise ValueError("done must be the only action")
            continue
        key = action.get("key")
        if not isinstance(key, str) or key not in fields:
            raise ValueError(f"Action {index + 1}: unknown field key {key!r}")
        field = fields[key]
        allowed = {field["tool"]}
        if field.get("kind") in {"single-checkbox", "multiple-checkbox"}:
            allowed.add("uncheck")
        if tool not in allowed:
            raise ValueError(f"{key}: expected tool {sorted(allowed)}")
        if tool == "click":
            if key not in {"start-survey", "submit-survey"}:
                raise ValueError("Only o1 start and submit navigation is supported")
            if index != len(plan) - 1:
                raise ValueError("Navigation must be the final action")
            continue
        value = action.get("value")
        if tool in {"check", "uncheck"}:
            options = field["options"]
            if not isinstance(value, str):
                raise ValueError(f"{key}: expected an option label or value")
            if value not in [o["value"] for o in options]:
                matches = [o for o in options if o.get("label") == value]
                if len(matches) != 1:
                    raise ValueError(f"{key}: use an exact option value or an unambiguous exact label")
                # Normalize labels once, before the plan is logged and executed.
                action["value"] = matches[0]["value"]
        elif tool == "select":
            if value not in [o["value"] for o in field["options"]]:
                raise ValueError(f"{key}: value must be an exact option value")
        elif tool == "rank":
            options = [o["value"] for o in field["options"]]
            if (not isinstance(value, list) or not all(isinstance(x, str) for x in value)
                    or sorted(value) != sorted(options)):
                raise ValueError(f"{key}: rank must contain every item exactly once")
        elif tool in {"fill", "set_range"}:
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"{key}: expected text or a number")
            if tool == "set_range" or field.get("kind") == "numeric":
                try:
                    numeric = float(value)
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"{key}: expected a finite number") from exc
                if not math.isfinite(numeric):
                    raise ValueError(f"{key}: expected a finite number")
                if tool == "set_range":
                    low, high = float(field["min"]), float(field["max"])
                    step = float(field.get("step", 1))
                    if not low <= numeric <= high or not math.isclose(
                            (numeric - low) / step, round((numeric - low) / step), abs_tol=1e-7):
                        raise ValueError(f"{key}: value must fit slider range and step")


def card_for(page, key):
    return page.locator(f'section.question-card[data-question-id={json.dumps(key)}]')


async def observe(page):
    await page.locator('[data-action="start-survey"], section.question-card, .completion-card').first.wait_for()
    return await page.evaluate(OBSERVE_JS)


async def execute(page, action):
    tool, key = action["tool"], action.get("key")
    if tool == "done":
        return
    if tool == "click":
        button = page.locator(f'button[data-action={json.dumps(key)}]')
        if key == "submit-survey":
            async with page.expect_response(lambda r: r.url.split("?")[0].endswith("/api/submissions")
                                             and r.request.method == "POST", timeout=0) as pending:
                await button.click()
            response = await pending.value
            body = await response.json()
            snapshot = {"request": response.request.post_data_json,
                        "status": response.status, "response": body}
            return snapshot
        await button.click()
        await page.locator('section.question-card').first.wait_for()
        return
    card = card_for(page, key)
    value = action["value"]
    if tool == "fill":
        await card.locator('input, textarea').fill(str(value))
    elif tool == "select":
        await card.locator('select').select_option(value)
    elif tool in {"check", "uncheck"}:
        control = card.locator(f'input[value={json.dumps(value)}]')
        await control.set_checked(tool == "check")
    elif tool == "set_range":
        control = card.locator('input[type="range"]')
        # Native keyboard events update React, including an explicit midpoint choice.
        await control.focus()
        await control.press("Home")
        await control.press("ArrowRight")
        await control.press("Home")
        low = float(await control.get_attribute("min"))
        step = float(await control.get_attribute("step") or 1)
        count = round((float(value) - low) / step)
        if count > 10000:
            raise ValueError("Slider action exceeds 10000 native key presses")
        for _ in range(count):
            await control.press("ArrowRight")
        actual = float(await control.input_value())
        if not math.isclose(actual, float(value), abs_tol=1e-7):
            raise RuntimeError(f"Slider selected {actual}, expected {value}")
    elif tool == "rank":
        current = await card.locator('[data-ranking-item]').evaluate_all(
            "els => els.map(el => el.dataset.rankingItem)")
        if current == value:
            # The renderer does not store the untouched default order. Move and restore it.
            await card.locator('[data-action="rank-down"]').first.click()
        for target, item in enumerate(value):
            row = card.locator(f'[data-ranking-item={json.dumps(item)}]')
            position = int(await row.get_attribute("data-ranking-position")) - 1
            for _ in range(position - target):
                await row.locator('[data-action="rank-up"]').click()
        actual = await card.locator('[data-ranking-item]').evaluate_all(
            "els => els.map(el => el.dataset.rankingItem)")
        if actual != value:
            raise RuntimeError("Ranking UI did not match the requested order")
