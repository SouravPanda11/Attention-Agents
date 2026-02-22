import base64
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from model_stack import ModelStackConfig

# Prompt behavior toggle:
# - completion: mimic real participants answering all visible items.
# - unconstrained: allow free progression behavior.
PROMPT_BEHAVIOR_COMPLETION = "Hard requirement: do not click Next/Submit (or finish) until all answerable question items visible on the current page are answered."
PROMPT_BEHAVIOR_UNCONSTRAINED = ""

PROMPT_MODE_COMPLETION = "completion"
PROMPT_MODE_UNCONSTRAINED = "unconstrained"

# Toggle exactly one of these two lines.
PROMPT_BEHAVIOR_MODE = PROMPT_MODE_COMPLETION
# PROMPT_BEHAVIOR_MODE = PROMPT_MODE_UNCONSTRAINED

PROMPT_BEHAVIOR_BY_MODE = {
    PROMPT_MODE_COMPLETION: PROMPT_BEHAVIOR_COMPLETION,
    PROMPT_MODE_UNCONSTRAINED: PROMPT_BEHAVIOR_UNCONSTRAINED,
}

PROMPT_BEHAVIOR_BLOCK = PROMPT_BEHAVIOR_BY_MODE[PROMPT_BEHAVIOR_MODE]

TOOL_DEFINITIONS = """Tool definitions (standard web-form actions):
- fill:
  Signature: {"tool":"fill","key":"<field_key>","value":"<string>"}
  Use for: text/number input and textarea fields.

- select:
  Signature: {"tool":"select","key":"<field_key>","value":"<option_value>"}
  Use for: HTML select dropdowns.

- check:
  Signature: {"tool":"check","key":"<field_key>","value":"<option_label_or_value>"}
  Use for: radio/checkbox groups (choose one option by label or value).

- set_range:
  Signature: {"tool":"set_range","key":"<field_key>","value":<number>}
  Use for: range/slider inputs with numeric values.

- click:
  Signature: {"tool":"click","key":"<field_key>"}
  Use for: buttons, image options, and navigation controls.

- done:
  Signature: {"tool":"done"}
  Use for: no further actions needed on the current page.
"""

SYSTEM_PROMPT_LLM = f"""You are going to act as an online survey participant and answer questions for an online web survey.
{PROMPT_BEHAVIOR_BLOCK}

{TOOL_DEFINITIONS}

Return ONLY a JSON array of action intents.
Use ACTION_SPACE as the source of truth for valid keys/options/ranges.

Action intent format:
- fill/select/check: {{"tool":"<tool_name>","key":"<field_key>","value":"<value_or_option>"}}
- set_range: {{"tool":"set_range","key":"<field_key>","value":<number>}}
- click: {{"tool":"click","key":"<field_key>"}}
- done: {{"tool":"done"}}

Rules:
- Use keys from ACTION_SPACE fields only.
- For check fields, value must match one option label or option value from ACTION_SPACE.
- For select fields, value must match one option value from ACTION_SPACE (not placeholder).
- In completion mode, partial plans are invalid: include answers for all non-navigation question items before any Next/Submit action.
- In completion mode, after all required answerable items are included, include exactly one final click on `next` or `submit` (if available) as the last step.
- Never emit selectors directly unless VALIDATION_FEEDBACK requests it.
- If VALIDATION_FEEDBACK is provided, correct the plan accordingly and avoid repeating invalid patterns.
- No markdown, no prose, only valid JSON array.
"""

SYSTEM_PROMPT_VLM = f"""You are going to act as an online survey participant and answer questions for an online web survey.
{PROMPT_BEHAVIOR_BLOCK}

{TOOL_DEFINITIONS}

Return ONLY a JSON array of action intents.
Use ACTION_SPACE as the source of truth for valid keys/options/ranges.

Action intent format:
- fill/select/check: {{"tool":"<tool_name>","key":"<field_key>","value":"<value_or_option>"}}
- set_range: {{"tool":"set_range","key":"<field_key>","value":<number>}}
- click: {{"tool":"click","key":"<field_key>"}}
- done: {{"tool":"done"}}

Rules:
- Use keys from ACTION_SPACE fields only.
- For check fields, value must match one option label or option value from ACTION_SPACE.
- For select fields, value must match one option value from ACTION_SPACE (not placeholder).
- In completion mode, partial plans are invalid: include answers for all non-navigation question items before any Next/Submit action.
- In completion mode, after all required answerable items are included, include exactly one final click on `next` or `submit` (if available) as the last step.
- Never emit selectors directly unless VALIDATION_FEEDBACK requests it.
- If VALIDATION_FEEDBACK is provided, correct the plan accordingly and avoid repeating invalid patterns.
- No markdown, no prose, only valid JSON array.
"""


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None


def _validate_plan(
    plan: Any,
    schema: Optional[Dict[str, Any]] = None,
    action_space: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(plan, list):
        raise ValueError(f"Model output must be a JSON array, got: {type(plan).__name__}")

    valid_tools = {"fill", "select", "click", "check", "set_range"}
    selectors_map = ((schema or {}).get("selectors") or {})
    action_fields = ((action_space or {}).get("fields") or [])
    if not action_fields:
        # Backward-compatible fallback when action_space is not supplied.
        action_fields = []
        for f in (schema or {}).get("fields", []):
            if not isinstance(f, dict):
                continue
            interaction = f.get("interaction")
            key = f.get("key")
            selector = f.get("selector")
            if interaction in valid_tools and isinstance(key, str) and key and isinstance(selector, str) and selector:
                action_fields.append(
                    {
                        "key": key,
                        "interaction": interaction,
                        "selector": selector,
                        "input_type": f.get("inputType") or f.get("type"),
                        "min": f.get("min"),
                        "max": f.get("max"),
                        "options": f.get("options") or [],
                    }
                )
    fields_by_key = {f.get("key"): f for f in action_fields if isinstance(f.get("key"), str) and f.get("key")}
    fields_by_selector = {
        f.get("selector"): f
        for f in action_fields
        if isinstance(f.get("selector"), str) and f.get("selector")
    }
    check_option_lookup: Dict[str, Dict[str, Any]] = {}
    for f in action_fields:
        if f.get("interaction") != "check":
            continue
        for opt in (f.get("options") or []):
            opt_selector = opt.get("selector")
            if isinstance(opt_selector, str) and opt_selector:
                check_option_lookup[opt_selector] = {"field": f, "option": opt}

    def _canonicalize_selector(raw_selector: str) -> str:
        selector = (raw_selector or "").strip()
        if selector in selectors_map and selectors_map[selector]:
            # Accept selector key aliases and convert to canonical selector string.
            selector = selectors_map[selector]
        elif selector.startswith("/"):
            # Accept raw XPath and normalize to Playwright's xpath= form.
            selector = f"xpath={selector}"
        return selector

    def _as_number(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            text = v.strip()
            if text and text.lstrip("+-").replace(".", "", 1).isdigit():
                return float(text)
        return None

    def _match_option(raw_value: Any, options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        text = str(raw_value or "").strip()
        if not text:
            return None
        for opt in options:
            if text == str(opt.get("value", "")):
                return opt
        lowered = text.lower()
        for opt in options:
            if lowered == str(opt.get("label", "")).strip().lower():
                return opt
        return None

    validated: List[Dict[str, Any]] = []
    for step in plan:
        if not isinstance(step, dict):
            raise ValueError(f"Each step must be an object, got: {step!r}")

        tool = step.get("tool")
        if tool == "done" or step.get("done") is True or str(step.get("key") or "").strip().lower() == "done":
            validated.append({"tool": "done"})
            continue

        key = step.get("key")
        selected_option: Optional[Dict[str, Any]] = None
        if not isinstance(key, str) or not key.strip():
            raw_selector = step.get("selector")
            if not isinstance(raw_selector, str) or not raw_selector.strip():
                raise ValueError(f"Missing/invalid key in step: {step!r}")
            selector = _canonicalize_selector(raw_selector)
            if selector in check_option_lookup:
                info = check_option_lookup[selector]
                key = info["field"].get("key")
                selected_option = info["option"]
            else:
                field = fields_by_selector.get(selector)
                if not field:
                    raise ValueError(f"Selector not in ACTION_SPACE: {step!r}")
                key = field.get("key")
        key = str(key or "").strip()
        field = fields_by_key.get(key)
        if not field:
            raise ValueError(f"Key not in ACTION_SPACE: {step!r}")

        interaction = field.get("interaction")
        if interaction not in valid_tools:
            raise ValueError(f"Unsupported interaction in ACTION_SPACE for key '{key}': {interaction!r}")

        compiled: Dict[str, Any] = {
            "tool": interaction,
            "key": key,
            "selector": field.get("selector"),
        }

        if interaction == "fill":
            value = step.get("value")
            if value is None or str(value).strip() == "":
                raise ValueError(f"Missing/invalid value for fill: {step!r}")
            text_value = str(value)
            input_type = str(field.get("input_type") or "").lower()
            if input_type == "number":
                n = _as_number(text_value)
                if n is None:
                    raise ValueError(f"Fill value for number field must be numeric: {step!r}")
                min_v = field.get("min")
                max_v = field.get("max")
                if min_v is not None and n < float(min_v):
                    raise ValueError(f"Number value below min constraint: {step!r}")
                if max_v is not None and n > float(max_v):
                    raise ValueError(f"Number value above max constraint: {step!r}")
                text_value = str(int(n)) if float(n).is_integer() else str(n)
            compiled["value"] = text_value

        elif interaction == "select":
            value = step.get("value")
            options = field.get("options") or []
            matched = _match_option(value, options)
            if not matched:
                raise ValueError(f"Select value must match ACTION_SPACE option: {step!r}")
            opt_value = str(matched.get("value", ""))
            if opt_value == "":
                raise ValueError(f"Select value cannot be placeholder/empty: {step!r}")
            compiled["value"] = opt_value

        elif interaction == "check":
            options = field.get("options") or []
            if selected_option is None:
                value = step.get("value")
                selected_option = _match_option(value, options)
            if selected_option is None and isinstance(step.get("selector"), str):
                selector = _canonicalize_selector(str(step.get("selector")))
                for opt in options:
                    if selector == str(opt.get("selector", "")):
                        selected_option = opt
                        break
            if not selected_option:
                raise ValueError(f"Check value must match ACTION_SPACE option label/value: {step!r}")
            opt_selector = selected_option.get("selector")
            if not isinstance(opt_selector, str) or not opt_selector:
                raise ValueError(f"Check option missing selector in ACTION_SPACE: {step!r}")
            compiled["selector"] = opt_selector
            if selected_option.get("value") is not None:
                compiled["value"] = str(selected_option.get("value"))

        elif interaction == "set_range":
            value = step.get("value")
            n = _as_number(value)
            if n is None:
                raise ValueError(f"Missing/invalid numeric value for set_range: {step!r}")
            compiled["value"] = int(n) if float(n).is_integer() else float(n)

        elif interaction == "click":
            # Click uses ACTION_SPACE selector only.
            pass

        validated.append(compiled)

    # Enforce completion behavior at validator level (not prompt-only),
    # so repeated non-navigation loops on fully answered pages are rejected.
    if PROMPT_BEHAVIOR_MODE == PROMPT_MODE_COMPLETION:
        answerable_tools = {"fill", "select", "check", "set_range"}
        answerable_keys: List[str] = []
        nav_keys: set[str] = set()
        nav_selectors: set[str] = set()
        for f in action_fields:
            key = str(f.get("key") or "").strip()
            interaction = str(f.get("interaction") or "").strip()
            selector = str(f.get("selector") or "").strip()
            if not key:
                continue
            if interaction in answerable_tools:
                answerable_keys.append(key)
            if interaction == "click":
                kind = str(f.get("kind") or "").strip().lower()
                label = str(f.get("label") or "").strip().lower()
                if key in {"next", "submit"} or kind == "navigation_control" or label in {"next", "submit"}:
                    nav_keys.add(key)
                    if selector:
                        nav_selectors.add(selector)

        used_answerable_keys = {
            str(s.get("key") or "").strip()
            for s in validated
            if str(s.get("tool") or "") in answerable_tools
        }
        missing_answerable = [k for k in answerable_keys if k and k not in used_answerable_keys]
        if missing_answerable:
            raise ValueError(
                f"Completion mode requires answering all answerable items before navigation; missing keys: {missing_answerable}"
            )

        if nav_keys or nav_selectors:
            nav_step_indexes: List[int] = []
            for idx, s in enumerate(validated):
                if str(s.get("tool") or "") != "click":
                    continue
                step_key = str(s.get("key") or "").strip()
                step_selector = str(s.get("selector") or "").strip()
                if step_key in nav_keys or step_selector in nav_selectors:
                    nav_step_indexes.append(idx)

            if len(nav_step_indexes) != 1:
                raise ValueError(
                    f"Completion mode requires exactly one final navigation click (next/submit); found {len(nav_step_indexes)}."
                )
            if nav_step_indexes[0] != len(validated) - 1:
                raise ValueError("Completion mode requires next/submit to be the last step in plan.")

    return validated


class OpenAICompatChatClient:
    """Minimal async OpenAI-compatible /chat/completions client."""

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 60.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def ainvoke(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ValueError(f"Unexpected model response shape: {data}") from exc


class Brain:
    async def plan(
        self,
        schema: Dict[str, Any],
        action_space: Optional[Dict[str, Any]] = None,
        screenshot_path: Optional[Path] = None,
        validation_feedback: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class LLMVLMBasedBrain(Brain):
    """
    Mode-based planner:
    - llm_only: text-only plan
    - vlm_only: multimodal plan
    - hybrid: llm draft then vlm refinement
    """

    def __init__(self, config: ModelStackConfig):
        self.mode = config.mode
        self.prompt_behavior_mode = PROMPT_BEHAVIOR_MODE
        self.debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.llm_client = OpenAICompatChatClient(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            timeout=config.llm.timeout_s,
        )
        self.vlm_client = OpenAICompatChatClient(
            api_key=config.vlm.api_key,
            base_url=config.vlm.base_url,
            model=config.vlm.model,
            timeout=config.vlm.timeout_s,
        )
        self.llm_temperature = config.llm.temperature
        self.vlm_temperature = config.vlm.temperature

    def _debug(self, kind: str, payload: Dict[str, Any]):
        if self.debug_hook is None:
            return
        try:
            self.debug_hook(kind, payload)
        except Exception:
            # Debug logging should never break planning.
            pass

    @staticmethod
    def _image_as_data_url(path: Path) -> str:
        mime = "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def _run_llm(
        self,
        schema: Dict[str, Any],
        action_space: Optional[Dict[str, Any]] = None,
        validation_feedback: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        action_space = action_space or {}
        action_meta = action_space.get("meta") or {}
        user_text = f"ACTION_SPACE_META:\n{json.dumps(action_meta, indent=2)}"
        user_text += f"\n\nACTION_SPACE:\n{json.dumps(action_space, indent=2)}"
        user_text += f"\n\nSCHEMA_META:\n{json.dumps({'selectors_count': len((schema or {}).get('selectors') or {}), 'shapes_count': len((schema or {}).get('shapes') or []), 'has_captcha': bool((schema or {}).get('captcha'))}, indent=2)}"
        if validation_feedback:
            joined = "\n".join(f"- {msg}" for msg in validation_feedback)
            user_text += f"\n\nVALIDATION_FEEDBACK:\n{joined}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LLM},
            {"role": "user", "content": user_text},
        ]
        raw = await self.llm_client.ainvoke(messages, temperature=self.llm_temperature)
        self._debug(
            "model_raw",
            {
                "source": "llm",
                "mode": self.mode,
                "prompt_behavior_mode": self.prompt_behavior_mode,
                "action_space_field_count": len(action_space.get("fields") or []),
                "validation_feedback_count": len(validation_feedback or []),
                "raw": raw,
            },
        )
        parsed = _safe_json_loads(raw)
        if parsed is None:
            raise ValueError(f"LLM did not return JSON array. Raw:\n{raw}")
        return _validate_plan(parsed, schema=schema, action_space=action_space)

    async def _run_vlm(
        self,
        schema: Dict[str, Any],
        action_space: Optional[Dict[str, Any]],
        screenshot_path: Optional[Path],
        draft_plan: Optional[List[Dict[str, Any]]] = None,
        validation_feedback: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        action_space = action_space or {}
        action_meta = action_space.get("meta") or {}
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"ACTION_SPACE_META:\n{json.dumps(action_meta, indent=2)}"},
            {"type": "text", "text": f"ACTION_SPACE:\n{json.dumps(action_space, indent=2)}"},
            {
                "type": "text",
                "text": f"SCHEMA_META:\n{json.dumps({'selectors_count': len((schema or {}).get('selectors') or {}), 'shapes_count': len((schema or {}).get('shapes') or []), 'has_captcha': bool((schema or {}).get('captcha'))}, indent=2)}",
            },
        ]

        if screenshot_path and screenshot_path.exists():
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_as_data_url(screenshot_path)},
                }
            )

        if draft_plan is not None:
            user_content.append({"type": "text", "text": f"DRAFT_PLAN:\n{json.dumps(draft_plan, indent=2)}"})
        if validation_feedback:
            joined = "\n".join(f"- {msg}" for msg in validation_feedback)
            user_content.append({"type": "text", "text": f"VALIDATION_FEEDBACK:\n{joined}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VLM},
            {"role": "user", "content": user_content},
        ]

        raw = await self.vlm_client.ainvoke(messages, temperature=self.vlm_temperature)
        self._debug(
            "model_raw",
            {
                "source": "vlm",
                "mode": self.mode,
                "prompt_behavior_mode": self.prompt_behavior_mode,
                "raw": raw,
                "action_space_field_count": len(action_space.get("fields") or []),
                "has_screenshot": bool(screenshot_path and screenshot_path.exists()),
                "has_draft_plan": bool(draft_plan is not None),
                "validation_feedback_count": len(validation_feedback or []),
            },
        )
        parsed = _safe_json_loads(raw)
        if parsed is None:
            raise ValueError(f"VLM did not return JSON array. Raw:\n{raw}")

        return _validate_plan(parsed, schema=schema, action_space=action_space)

    async def plan(
        self,
        schema: Dict[str, Any],
        action_space: Optional[Dict[str, Any]] = None,
        screenshot_path: Optional[Path] = None,
        validation_feedback: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.mode == "llm_only":
            return await self._run_llm(
                schema,
                action_space=action_space,
                validation_feedback=validation_feedback,
            )
        if self.mode == "vlm_only":
            return await self._run_vlm(
                schema,
                action_space=action_space,
                screenshot_path=screenshot_path,
                validation_feedback=validation_feedback,
            )

        llm_draft = await self._run_llm(
            schema,
            action_space=action_space,
            validation_feedback=validation_feedback,
        )
        return await self._run_vlm(
            schema,
            action_space=action_space,
            screenshot_path=screenshot_path,
            draft_plan=llm_draft,
            validation_feedback=validation_feedback,
        )
