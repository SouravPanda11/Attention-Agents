import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from model_stack import ModelStackConfig

SYSTEM_PROMPT_LLM = """You are a web form planning agent.
Input: DOM-derived schema with selectors/fields/shapes.
Output: ONLY a JSON array of tool calls.

Allowed tools:
- {"tool":"fill","selector":"...","value":"..."}
- {"tool":"select","selector":"...","value":"..."}
- {"tool":"click","selector":"..."}
- {"tool":"check","selector":"..."}
- {"tool":"set_range","selector":"...","value":5}
- {"tool":"done"}

Rules:
- Use only selectors present in schema.selectors, or shape selectors:
  [data-shape-id="<id>"].
- Complete ALL interactions requested by schema.fields (respect field["interaction"]).
- For select: never choose placeholder ("Select..." or empty value) if other options exist.
- For radio groups: use tool="check" with one selector from field.options only (#id or nth selector).
- Do NOT use [value=...] selectors for radios.
- For check generally: choose one option selector from field.options (not the group selector).
- For set_range: choose a mid value, e.g. (min+max)//2.
- If input.type == "number", value must be a numeric string within [min,max] when provided.
- Never put free text into number fields.
- Prefer filling all required fields before clicking Next/Submit.
- Complete text attention checks.
- Always end by clicking a visible Next or Submit control when present.
No markdown, no prose, only valid JSON array.
"""

SYSTEM_PROMPT_VLM = """You are a multimodal web form planning agent.
Inputs:
1) DOM-derived schema
2) Screenshot
Optional:
3) Draft plan from a text model

Return ONLY a JSON array of tool calls.

Allowed tools:
- {"tool":"fill","selector":"...","value":"..."}
- {"tool":"select","selector":"...","value":"..."}
- {"tool":"click","selector":"..."}
- {"tool":"check","selector":"..."}
- {"tool":"set_range","selector":"...","value":5}
- {"tool":"done"}

Rules:
- Use only selectors present in schema.selectors, or shape selectors:
  [data-shape-id="<id>"].
- Complete ALL interactions requested by schema.fields (respect field["interaction"]).
- For select: never choose placeholder ("Select..." or empty value) if other options exist.
- For radio groups: use tool="check" with one selector from field.options only (#id or nth selector).
- Do NOT use [value=...] selectors for radios.
- For check generally: choose one option selector from field.options (not the group selector).
- For set_range: choose a mid value, e.g. (min+max)//2.
- If input.type == "number", value must be a numeric string within [min,max] when provided.
- Never put free text into number fields.
- Prefer filling all required fields before clicking Next/Submit.
- Satisfy text and image attention checks.
- If draft plan is provided, improve/fix it.
- Always end by clicking a visible Next or Submit control when present.
No markdown, no prose, only valid JSON array.
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


def _validate_plan(plan: Any, schema: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not isinstance(plan, list):
        raise ValueError(f"Model output must be a JSON array, got: {type(plan).__name__}")

    valid_tools = {"fill", "select", "click", "check", "set_range", "done"}
    validated: List[Dict[str, Any]] = []
    fields = (schema or {}).get("fields", [])
    selector_pool = set(((schema or {}).get("selectors") or {}).values())
    fields_by_selector = {
        f.get("selector"): f
        for f in fields
        if isinstance(f, dict) and isinstance(f.get("selector"), str)
    }
    check_option_selectors = {
        opt.get("selector")
        for f in fields
        if f.get("interaction") == "check"
        for opt in (f.get("options") or [])
        if opt.get("selector")
    }

    for step in plan:
        if not isinstance(step, dict):
            raise ValueError(f"Each step must be an object, got: {step!r}")

        tool = step.get("tool")
        if tool not in valid_tools:
            raise ValueError(f"Invalid tool in step: {step!r}")
        if tool == "done":
            validated.append(step)
            continue

        selector = step.get("selector")
        if not isinstance(selector, str) or not selector:
            raise ValueError(f"Missing/invalid selector in step: {step!r}")
        if selector_pool and selector not in selector_pool and not selector.startswith('[data-shape-id="'):
            if tool != "check" or selector not in check_option_selectors:
                raise ValueError(f"Selector not in schema.selectors/options: {step!r}")
        if tool in {"fill", "select"} and (not isinstance(step.get("value"), str) or not step["value"]):
            raise ValueError(f"Missing/invalid value for {tool}: {step!r}")
        if tool == "fill":
            field = fields_by_selector.get(selector) or {}
            input_type = (field.get("inputType") or field.get("type") or "").lower()
            if input_type == "number":
                value = step.get("value")
                if not isinstance(value, str) or not value.strip().lstrip("+-").replace(".", "", 1).isdigit():
                    raise ValueError(f"Fill value for number field must be numeric string: {step!r}")
                n = float(value)
                min_v = field.get("min")
                max_v = field.get("max")
                if min_v is not None and n < float(min_v):
                    raise ValueError(f"Number value below min constraint: {step!r}")
                if max_v is not None and n > float(max_v):
                    raise ValueError(f"Number value above max constraint: {step!r}")
        if tool == "select":
            field = fields_by_selector.get(selector) or {}
            options = field.get("options") or []
            if options:
                value = step.get("value")
                valid_values = [o.get("value") for o in options if o.get("value") != ""]
                if valid_values and value not in valid_values:
                    raise ValueError(f"Select value must be non-placeholder option value: {step!r}")
        if tool == "set_range":
            value = step.get("value")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Missing/invalid numeric value for set_range: {step!r}")
        validated.append(step)

    return validated


def _autofill_missing_fields(plan: List[Dict[str, Any]], schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    completed = {
        (step.get("tool"), step.get("selector"))
        for step in plan
        if isinstance(step, dict)
    }

    generated: List[Dict[str, Any]] = []
    for field in schema.get("fields", []):
        interaction = field.get("interaction")
        selector = field.get("selector")
        if not interaction or not selector:
            continue
        if interaction not in {"fill", "select", "check", "set_range"}:
            continue

        # For check fields, completion is tracked on option selector (not group selector).
        if interaction == "check":
            options = field.get("options") or []
            if any((interaction, opt.get("selector")) in completed for opt in options if opt.get("selector")):
                continue
            target = next((opt for opt in options if opt.get("selector")), None)
            if target:
                generated.append({"tool": "check", "selector": target["selector"]})
            continue

        if (interaction, selector) in completed:
            continue

        if interaction == "fill":
            input_type = (field.get("inputType") or field.get("type") or "").lower()
            if input_type == "number":
                min_v = field.get("min")
                max_v = field.get("max")
                if min_v is not None and max_v is not None:
                    n = int((float(min_v) + float(max_v)) // 2)
                elif min_v is not None:
                    n = int(float(min_v))
                elif max_v is not None:
                    n = int(float(max_v))
                else:
                    n = 25
                generated.append({"tool": "fill", "selector": selector, "value": str(n)})
            else:
                generated.append({"tool": "fill", "selector": selector, "value": "Sample response"})
        elif interaction == "select":
            options = field.get("options") or []
            non_placeholder = [
                o for o in options if o.get("value") and "select" not in (o.get("label") or "").strip().lower()
            ]
            target = (non_placeholder[0] if non_placeholder else (options[0] if options else None))
            if target and target.get("value"):
                generated.append({"tool": "select", "selector": selector, "value": target["value"]})
        elif interaction == "set_range":
            min_v = int(field.get("min", 0))
            max_v = int(field.get("max", 100))
            mid = (min_v + max_v) // 2
            generated.append({"tool": "set_range", "selector": selector, "value": mid})

    check_option_selectors = {
        opt.get("selector")
        for f in schema.get("fields", [])
        if f.get("interaction") == "check"
        for opt in (f.get("options") or [])
        if opt.get("selector")
    }

    non_end_steps = []
    end_click = None
    seen_pairs = set()
    for step in plan:
        if not isinstance(step, dict):
            continue
        if step.get("tool") == "check" and check_option_selectors and step.get("selector") not in check_option_selectors:
            # Drop coarse group-level checks and keep option-level checks only.
            continue
        pair = (step.get("tool"), step.get("selector"))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if (
            step.get("tool") == "click"
            and isinstance(step.get("selector"), str)
            and ("next" in step["selector"].lower() or "submit" in step["selector"].lower())
        ):
            end_click = step
            continue
        non_end_steps.append(step)

    final_plan = [*generated, *non_end_steps]
    if end_click:
        final_plan.append(end_click)
    else:
        next_sel = schema.get("selectors", {}).get("next")
        submit_sel = schema.get("selectors", {}).get("submit")
        if next_sel:
            final_plan.append({"tool": "click", "selector": next_sel})
        elif submit_sel:
            final_plan.append({"tool": "click", "selector": submit_sel})

    return final_plan


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
    async def plan(self, schema: Dict[str, Any], screenshot_path: Optional[Path] = None) -> List[Dict[str, Any]]:
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

    @staticmethod
    def _image_as_data_url(path: Path) -> str:
        mime = "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def _run_llm(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LLM},
            {"role": "user", "content": f"SCHEMA:\n{json.dumps(schema, indent=2)}"},
        ]
        raw = await self.llm_client.ainvoke(messages, temperature=self.llm_temperature)
        parsed = _safe_json_loads(raw)
        if parsed is None:
            raise ValueError(f"LLM did not return JSON array. Raw:\n{raw}")
        validated = _validate_plan(parsed, schema=schema)
        return _autofill_missing_fields(validated, schema)

    async def _run_vlm(
        self,
        schema: Dict[str, Any],
        screenshot_path: Optional[Path],
        draft_plan: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"SCHEMA:\n{json.dumps(schema, indent=2)}"}]

        if screenshot_path and screenshot_path.exists():
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_as_data_url(screenshot_path)},
                }
            )

        if draft_plan is not None:
            user_content.append({"type": "text", "text": f"DRAFT_PLAN:\n{json.dumps(draft_plan, indent=2)}"})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VLM},
            {"role": "user", "content": user_content},
        ]

        raw = await self.vlm_client.ainvoke(messages, temperature=self.vlm_temperature)
        parsed = _safe_json_loads(raw)
        if parsed is None:
            raise ValueError(f"VLM did not return JSON array. Raw:\n{raw}")

        validated = _validate_plan(parsed, schema=schema)
        return _autofill_missing_fields(validated, schema)

    async def plan(self, schema: Dict[str, Any], screenshot_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        if self.mode == "llm_only":
            return await self._run_llm(schema)
        if self.mode == "vlm_only":
            return await self._run_vlm(schema, screenshot_path=screenshot_path)

        llm_draft = await self._run_llm(schema)
        return await self._run_vlm(schema, screenshot_path=screenshot_path, draft_plan=llm_draft)
