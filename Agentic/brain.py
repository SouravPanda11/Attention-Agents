"""LM Studio chat completions with explicit, logged JSON action plans."""
import asyncio
import base64
import json
import time
import urllib.error
import urllib.request

PROMPT_VERSION = "agentic-zero-shot"

# Prompt behavior toggle:
# - completion: mimic participants answering all items.
# - unconstrained: allow free progression behavior.
# Uncomment exactly one line. Explicit CLI/environment settings can override it.
PROMPT_BEHAVIOR_MODE = "completion"
# PROMPT_BEHAVIOR_MODE = "unconstrained"

PB_COMPLETION = """Hard requirement: answer the current question before finishing it.
Partial answer plans are invalid: include all actions needed to answer the current question."""
PB_UNCONSTRAINED = ""

TOOL_DEFINITIONS = """Tool definitions (standard web-form actions):
- fill:
  Signature: {"tool":"fill","key":"<field_key>","value":"<string>"}
  Use for: short-text inputs, long-text textareas, and numeric inputs.

- select:
  Signature: {"tool":"select","key":"<field_key>","value":"<option_value>"}
  Use for: HTML select dropdowns.

- check:
  Signature: {"tool":"check","key":"<field_key>","value":"<option_label_or_value>"}
  Use for: radio groups, Likert scales, image single-selection options, and checkboxes.
  For multiple selections, use one action per option. Checking a selected option leaves it selected.

- uncheck:
  Signature: {"tool":"uncheck","key":"<field_key>","value":"<option_label_or_value>"}
  Use for: removing a checkbox selection.

- set_range:
  Signature: {"tool":"set_range","key":"<field_key>","value":<number>}
  Use for: sliders, with a value matching the field's minimum, maximum, and step.
  An explicit action is needed even when choosing the initial displayed value.

- rank:
  Signature: {"tool":"rank","key":"<field_key>","value":["<item_value>","<item_value>"]}
  Use for: ranking items. Include every item value exactly once in the desired order.
  An explicit action is needed even when keeping the initial displayed order.

- click:
  Signature: {"tool":"click","key":"<field_key>"}
  Use for: available buttons and navigation controls, including Start Survey and Submit.

- done:
  Signature: {"tool":"done"}
  Use for: finishing the current question without further actions. This does not submit the survey.
  Emit done as the only action in its array.
"""


def build_system_prompt(behavior=None):
    behavior = PROMPT_BEHAVIOR_MODE if behavior is None else behavior
    if behavior not in {"completion", "unconstrained"}:
        raise ValueError(f"Unknown prompt behavior: {behavior}")
    behavior_block = PB_COMPLETION if behavior == "completion" else PB_UNCONSTRAINED
    return f"""You are going to act as an online survey participant and answer questions for an online web survey.
Follow the instructions provided in the survey.
Read the questions and their respective options carefully.

{behavior_block}

{TOOL_DEFINITIONS}

Return ONLY a JSON array of action intents.
Use ACTION_SPACE as the source of truth for valid keys/options/ranges.

Interaction requirements:
- Use keys from ACTION_SPACE fields only.
- For check and uncheck, use an exact option value or an unambiguous exact option label from ACTION_SPACE.
- For select, use an exact option value from ACTION_SPACE, not a placeholder.
- Put any navigation click last in the action array.
"""


SYSTEM_PROMPT = build_system_prompt()


def request_json(url, payload=None, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def normalize_endpoint(url):
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def messages_for(observation, feedback, behavior=None, images=()):
    prompt = build_system_prompt(behavior)
    content = [{"type": "text", "text": json.dumps({
        "ACTION_SPACE": observation, "FEEDBACK": feedback,
    }, ensure_ascii=False)}]
    for path in images:
        content.append({"type": "text", "text": f"Rendered image question: {path.stem}"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        }})
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


async def complete(args, messages):
    payload = {"model": args.model, "messages": messages, "temperature": 0, "stream": False}
    started = time.perf_counter()
    response = await asyncio.to_thread(request_json, normalize_endpoint(args.lm_base_url) +
                                       "/chat/completions", payload, args.api_key)
    return response, time.perf_counter() - started


def parse_plan(response):
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("Model did not return text containing a JSON array")
    content = content.strip()
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    plan = json.loads(content)
    if not isinstance(plan, list) or not plan or len(plan) > 100:
        raise ValueError("Expected a nonempty JSON action array with at most 100 actions")
    return plan
