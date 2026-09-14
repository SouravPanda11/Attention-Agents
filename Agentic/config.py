"""Configuration belongs exclusively to Agentic; never loads Agent/.env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env():
    path = ROOT / ".env"
    if not path.exists():
        return
    values = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if not separator:
            raise ValueError(f"Invalid configuration line in {path}")
        values[key.strip()] = value.strip().strip("\"'")
    for key, value in values.items():
        os.environ.setdefault(key, value)


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def runtime_defaults():
    """Resolve the familiar endpoint sections, retaining earlier AGENTIC_* aliases."""
    mode = os.getenv("AGENT_BRAIN_MODE", "llm_only").strip()
    if mode not in {"llm_only", "vlm_only"}:
        raise ValueError("AGENT_BRAIN_MODE must be llm_only or vlm_only; hybrid is not implemented")
    prefix = "LLM" if mode == "llm_only" else "VLM"
    enabled = os.getenv(f"{prefix}_ENABLED", "1").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        raise ValueError(f"{prefix}_ENABLED must be 1 for {mode}")

    def setting(name, alias, default):
        return os.getenv(name, os.getenv(alias, default))

    return {
        "base_url": setting("SURVEY_TARGET", "AGENTIC_BASE_URL", "http://127.0.0.1:3001"),
        "suite_version": os.getenv("SURVEY_VERSION", "v0"),
        "occurrence": os.getenv("SURVEY_OCCURRENCE", "o1"),
        "orders": os.getenv("SURVEY_ORDERS", "order01 order02 order03").split(),
        "themes": os.getenv("SURVEY_THEMES", "all").split(),
        "repeats": os.getenv("SURVEY_REPEATS", "1"),
        "model": setting(f"{prefix}_MODEL", "AGENTIC_MODEL", ""),
        "model_name": os.getenv("MODEL_NAME", "").strip(),
        "lm_base_url": setting(f"{prefix}_BASE_URL", "AGENTIC_LM_BASE_URL", "http://127.0.0.1:1234/v1"),
        "api_key": setting(f"{prefix}_API_KEY", "AGENTIC_API_KEY", os.getenv("OPENAI_API_KEY", "local")),
        "temperature": setting(f"{prefix}_TEMPERATURE", "AGENTIC_TEMPERATURE", "0"),
        "observation": ("vision" if mode == "vlm_only" else "dom") if "AGENT_BRAIN_MODE" in os.environ
                       else os.getenv("AGENTIC_OBSERVATION", "dom"),
    }
