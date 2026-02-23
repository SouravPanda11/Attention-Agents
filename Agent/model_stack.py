import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BrainMode = Literal["llm_only", "vlm_only", "hybrid"]
_ENV_LOADED = False


def _load_env_file(path: Path):
    """
    Minimal .env loader with no third-party dependency.
    - Supports: KEY=VALUE
    - Ignores blank lines and comments (# ...)
    - Supports optional leading 'export '
    - Does not override environment variables already set by the shell.
    """
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _ensure_env_loaded():
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    explicit = os.getenv("AGENT_ENV_FILE", "").strip()
    if explicit:
        env_path = Path(explicit)
        if not env_path.is_absolute():
            env_path = (Path.cwd() / env_path).resolve()
        if not env_path.exists():
            raise RuntimeError(f"AGENT_ENV_FILE points to missing file: {env_path}")
        _load_env_file(env_path)
        _ENV_LOADED = True
        return

    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / ".env",
        here / ".env",
        here / ".env.agent",
    ]
    for candidate in candidates:
        if candidate.exists():
            _load_env_file(candidate)
            break
    _ENV_LOADED = True


def ensure_env_loaded() -> None:
    """Public wrapper so other modules can load Agent/.env before reading os.getenv."""
    _ensure_env_loaded()


def _normalize_openai_base(url: str) -> str:
    """
    Normalize base_url so our client can safely do:
      base_url + "/chat/completions"
    We require OpenAI-compatible endpoints, so it should end with /v1.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        raise RuntimeError("Empty base_url")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


@dataclass(frozen=True)
class ModelEndpoint:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    timeout_s: float = 90.0


@dataclass(frozen=True)
class ModelStackConfig:
    mode: BrainMode
    llm: ModelEndpoint
    vlm: ModelEndpoint
    llm_enabled: bool = True
    vlm_enabled: bool = True


def get_model_name_for_path(default: str = "default_model") -> str:
    """
    Resolve MODEL_NAME from env and sanitize it for filesystem-safe run folders.
    """
    _ensure_env_loaded()
    raw = os.getenv("MODEL_NAME", "").strip()
    if not raw:
        raw = default
    # Keep only safe path segment characters; collapse everything else to '_'.
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe or default


def load_model_stack() -> ModelStackConfig:
    """
    Central place to configure local/open-source LLM + VLM endpoints.
    Defaults are LM Studio-friendly. Override via env vars.
    The loader also reads .env from the current working directory or Agent/.env.

    Required for OpenAI-compatible servers:
      BASE_URL should point to .../v1
    """
    _ensure_env_loaded()
    mode = os.getenv("AGENT_BRAIN_MODE", "llm_only").strip().lower()
    if mode not in {"llm_only", "vlm_only", "hybrid"}:
        raise RuntimeError("AGENT_BRAIN_MODE must be one of: llm_only, vlm_only, hybrid")

    # LM Studio common default:
    default_base = "http://127.0.0.1:1234/v1"
    default_key = "local"

    llm_enabled = os.getenv("LLM_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
    vlm_enabled = os.getenv("VLM_ENABLED", "0").strip().lower() not in {"0", "false", "no"}

    llm = ModelEndpoint(
        base_url=_normalize_openai_base(os.getenv("LLM_BASE_URL", default_base)),
        model=os.getenv("LLM_MODEL", "qwen2.5-7b-instruct"),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", default_key)),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", "90")),
    )

    vlm = ModelEndpoint(
        base_url=_normalize_openai_base(os.getenv("VLM_BASE_URL", default_base)),
        model=os.getenv("VLM_MODEL", "qwen2-vl-7b-instruct"),
        api_key=os.getenv("VLM_API_KEY", os.getenv("OPENAI_API_KEY", default_key)),
        temperature=float(os.getenv("VLM_TEMPERATURE", "0.0")),
        timeout_s=float(os.getenv("VLM_TIMEOUT_S", "90")),
    )

    # Guardrails so you don't accidentally run hybrid without VLM configured
    if mode == "vlm_only" and not vlm_enabled:
        raise RuntimeError("AGENT_BRAIN_MODE=vlm_only but VLM_ENABLED=0. Set VLM_ENABLED=1.")
    if mode == "hybrid" and (not llm_enabled or not vlm_enabled):
        raise RuntimeError("AGENT_BRAIN_MODE=hybrid requires LLM_ENABLED=1 and VLM_ENABLED=1.")

    return ModelStackConfig(
        mode=mode,
        llm=llm,
        vlm=vlm,
        llm_enabled=llm_enabled,
        vlm_enabled=vlm_enabled,
    )
