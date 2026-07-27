"""Multi-provider chat completions client for evocast.

Adapted from research_automation/scripts/deepseek_api_client.py.
Provides API calling, logging, retry, JSON parsing, schema validation,
and optional multi-turn read-only tool calling.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import sys
import uuid
from datetime import datetime
from pathlib import Path
from contextlib import suppress
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evocast.domain.config_paths import resolve_config_path
from evocast.domain.knowledge_paths import runtime_root, task_knowledge_dir, task_runs_dir
from evocast.state.cost_ledger import append_cost_event, estimate_llm_cost
from evocast.state.domain_store import load_task_config


AGENT_DIR = Path(__file__).resolve().parent
EVOCAST_DIR = AGENT_DIR.parent
CONFIGS_DIR = EVOCAST_DIR / "configs"
LOGS_DIR = runtime_root() / "runs"
FLASH_MODEL_NAME = "deepseek-v4-flash"
FORBIDDEN_MODEL_NAMES = {"deepseek-v4-pro"}
DEFAULT_API_CONFIG_NAME = "providers/deepseek.yaml"
GENERIC_API_CONFIG_ENV = "EVOCAST_API_CONFIG"
SUPPORTED_PROVIDERS = {"deepseek", "openai", "minimax"}
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504, 529}


class ProviderAPIError(RuntimeError):
    """Raised when the DeepSeek API call fails irrecoverably."""


def load_yaml_config(path: Path) -> Dict:
    """Load a provider YAML config file.

    Provider configuration is part of the task identity. Missing or malformed
    files must fail loudly instead of silently falling back to DeepSeek.
    """
    if not path.exists():
        raise ProviderAPIError(f"api_config_not_found: {path}")
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except ImportError:
        if path.suffix != ".json":
            raise
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ProviderAPIError(f"api_config_invalid_or_empty: {path}")
    return payload


def resolve_provider_config_path(value: str | Path | None, *, base_dir: str | Path | None = None) -> Path:
    """Resolve an API provider config against task-local or packaged configs."""
    raw = str(value or "").strip()
    if not raw:
        raw = DEFAULT_API_CONFIG_NAME
    path = Path(raw)
    if path.is_absolute():
        resolved = path
    elif path.exists():
        resolved = path
    else:
        candidates: List[Path] = []
        if base_dir is not None:
            runtime = runtime_root(str(base_dir))
            repo_like_root = runtime.parent if runtime.name == ".evocast" else Path(base_dir)
            candidates.extend(
                [
                    repo_like_root / "evocast" / "configs" / raw,
                    repo_like_root / "configs" / raw,
                ]
            )
        candidates.append(resolve_config_path(raw))
        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[-1])
    if not resolved.exists():
        raise ProviderAPIError(f"api_config_not_found: {raw} -> {resolved}")
    return resolved


def _read_task_api_config(base_dir: str, task_id: str) -> str:
    payload = load_task_config(base_dir, task_id)
    return str(
        payload.get("api_config")
        or payload.get("provider_config")
        or payload.get("llm_provider_config")
        or ""
    ).strip()


def resolve_task_api_config_path(
    *,
    base_dir: str,
    task_id: str,
    explicit_config: str | Path | None = None,
) -> Path:
    """Resolve the provider config for a task using the canonical precedence.

    Precedence: explicit argument -> task_config/api_config -> EVOCAST_API_CONFIG
    -> packaged default.
    """
    value = str(explicit_config or "").strip()
    if not value:
        value = _read_task_api_config(base_dir, task_id)
    if not value:
        value = os.environ.get(GENERIC_API_CONFIG_ENV, "").strip()
    if not value:
        value = DEFAULT_API_CONFIG_NAME
    return resolve_provider_config_path(value, base_dir=base_dir)


def create_task_client(
    *,
    base_dir: str,
    task_id: str,
    explicit_config: str | Path | None = None,
) -> "ProviderClient":
    """Create an LLM client from the task's provider config."""
    return ProviderClient(
        config_path=resolve_task_api_config_path(
            base_dir=base_dir,
            task_id=task_id,
            explicit_config=explicit_config,
        ),
        task_id=task_id,
        base_dir=base_dir,
    )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path, text):
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def remove_file_if_exists(path) -> None:
    with suppress(FileNotFoundError, OSError):
        os.remove(str(path))


_STDOUT_BROKEN = False


def safe_api_print(*args: Any, **kwargs: Any) -> None:
    """Best-effort API progress printing that never aborts a run."""
    global _STDOUT_BROKEN
    if _STDOUT_BROKEN:
        return
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        return
    except (BrokenPipeError, OSError):
        _STDOUT_BROKEN = True
        with suppress(Exception):
            sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _api_error_payload(
    *,
    stage: str,
    round_num: int,
    attempt: int,
    provider: str,
    model: Any,
    error_type: str,
    error_message: str,
    content_chars: int = 0,
    reasoning_chars: int = 0,
    stream_finished: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "round": round_num,
        "attempt": attempt,
        "generated_at": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "error_type": error_type,
        "error_message": error_message,
        "content_chars": int(content_chars or 0),
        "reasoning_chars": int(reasoning_chars or 0),
        "stream_finished": stream_finished,
    }


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON text."""
    text = text.strip()
    if "```json" in text.lower():
        start = text.lower().find("```json")
        text = text[start:]
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _candidate_json_object_slices(text: str) -> List[str]:
    candidates: List[str] = []
    for marker in ('"decision"', '"status"', '"error"', '"reason"'):
        idx = text.find(marker)
        if idx > 0:
            start = text.rfind("{", 0, idx)
            end = text.rfind("}")
            if start != -1 and end > start:
                candidate = text[start : end + 1].strip()
                if candidate and candidate not in candidates:
                    candidates.append(candidate)

    in_string = False
    escaped = False
    depth = 0
    start: Optional[int] = None

    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : idx + 1].strip()
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
                start = None

    if not candidates:
        starts = [idx for idx, char in enumerate(text) if char == "{"]
        ends = [idx for idx, char in enumerate(text) if char == "}"]
        for start_idx in reversed(starts):
            for end_idx in reversed(ends):
                if end_idx <= start_idx:
                    continue
                candidate = text[start_idx : end_idx + 1].strip()
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
                break
    return candidates


def _escape_control_chars_in_json_strings(text: str) -> str:
    """Escape raw newlines/tabs emitted inside JSON strings by some providers."""

    out: List[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char == '"':
                out.append(char)
                in_string = False
                continue
            if char == "\n":
                out.append("\\n")
                continue
            if char == "\r":
                out.append("\\r")
                continue
            if char == "\t":
                out.append("\\t")
                continue
            out.append(char)
            continue

        out.append(char)
        if char == '"':
            in_string = True
    return "".join(out)


def _repair_common_json_annotation_leaks(text: str) -> str:
    text = re.sub(
        r':\s*(-?\d+(?:\.\d+)?)\s*\(([^()\n{}[\]"]+)\)"?',
        lambda match: ": " + json.dumps(f"{match.group(1)} ({match.group(2)})", ensure_ascii=False),
        text,
    )
    text = text.replace('},"{', '},{')
    text = text.replace(']","{', '],{')
    text = text.replace('}","{', '},{')
    return text


def _repair_missing_array_item_object_closers(text: str) -> str:
    """Repair common LLM JSON where an array item misses its outer closer.

    Some providers correctly close a nested object such as ``edit_spec`` but
    forget the extra ``}`` for the enclosing list item before starting the next
    item.  Keep this syntax-only: only add closers at clear item boundaries
    headed by stable id fields, and do not invent missing semantic fields.
    """

    item_start = r'{\s*"(?:target_id|question_id|candidate_id|action_id|opportunity_id|mechanism_id)"\s*:'
    next_parent_field = (
        r'"(?:selection_rationale|preserve_contract|expected_behavior_delta|'
        r'source_influence|research_context_summary|evidence|raw_response|created_at)"\s*:'
    )
    text = re.sub(
        r'("risk"\s*:\s*"(?:low|medium|high)"\s*})\s*,\s*(' + item_start + r")",
        r"\1}, \2",
        text,
    )
    return re.sub(
        r'("risk"\s*:\s*"(?:low|medium|high)"\s*})\s*]\s*,\s*(' + next_parent_field + r")",
        r"\1}], \2",
        text,
    )


def _close_balanced_json_suffix(text: str) -> str:
    if text.count("{") > text.count("}"):
        text = text + ("}" * (text.count("{") - text.count("}")))
    if text.count("[") > text.count("]"):
        text = text + ("]" * (text.count("[") - text.count("]")))
    return text


def _json_answer_suffixes(text: str) -> List[str]:
    suffixes: List[str] = []
    lower = text.lower()
    for marker in ("</think>", "</analysis>"):
        idx = lower.rfind(marker)
        if idx != -1:
            suffix = text[idx + len(marker) :].strip()
            if suffix and suffix not in suffixes:
                suffixes.append(suffix)
    if "```json" in lower:
        idx = lower.rfind("```json")
        suffix = strip_json_fences(text[idx:]).strip()
        if suffix and suffix not in suffixes:
            suffixes.append(suffix)
    return suffixes


def _json_object_has_required_key(
    parsed: Dict[str, Any],
    required_top_level_keys: Optional[Iterable[str]],
    *,
    require_all: bool = False,
) -> bool:
    if not required_top_level_keys:
        return True
    keys = [str(key) for key in required_top_level_keys]
    # Stage-level recovery keys are alternatives (for example ``status`` or
    # ``items``).  A declared response schema, however, is a contract: a
    # nested object containing only one coincidental field must never be
    # accepted as the full answer.
    return all(key in parsed for key in keys) if require_all else any(key in parsed for key in keys)


def _json_text_ends_inside_string(text: str) -> bool:
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    return in_string


def _raise_missing_required_json_key(text: str, required_top_level_keys: Iterable[str]) -> None:
    keys = ", ".join(str(key) for key in required_top_level_keys)
    raise json.JSONDecodeError(f"JSON object missing required top-level key(s): {keys}", text, 0)


def _coerce_json_object(
    parsed: Any,
    source_text: str,
    required_top_level_keys: Optional[Iterable[str]],
    *,
    error_label: str,
    require_all_required_keys: bool = False,
) -> Dict:
    if isinstance(parsed, list):
        parsed = {"items": parsed}
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError(f"{error_label} JSON value is not an object", source_text, 0)
    if not _json_object_has_required_key(
        parsed,
        required_top_level_keys,
        require_all=require_all_required_keys,
    ):
        _raise_missing_required_json_key(source_text, required_top_level_keys or [])
    return parsed


def parse_json_object(
    text: str,
    required_top_level_keys: Optional[Iterable[str]] = None,
    *,
    require_all_required_keys: bool = False,
) -> Dict:
    """Parse a JSON object string, with repair fallbacks."""
    text = strip_json_fences(text)
    try:
        parsed = json.loads(text)
        return _coerce_json_object(
            parsed, text, required_top_level_keys, error_label="top-level",
            require_all_required_keys=require_all_required_keys,
        )
    except json.JSONDecodeError as exc:
        last_error = exc

    try:
        repaired = _repair_missing_array_item_object_closers(_escape_control_chars_in_json_strings(text))
        parsed = json.loads(repaired)
        return _coerce_json_object(
            parsed, repaired, required_top_level_keys, error_label="array-item repaired",
            require_all_required_keys=require_all_required_keys,
        )
    except json.JSONDecodeError:
        pass

    for suffix in _json_answer_suffixes(text):
        for repaired in (
            suffix,
            _escape_control_chars_in_json_strings(suffix),
            _repair_common_json_annotation_leaks(
                _repair_missing_array_item_object_closers(_escape_control_chars_in_json_strings(suffix))
            ),
            _close_balanced_json_suffix(
                _repair_common_json_annotation_leaks(
                    _repair_missing_array_item_object_closers(_escape_control_chars_in_json_strings(suffix))
                )
            ),
        ):
            try:
                parsed = json.loads(repaired)
                return _coerce_json_object(
                    parsed, repaired, required_top_level_keys, error_label="answer-suffix repaired",
                    require_all_required_keys=require_all_required_keys,
                )
            except json.JSONDecodeError:
                pass

    for candidate in reversed(_candidate_json_object_slices(text)):
        try:
            parsed = json.loads(candidate)
            return _coerce_json_object(
                parsed, candidate, required_top_level_keys, error_label="sliced",
                require_all_required_keys=require_all_required_keys,
            )
        except json.JSONDecodeError:
            pass
        try:
            repaired = _escape_control_chars_in_json_strings(candidate)
            parsed = json.loads(repaired)
            return _coerce_json_object(
                parsed, repaired, required_top_level_keys, error_label="control-char repaired",
                require_all_required_keys=require_all_required_keys,
            )
        except json.JSONDecodeError:
            pass
        try:
            repaired = _repair_common_json_annotation_leaks(
                _repair_missing_array_item_object_closers(_escape_control_chars_in_json_strings(candidate))
            )
            parsed = json.loads(repaired)
            return _coerce_json_object(
                parsed, repaired, required_top_level_keys, error_label="annotation repaired",
                require_all_required_keys=require_all_required_keys,
            )
        except json.JSONDecodeError:
            pass
        try:
            repaired = _close_balanced_json_suffix(
                _repair_common_json_annotation_leaks(
                    _repair_missing_array_item_object_closers(_escape_control_chars_in_json_strings(candidate))
                )
            )
            parsed = json.loads(repaired)
            return _coerce_json_object(
                parsed, repaired, required_top_level_keys, error_label="closed",
                require_all_required_keys=require_all_required_keys,
            )
        except json.JSONDecodeError:
            pass

    if required_top_level_keys and _json_text_ends_inside_string(text):
        raise last_error

    try:
        from json_repair import repair_json

        repaired = repair_json(text)
        parsed = json.loads(repaired)
        return _coerce_json_object(
            parsed, repaired, required_top_level_keys, error_label="top-level repaired",
            require_all_required_keys=require_all_required_keys,
        )
    except (ImportError, Exception):
        pass

    raise last_error


def _load_env_key(key: str) -> Optional[str]:
    """Load an API key from .env files (no python-dotenv dependency)."""
    candidates = [
        EVOCAST_DIR.parent / ".env",
        EVOCAST_DIR / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    val = v.strip().strip('"').strip("'")
                    if val:
                        return val
        except OSError:
            continue
    return None


def _normalize_api_key(value: Optional[str]) -> Optional[str]:
    """Return a usable API key or None for empty/placeholder values."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return None
    if text.lower() in {"none", "null", "nil", "changeme", "your_api_key", "your_api_key_here"}:
        return None
    return text


class ProviderClient:
    """Chat-completions client with logging, retry, and tool-calling support."""

    def __init__(self, config_path: Optional[Path] = None, task_id: str = "default", base_dir: str | None = None):
        if config_path is None:
            override = os.environ.get(GENERIC_API_CONFIG_ENV, "")
            if override:
                config_path = resolve_provider_config_path(override)
            else:
                config_path = resolve_provider_config_path(DEFAULT_API_CONFIG_NAME)
        else:
            config_path = resolve_provider_config_path(config_path)
        self.config = load_yaml_config(config_path)
        self.config_path = config_path
        self.task_id = task_id
        self.base_dir = str(runtime_root(base_dir)) if base_dir else str(runtime_root())

        api_key_env = self.config.get("api_key_env", "DEEPSEEK_API_KEY")
        self.api_key = _normalize_api_key(os.environ.get(api_key_env))
        if not self.api_key:
            for fallback in self.config.get("api_key_fallback_env", []):
                self.api_key = _normalize_api_key(os.environ.get(fallback))
                if self.api_key:
                    break
        if not self.api_key:
            self.api_key = _normalize_api_key(_load_env_key(api_key_env))

        self.base_url = self.config.get("base_url", "https://api.deepseek.com").rstrip("/")
        self._api_available = bool(self.api_key)
        self._api_log_dir = task_runs_dir(self.base_dir, task_id) / "logs" / "api"

    @property
    def api_available(self) -> bool:
        return self._api_available

    def _ensure_api_key(self):
        if not self.api_key:
            raise ProviderAPIError(
                "No API key configured. Set the provider key environment variable "
                f"or configure configs/{DEFAULT_API_CONFIG_NAME}."
            )

    @staticmethod
    def _resolve_model_name(model_name: Any) -> str:
        """Disallow forbidden models and fall back to flash."""
        raw = str(model_name or FLASH_MODEL_NAME).strip()
        if not raw:
            return FLASH_MODEL_NAME
        if raw.lower() in FORBIDDEN_MODEL_NAMES:
            return FLASH_MODEL_NAME
        return raw

    @staticmethod
    def _resolve_provider_name(provider_name: Any) -> str:
        raw = str(provider_name or "deepseek").strip().lower()
        return raw if raw in SUPPORTED_PROVIDERS else "deepseek"

    @staticmethod
    def _normalize_stage_name(stage: str) -> str:
        """Map internal/logging stage aliases back to canonical config stages."""
        value = str(stage or "").strip()
        if not value:
            return value

        alias_map = {
            "proposal_generation": "proposal",
            "proposal_understanding": "proposal",
            "proposal_recovery": "proposal",
            "proposal_r1": "proposal",
        }
        if value in alias_map:
            return alias_map[value]
        return value

    def stage_config(self, stage: str) -> Dict:
        """Get configuration for a specific pipeline stage."""
        stage = self._normalize_stage_name(stage)
        cfg = self.config
        provider = self._resolve_provider_name(cfg.get("provider", "deepseek"))
        models = cfg.get("models", FLASH_MODEL_NAME)
        # Model identity is a task-level invariant.  Stage-specific routing
        # makes one research attempt silently mix different LLMs (for example
        # M3 for design and a different model for Builder), invalidating both
        # reproducibility and failure attribution.  Legacy mapping configs
        # resolve only their explicit default; role keys are deliberately
        # ignored.
        model = models.get("default", FLASH_MODEL_NAME) if isinstance(models, dict) else models
        model = self._resolve_model_name(model)

        thinking_cfg = cfg.get("thinking", {})
        thinking = thinking_cfg.get(stage, thinking_cfg.get("default", False)) if isinstance(thinking_cfg, dict) else bool(thinking_cfg)

        effort_cfg = cfg.get("effort", {})
        effort = effort_cfg.get(stage, effort_cfg.get("default", "high")) if isinstance(effort_cfg, dict) else effort_cfg
        effort = str(effort or "high").strip().lower()

        temp_cfg = cfg.get("temperature", {})
        temperature = float(temp_cfg.get(stage, temp_cfg.get("default", 0.1))) if isinstance(temp_cfg, dict) else float(temp_cfg)

        tokens_cfg = cfg.get("max_tokens", {})
        max_tokens = int(tokens_cfg.get(stage, tokens_cfg.get("default", 12000))) if isinstance(tokens_cfg, dict) else int(tokens_cfg)
        context_cfg = cfg.get("context_window", {})
        if isinstance(context_cfg, dict):
            context_window = int(
                context_cfg.get(stage)
                or context_cfg.get(model)
                or context_cfg.get("default")
                or 0
            )
        else:
            context_window = int(context_cfg or 0)

        timeout_cfg = cfg.get("timeout_sec", 900)
        timeout_sec = int(
            timeout_cfg.get(stage, timeout_cfg.get("default", 900))
            if isinstance(timeout_cfg, dict)
            else timeout_cfg
        )
        retries_cfg = cfg.get("max_retries", 3)
        max_retries = int(
            retries_cfg.get(stage, retries_cfg.get("default", 3))
            if isinstance(retries_cfg, dict)
            else retries_cfg
        )

        return {
            "provider": provider,
            "model": model,
            "thinking": bool(thinking),
            "effort": effort,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_window": context_window,
            "chat_completions_path": str(cfg.get("chat_completions_path", "/chat/completions") or "/chat/completions"),
            "token_param": str(cfg.get("token_param", "") or "").strip().lower(),
            "minimax_reasoning_split": bool(cfg.get("minimax_reasoning_split", True)),
            "timeout_sec": timeout_sec,
            "stream": bool(cfg.get("stream", False)),
            "max_retries": max_retries,
        }

    @staticmethod
    def _extract_message_payload(result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize API response payload into a message dict."""
        choices = result.get("choices") or []
        if not choices:
            raise ProviderAPIError("API response missing choices")
        first_choice = choices[0] or {}
        msg = first_choice.get("message")
        if not isinstance(msg, dict):
            raise ProviderAPIError("API response missing message payload")
        return msg

    @staticmethod
    def _provider_token_param(provider: str, configured: str) -> str:
        configured = str(configured or "").strip()
        if configured:
            return configured
        if provider == "openai":
            return "max_completion_tokens"
        return "max_tokens"

    @staticmethod
    def _provider_temperature(provider: str, value: float) -> float:
        if provider == "minimax":
            return min(max(float(value), 0.01), 1.0)
        return float(value)

    @staticmethod
    def _provider_reasoning_effort(provider: str, effort: str) -> str:
        value = str(effort or "").strip().lower() or "high"
        if provider == "deepseek":
            if value in {"low", "medium"}:
                return "high"
            if value in {"xhigh", "max"}:
                return "max"
            return value
        if provider == "openai":
            if value == "max":
                return "xhigh"
            if value not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
                return "medium"
            return value
        return value

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                refusal = item.get("refusal")
                if isinstance(text, str) and text:
                    parts.append(text)
                elif isinstance(refusal, str) and refusal:
                    parts.append(refusal)
            return "".join(parts)
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text
        return str(content)

    @staticmethod
    def _reasoning_details_to_text(details: Any) -> str:
        if not isinstance(details, list):
            return ""
        parts: List[str] = []
        for item in details:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _stream_delta_piece(current: Any, prior_full: str) -> Tuple[str, str]:
        text = current if isinstance(current, str) else ""
        if not text:
            return "", prior_full
        if prior_full and text.startswith(prior_full):
            return text[len(prior_full):], text
        if not prior_full:
            return text, text
        return text, prior_full + text

    def _normalize_response_message(self, msg: Dict[str, Any], provider: str) -> Tuple[Dict[str, Any], str, str]:
        assistant_message: Dict[str, Any] = {
            "role": msg.get("role", "assistant"),
            "content": self._content_to_text(msg.get("content")),
        }
        if msg.get("tool_calls"):
            assistant_message["tool_calls"] = msg["tool_calls"]

        reasoning = ""
        if isinstance(msg.get("reasoning_content"), str) and msg.get("reasoning_content"):
            reasoning = msg["reasoning_content"]
            assistant_message["reasoning_content"] = reasoning

        if msg.get("reasoning_details"):
            assistant_message["reasoning_details"] = msg["reasoning_details"]
            if not reasoning:
                reasoning = self._reasoning_details_to_text(msg["reasoning_details"])
                if reasoning:
                    assistant_message["reasoning_content"] = reasoning

        return assistant_message, assistant_message["content"], reasoning

    def _chat_completions_url(self, stage: str) -> str:
        cfg = self.stage_config(stage)
        path = cfg["chat_completions_path"]
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.base_url}{path}"

    @staticmethod
    def _repair_tool_message_sequence(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop malformed tool-call groups before sending Chat Completions history.

        Providers require every ``role=tool`` message to answer a preceding
        assistant ``tool_calls`` item, and an assistant tool-call message must
        be followed immediately by all matching tool results.  Context
        compaction can cut either side of that group; sending the partial group
        makes providers reject the whole request.
        """
        repaired: List[Dict[str, Any]] = []
        idx = 0

        while idx < len(messages):
            msg = dict(messages[idx] or {})
            role = str(msg.get("role") or "user").strip().lower()
            if role == "assistant":
                tool_calls = msg.get("tool_calls") or []
                pending_tool_ids = [
                    str(call.get("id") or "")
                    for call in tool_calls
                    if isinstance(call, dict) and call.get("id")
                ]
                if not pending_tool_ids:
                    repaired.append(msg)
                    idx += 1
                    continue
                group: List[Dict[str, Any]] = [msg]
                remaining = set(pending_tool_ids)
                cursor = idx + 1
                while cursor < len(messages):
                    tool_msg = dict(messages[cursor] or {})
                    tool_role = str(tool_msg.get("role") or "user").strip().lower()
                    if tool_role != "tool":
                        break
                    tool_call_id = str(tool_msg.get("tool_call_id") or "")
                    if tool_call_id not in remaining:
                        break
                    group.append(tool_msg)
                    remaining.discard(tool_call_id)
                    cursor += 1
                    if not remaining:
                        break
                if not remaining:
                    repaired.extend(group)
                    idx = cursor
                else:
                    idx = cursor
                continue

            if role == "tool":
                idx += 1
                continue

            repaired.append(msg)
            idx += 1

        return repaired

    @staticmethod
    def _sanitize_messages_for_provider(provider: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize outbound message history for provider-specific chat constraints."""
        messages = ProviderClient._repair_tool_message_sequence(messages)
        normalized: List[Dict[str, Any]] = []
        initial_system_allowed = True
        system_consumed = False

        for index, message in enumerate(messages):
            msg = dict(message or {})
            role = str(msg.get("role") or "user").strip().lower()

            if provider == "minimax":
                if role == "system":
                    if initial_system_allowed and not system_consumed and index == 0:
                        system_consumed = True
                    else:
                        role = "user"
                if role == "assistant":
                    msg = {
                        "role": "assistant",
                        "content": msg.get("content", ""),
                    }
                    if msg["content"] is None:
                        msg["content"] = ""
                    if message.get("tool_calls"):
                        msg["tool_calls"] = message["tool_calls"]
                    normalized.append(msg)
                    continue
                elif role == "tool":
                    msg = {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id"),
                        "name": msg.get("name"),
                        "content": msg.get("content", ""),
                    }
                    if msg["content"] is None:
                        msg["content"] = ""
                    normalized.append({k: v for k, v in msg.items() if v is not None})
                    continue
                else:
                    msg = {
                        "role": role,
                        "content": msg.get("content", ""),
                    }
                    if msg["content"] is None:
                        msg["content"] = ""
                    normalized.append(msg)
                    continue

            normalized.append(msg)

        return normalized

    def build_request_body(
        self,
        *,
        stage: str,
        messages: List[Dict],
        stream: Optional[bool] = None,
        expect_json: bool = False,
    ) -> Dict[str, Any]:
        """Build a Chat Completions request body for a stage."""
        cfg = self.stage_config(stage)
        if stream is None:
            stream = cfg["stream"]
        provider = cfg["provider"]
        token_param = self._provider_token_param(provider, cfg.get("token_param", ""))
        normalized_messages = self._sanitize_messages_for_provider(provider, messages)
        body: Dict[str, Any] = {
            "model": cfg["model"],
            "messages": normalized_messages,
            "temperature": self._provider_temperature(provider, cfg["temperature"]),
            "stream": bool(stream),
            token_param: cfg["max_tokens"],
        }
        if provider == "deepseek":
            thinking_obj: Dict[str, Any] = {"type": "enabled" if cfg["thinking"] else "disabled"}
            if cfg["thinking"]:
                thinking_obj["reasoning_effort"] = self._provider_reasoning_effort(provider, cfg["effort"])
            body["thinking"] = thinking_obj
        elif provider == "openai":
            if cfg["thinking"]:
                body["reasoning_effort"] = self._provider_reasoning_effort(provider, cfg["effort"])
        elif provider == "minimax":
            body["thinking"] = {"type": "adaptive" if cfg["thinking"] else "disabled"}
            if cfg["thinking"]:
                body["thinking"]["reasoning_effort"] = self._provider_reasoning_effort(provider, cfg["effort"])
            # Keep MiniMax reasoning separated from the JSON protocol when
            # thinking is enabled. The thinking field above controls whether
            # reasoning is produced at all.
            if cfg.get("minimax_reasoning_split", True):
                body["reasoning_split"] = True
        if expect_json:
            body["response_format"] = {"type": "json_object"}
        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _schema_hint_required_keys(schema_hint: Optional[str]) -> List[str]:
        if not schema_hint:
            return []
        try:
            payload = json.loads(schema_hint)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        keys = payload.get("required")
        if isinstance(keys, list):
            return [str(key) for key in keys if str(key).strip()]
        if "required_schema" in payload and isinstance(payload["required_schema"], dict):
            return [str(key) for key in payload["required_schema"].keys() if str(key).strip()]
        return []

    @classmethod
    def _json_required_top_level_keys(cls, stage: str, schema_hint: Optional[str]) -> List[str]:
        normalized = cls._normalize_stage_name(stage)
        stage_keys: Dict[str, List[str]] = {
            "active_path_ablation_target_plan": ["ablation_questions", "questions", "items"],
            "dataset_diagnosis": ["diagnostics", "claims", "dataset_profile", "status"],
        }
        keys = list(stage_keys.get(normalized, []))
        for key in cls._schema_hint_required_keys(schema_hint):
            if key not in keys:
                keys.append(key)
        return keys

    def call_json(
        self,
        stage: str,
        round_num: int,
        messages: List[Dict],
        schema_hint: Optional[str] = None,
        stream: Optional[bool] = None,
        stage_label: Optional[str] = None,
        execution_label: Optional[str] = None,
        tool_context: Optional[Any] = None,
        require_all_top_level_keys: bool = False,
        timeout_sec_override: Optional[int] = None,
        stream_override: Optional[bool] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Dict:
        """Call the API and return parsed JSON."""
        # Import locally to avoid a runtime-state/API-client import cycle.
        from evocast.state.runtime.terminal_ui import wait_if_paused

        wait_if_paused(str(runtime_root()), self.task_id)
        self._ensure_api_key()
        effective_stage = stage_label or stage
        del tool_context
        required_top_level_keys = self._json_required_top_level_keys(stage, schema_hint)

        body = self.build_request_body(stage=stage, messages=messages, stream=stream_override, expect_json=True)
        if max_tokens_override is not None:
            token_param = self._provider_token_param(
                self.stage_config(stage)["provider"], self.stage_config(stage).get("token_param", "")
            )
            body[token_param] = max(1, int(max_tokens_override))
        _, assistant_message, content, reasoning, _ = self.request_message_with_retries(
            stage=effective_stage,
            round_num=round_num,
            body=body,
            timeout_sec=(
                max(1, int(timeout_sec_override))
                if timeout_sec_override is not None
                else self.stage_config(stage)["timeout_sec"]
            ),
            max_retries=self.stage_config(stage)["max_retries"],
            execution_label=execution_label,
        )
        content = self._ensure_non_empty_content(
            stage=stage,
            effective_stage=effective_stage,
            round_num=round_num,
            body=body,
            content=content,
            reasoning=reasoning,
            execution_label=execution_label,
        )
        try:
            parsed = parse_json_object(
                content,
                required_top_level_keys=required_top_level_keys,
                require_all_required_keys=require_all_top_level_keys,
            )
        except Exception as exc:
            parsed = self._retry_json_response_same_agent(
                stage=stage,
                effective_stage=effective_stage,
                round_num=round_num,
                messages=messages,
                original_content=content,
                original_error=exc,
                required_top_level_keys=required_top_level_keys,
                require_all_top_level_keys=require_all_top_level_keys,
                execution_label=execution_label,
            )
        parsed_path = self._api_log_dir / f"{self._request_label(round_num, effective_stage, execution_label)}.parsed.json"
        write_text(str(parsed_path), json.dumps(parsed, indent=2, ensure_ascii=False))
        return parsed

    def _retry_json_response_same_agent(
        self,
        *,
        stage: str,
        effective_stage: str,
        round_num: int,
        messages: List[Dict],
        original_content: str,
        original_error: Exception,
        required_top_level_keys: Optional[Iterable[str]] = None,
        require_all_top_level_keys: bool = False,
        execution_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retry the same agent after local JSON parsing fails.

        Keep JSON recovery syntax-level: parse_json_object already strips fences,
        extracts object slices, and applies local json_repair/control-char fixes.
        If that fails, ask the same stage to restate its own answer as valid JSON
        instead of asking a separate repair call to invent or complete semantics.
        """

        cfg = self.stage_config(stage)
        retry_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as one JSON object. "
                    "Return the same intended content as exactly one valid JSON object, with no markdown or commentary. "
                    "Do not add new fields that were not part of your intended answer. "
                    + (
                        (
                            "The top-level object must include all of these keys: "
                            if require_all_top_level_keys
                            else "The top-level object must include at least one of these keys: "
                        )
                        + ", ".join(str(key) for key in required_top_level_keys or [])
                        + ". "
                        if required_top_level_keys
                        else ""
                    )
                    + "\n\n"
                    + json.dumps(
                        {
                            "parse_error": str(original_error),
                            "previous_response_excerpt": str(original_content or "")[:8000],
                        },
                        ensure_ascii=False,
                    )
                ),
            },
        ]
        retry_body = self.build_request_body(stage=stage, messages=retry_messages, stream=False, expect_json=True)
        try:
            _, _, retry_content, retry_reasoning, _ = self.request_message_with_retries(
                stage=f"{effective_stage}_json_retry",
                round_num=round_num,
                body=retry_body,
                timeout_sec=cfg["timeout_sec"],
                max_retries=0,
                execution_label=execution_label,
            )
            retry_content = self._ensure_non_empty_content(
                stage=stage,
                effective_stage=f"{effective_stage}_json_retry",
                round_num=round_num,
                body=retry_body,
                content=retry_content,
                reasoning=retry_reasoning,
                execution_label=execution_label,
            )
            parsed = parse_json_object(
                retry_content,
                required_top_level_keys=required_top_level_keys,
                require_all_required_keys=require_all_top_level_keys,
            )
            write_text(
                str(self._api_log_dir / f"{self._request_label(round_num, effective_stage, execution_label)}.json_retry.parsed.json"),
                json.dumps(parsed, indent=2, ensure_ascii=False),
            )
            return parsed
        except Exception as retry_exc:
            raise ProviderAPIError(
                f"{stage}: JSON parse failed after local repair and same-agent retry. "
                f"See logs/api/{self._request_label(round_num, effective_stage, execution_label)}.content.txt"
            ) from retry_exc

    def _repair_json_response_with_model(self, **kwargs: Any) -> Dict[str, Any]:
        """Deprecated compatibility shim: semantic JSON repair is disabled."""

        return self._retry_json_response_same_agent(**kwargs)

    def call_text(
        self,
        stage: str,
        round_num: int,
        messages: List[Dict],
        stream: Optional[bool] = None,
        execution_label: Optional[str] = None,
        tool_context: Optional[Any] = None,
    ) -> str:
        """Call the API and return raw text."""
        from evocast.state.runtime.terminal_ui import wait_if_paused

        wait_if_paused(str(runtime_root()), self.task_id)
        self._ensure_api_key()
        del tool_context

        body = self.build_request_body(stage=stage, messages=messages, stream=stream, expect_json=False)
        _, _, content, _, _ = self.request_message_with_retries(
            stage=stage,
            round_num=round_num,
            body=body,
            timeout_sec=self.stage_config(stage)["timeout_sec"],
            max_retries=self.stage_config(stage)["max_retries"],
            execution_label=execution_label,
        )
        return content

    def _ensure_non_empty_content(
        self,
        *,
        stage: str,
        effective_stage: str,
        round_num: int,
        body: Dict[str, Any],
        content: str,
        reasoning: str,
        execution_label: Optional[str] = None,
    ) -> str:
        if content and content.strip():
            return content
        cfg = self.stage_config(stage)
        if reasoning and cfg["thinking"]:
            print(f"[api:{stage}] reasoning exhausted output; retrying with thinking=false")
            retry_body = dict(body)
            retry_body["thinking"] = {"type": "disabled"}
            retry_body["stream"] = False
            retry_body.pop("stream_options", None)
            _, _, content, _, _ = self.request_message_with_retries(
                stage=effective_stage,
                round_num=round_num,
                body=retry_body,
                timeout_sec=cfg["timeout_sec"],
                max_retries=0,
                execution_label=execution_label,
            )
        if not content or not content.strip():
            raise ProviderAPIError(
                f"{stage}: empty API content. See logs/api/{self._request_label(round_num, effective_stage, execution_label)}.content.txt"
            )
        return content

    def request_message_with_retries(
        self,
        *,
        stage: str,
        round_num: int,
        body: Dict[str, Any],
        timeout_sec: int,
        max_retries: int,
        execution_label: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], str, str, Optional[Dict[str, Any]]]:
        """Send a request and return assistant message plus rendered content."""
        last_error = None
        logical_call_id = uuid.uuid4().hex
        for attempt in range(max_retries + 1):
            try:
                return self._request(
                    stage,
                    round_num,
                    body,
                    timeout_sec,
                    attempt,
                    execution_label=execution_label,
                    logical_call_id=logical_call_id,
                )
            except urllib.error.HTTPError as exc:
                status = exc.code
                error_body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {status}: {error_body[:500]}"
                if status not in TRANSIENT_HTTP_STATUSES or attempt >= max_retries:
                    raise ProviderAPIError(last_error) from exc
            except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionError) as exc:
                last_error = repr(exc)
                if attempt >= max_retries:
                    raise ProviderAPIError(last_error) from exc
            sleep_s = min(60, (2 ** attempt) + random.random())
            safe_api_print(f"[api:{stage}] retry {attempt + 1}/{max_retries} after {sleep_s:.1f}s")
            time.sleep(sleep_s)
        raise ProviderAPIError(last_error or "unknown API failure")

    def _request_label(self, round_num: int, stage: str, execution_label: Optional[str] = None) -> str:
        prefix = str(execution_label or "").strip()
        if not prefix:
            prefix = f"Research{round_num:03d}" if int(round_num or 0) > 0 else f"r{round_num:03d}"
        return f"{prefix}_{stage}"

    def _request(
        self,
        stage,
        round_num,
        body,
        timeout_sec,
        attempt,
        execution_label: Optional[str] = None,
        logical_call_id: Optional[str] = None,
    ):
        url = self._chat_completions_url(stage)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        os.makedirs(str(self._api_log_dir), exist_ok=True)
        logical_call_id = logical_call_id or uuid.uuid4().hex
        request_id = uuid.uuid4().hex
        prefix = self._api_log_dir / (
            f"{self._request_label(round_num, stage, execution_label)}__api_{request_id}_attempt{int(attempt) + 1:02d}"
        )

        safe_api_print(f"[api:{stage}] model={body['model']} stream={body.get('stream')} attempt={attempt}")

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage = None
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": ""}
        provider = self.stage_config(stage).get("provider", "deepseek")
        stream_finished = None
        json_completed_early = False
        reasoning_stream_open = False
        started_at = datetime.now().isoformat()
        started = time.monotonic()

        request_deadline = time.monotonic() + max(1, int(timeout_sec))
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if body.get("stream"):
                    stream_finished = False
                    last_progress = time.time()
                    tool_calls: Dict[int, Dict[str, Any]] = {}
                    content_full = ""
                    reasoning_full = ""
                    with open(str(prefix) + ".content.txt", "w", encoding="utf-8") as cf, open(
                        str(prefix) + ".reasoning.txt", "w", encoding="utf-8"
                    ) as rf:
                        for raw_line in resp:
                            # ``urlopen(timeout=...)`` only limits an idle
                            # socket read. A streaming model can keep sending
                            # small reasoning chunks indefinitely, so enforce
                            # the configured request budget across the whole
                            # turn as well. This matters especially for the
                            # Builder, whose next action must be prompt and
                            # resumable rather than a multi-minute monologue.
                            if time.monotonic() >= request_deadline:
                                raise TimeoutError(
                                    f"{stage}: total streaming request budget of {timeout_sec}s exceeded"
                                )
                            line = raw_line.decode("utf-8", errors="replace").strip()
                            if not line or line.startswith(":") or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                stream_finished = True
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("usage"):
                                usage = chunk.get("usage")
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            finish_reason = choices[0].get("finish_reason")
                            if finish_reason:
                                stream_finished = True
                            rd = delta.get("reasoning_content") or ""
                            if not rd and delta.get("reasoning_details"):
                                latest_reasoning = self._reasoning_details_to_text(delta.get("reasoning_details"))
                                rd, reasoning_full = self._stream_delta_piece(latest_reasoning, reasoning_full)
                            cd = delta.get("content") or ""
                            if provider == "minimax":
                                cd, content_full = self._stream_delta_piece(cd, content_full)
                            else:
                                _, content_full = self._stream_delta_piece(cd, content_full)
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", len(tool_calls))
                                existing = tool_calls.setdefault(
                                    idx,
                                    {
                                        "id": tc.get("id"),
                                        "type": tc.get("type", "function"),
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc.get("id"):
                                    existing["id"] = tc.get("id")
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    existing["function"]["name"] = fn.get("name")
                                if fn.get("arguments"):
                                    existing["function"]["arguments"] += fn.get("arguments")
                            if rd:
                                reasoning_parts.append(rd)
                                rf.write(rd)
                                rf.flush()
                                if not reasoning_stream_open:
                                    safe_api_print(f"[reasoning:{stage}] ", end="", flush=True)
                                    reasoning_stream_open = True
                                safe_api_print(rd, end="", flush=True)
                                now = time.time()
                                if now - last_progress >= 5:
                                    safe_api_print(f"\n[api:{stage}] reasoning... {len(''.join(reasoning_parts))} chars", flush=True)
                                    last_progress = now
                            if cd:
                                if reasoning_stream_open:
                                    safe_api_print("\n[/reasoning]", flush=True)
                                    reasoning_stream_open = False
                                content_parts.append(cd)
                                cf.write(cd)
                                cf.flush()
                                safe_api_print(cd, end="", flush=True)
                                last_progress = time.time()
                                # Providers occasionally emit a complete
                                # JSON object but keep the SSE channel open
                                # (or delay [DONE]) while producing trailing
                                # hidden text. For a JSON protocol turn the
                                # valid object is the completion boundary;
                                # waiting discards usable work as a timeout.
                                if body.get("response_format", {}).get("type") == "json_object":
                                    try:
                                        parse_json_object("".join(content_parts))
                                    except Exception:
                                        pass
                                    else:
                                        json_completed_early = True
                                        stream_finished = True
                                        break
                    if reasoning_stream_open:
                        safe_api_print("\n[/reasoning]", flush=True)
                    safe_api_print()
                    if not stream_finished:
                        raise ProviderAPIError(
                            f"{stage}: stream ended before [DONE]; partial content logged to "
                            f"logs/api/{self._request_label(round_num, stage, execution_label)}.content.txt"
                        )
                    if tool_calls:
                        assistant_message["tool_calls"] = [tool_calls[idx] for idx in sorted(tool_calls)]
                else:
                    stream_finished = True
                    text = resp.read().decode("utf-8", errors="replace")
                    result = json.loads(text)
                    usage = result.get("usage")
                    msg = self._extract_message_payload(result)
                    assistant_message, content_text, reasoning_text = self._normalize_response_message(msg, provider)
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                        safe_api_print(f"[reasoning:{stage}] {reasoning_text}\n[/reasoning]", flush=True)
                    content_parts.append(content_text)
                    if content_text:
                        safe_api_print(content_text, flush=True)
                    write_text(str(prefix) + ".content.txt", "".join(content_parts))
                    write_text(str(prefix) + ".reasoning.txt", "".join(reasoning_parts))
        except Exception as exc:
            elapsed_seconds = round(time.monotonic() - started, 6)
            write_text(
                str(prefix) + ".error.json",
                json.dumps(
                    _api_error_payload(
                        stage=stage,
                        round_num=round_num,
                        attempt=attempt,
                        provider=provider,
                        model=body.get("model"),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        content_chars=len("".join(content_parts)),
                        reasoning_chars=len("".join(reasoning_parts)),
                        stream_finished=stream_finished,
                    ),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
            )
            append_cost_event(
                self.base_dir,
                self.task_id,
                {
                    "event_id": request_id,
                    "logical_call_id": logical_call_id,
                    "transport_attempt": int(attempt) + 1,
                    "kind": "llm_api",
                    "stage": stage,
                    "round": round_num,
                    "round_id": f"Research{int(round_num):03d}" if int(round_num or 0) > 0 else "",
                    "attempt_id": str(execution_label or ""),
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now().isoformat(),
                    "elapsed_seconds": elapsed_seconds,
                    "provider": provider,
                    "model": body.get("model"),
                    "usage": usage or {},
                    "usage_status": "returned" if usage else "unavailable",
                    "cost": estimate_llm_cost(self.config, str(body.get("model") or ""), usage),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "api_log_prefix": str(prefix),
                },
            )
            raise

        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        assistant_message["content"] = content
        if reasoning:
            assistant_message["reasoning_content"] = reasoning

        elapsed_seconds = round(time.monotonic() - started, 6)
        cost = estimate_llm_cost(self.config, str(body.get("model") or ""), usage)
        append_cost_event(
            self.base_dir,
            self.task_id,
            {
                "event_id": request_id,
                "logical_call_id": logical_call_id,
                "transport_attempt": int(attempt) + 1,
                "kind": "llm_api",
                "stage": stage,
                "round": round_num,
                "round_id": f"Research{int(round_num):03d}" if int(round_num or 0) > 0 else "",
                "attempt_id": str(execution_label or ""),
                "status": "success",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "provider": provider,
                "model": body.get("model"),
                "usage": usage or {},
                "usage_status": "returned" if usage else "unavailable",
                "cost": cost,
                "api_log_prefix": str(prefix),
            },
        )
        if assistant_message.get("tool_calls"):
            write_text(
                str(prefix) + ".message.json",
                json.dumps(assistant_message, indent=2, ensure_ascii=False),
            )
        else:
            remove_file_if_exists(str(prefix) + ".message.json")
        return None, assistant_message, content, reasoning, usage
