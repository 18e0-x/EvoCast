"""Fixed-panel terminal console and persistent raw transcript for EVOCAST."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import textwrap
import threading
import time
import unicodedata
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from evocast.domain.atomic_io import atomic_write_json
from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness.rounds import TERMINAL_ROUND_STATUSES, list_rounds
from evocast.state.runtime.store import load_runtime_state
from evocast.state.token_usage import build_token_usage_summary
from evocast.state.domain_store import load_task_config


def terminal_control_path(base_dir: str, task_id: str) -> Path:
    return task_knowledge_dir(base_dir, task_id) / "terminal_control.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, type(default)) else default
    except Exception:
        return default


def pause_requested(base_dir: str, task_id: str) -> bool:
    return bool(_read_json(terminal_control_path(base_dir, task_id), {}).get("paused"))


def set_pause_requested(base_dir: str, task_id: str, paused: bool) -> None:
    path = terminal_control_path(base_dir, task_id)
    atomic_write_json(path, {**_read_json(path, {}), "paused": bool(paused), "updated_at": datetime.now().isoformat(), "source": "terminal_ui"}, ensure_ascii=False)


def wait_if_paused(base_dir: str, task_id: str) -> None:
    while pause_requested(base_dir, task_id):
        time.sleep(0.2)


class TerminalLog:
    """Append-only transcript with a fast live tail and disk-backed history."""

    def __init__(self, base_dir: str, task_id: str, *, max_tail_lines: int = 40000) -> None:
        root = task_knowledge_dir(base_dir, task_id)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "terminal_raw.log"
        self._tail: deque[str] = deque(maxlen=max_tail_lines)
        self._lock = threading.Lock()
        self._handle = self.path.open("a", encoding="utf-8", errors="replace")
        if self.path.exists():
            try:
                self._tail.extend(self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_tail_lines:])
            except OSError:
                pass

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._handle.write(text)
            self._handle.flush()
            self._tail.extend(text.rstrip("\n").splitlines() or [""])

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def tail(self) -> list[str]:
        with self._lock:
            return list(self._tail)

    def snapshot(self) -> list[str]:
        """Return the current in-memory tail for diagnostics and tests."""
        return self.tail()

    def full_history(self) -> list[str]:
        with self._lock:
            self._handle.flush()
            try:
                return self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return list(self._tail)


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def _duration(seconds: float | int | None) -> str:
    seconds = max(0, int(seconds or 0))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1 for char in text)


def _fit(text: Any, width: int) -> str:
    value, used = " ".join(str(text or "--").split()), 0
    out: list[str] = []
    for char in value:
        char_width = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if used + char_width > max(1, width - 1):
            return "".join(out) + "…"
        out.append(char)
        used += char_width
    return "".join(out)


def _column(text: Any, width: int) -> str:
    value = _fit(text, width)
    return value + " " * max(0, width - _display_width(value))


def _metric(record: dict[str, Any], objective: str) -> str:
    for source in (record.get("metrics"), record.get("candidate_metrics"), (record.get("gate_event") or {}).get("candidate_metrics")):
        if isinstance(source, dict) and source.get(objective) is not None:
            value = source[objective]
            return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)
    return "--"


def _promoted_to_current_best(record: dict[str, Any]) -> bool:
    seed_eval = record.get("seed_eval") if isinstance(record.get("seed_eval"), dict) else {}
    if seed_eval.get("promoted_to_current_best") is True:
        return True
    if record.get("promoted_to_current_best") is True or record.get("is_current_best") is True:
        return True
    promotion = record.get("promotion_decision") if isinstance(record.get("promotion_decision"), dict) else {}
    seed_promotion = seed_eval.get("promotion_decision") if isinstance(seed_eval.get("promotion_decision"), dict) else {}
    decisions = [
        promotion.get("decision"),
        seed_promotion.get("decision"),
        record.get("promotion_decision") if not isinstance(record.get("promotion_decision"), dict) else None,
    ]
    return any(str(value or "").strip().lower() in {"promote", "promoted"} for value in decisions)


def _research_result_state(record: dict[str, Any], objective: str) -> str:
    status = str(record.get("status") or "").strip().lower()
    if str(record.get("round_scope") or "").strip().lower() == "baseline_diagnosis" and status in {"success", "failed_ablation_round"}:
        return "success" if _promoted_to_current_best(record) else "failed"
    if status not in TERMINAL_ROUND_STATUSES:
        return "running"
    return "success" if _promoted_to_current_best(record) else "failed"


def _model_label(value: Any) -> str:
    text = " ".join(str(value or "--").split())
    if "." in text:
        parts = [part for part in text.split(".") if part]
        if parts:
            text = parts[-1]
    return _short_terminal_title(text, max_words=1) or "--"


def _extract_model_from_text(text: str) -> tuple[str, str]:
    cleaned = " ".join(str(text or "").split())
    for token in cleaned.split():
        bare = token.strip("()[]{}:,;")
        if "." not in bare:
            continue
        parts = [part for part in bare.split(".") if part]
        if not parts:
            continue
        model = _model_label(parts[-1])
        without = " ".join(part for part in cleaned.split() if part != token)
        return model, without
    return "", cleaned


def _record_duration(record: dict[str, Any], now: datetime) -> str:
    start = _parse_time(record.get("created_at"))
    if not start:
        return "--"
    status = str(record.get("status") or "").strip()
    if status not in TERMINAL_ROUND_STATUSES:
        return _duration((now - start).total_seconds())
    end = _parse_time(record.get("closed_at") or record.get("updated_at"))
    if end:
        return _duration((end - start).total_seconds())
    return "--"


def _artifact_duration(path: Path, now: datetime) -> str:
    """Best available duration for older artifacts that lack explicit timing."""
    try:
        start = datetime.fromtimestamp((path.parent / "ablation_task.json").stat().st_mtime)
    except OSError:
        try:
            start = datetime.fromtimestamp(path.stat().st_ctime)
        except OSError:
            return "--"
    try:
        end = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        end = now
    return _duration((end - start).total_seconds())


def _run_duration(root: Path, run_id: Any) -> str:
    if not run_id:
        return "--"
    result = _read_json(root / "run_records" / f"{run_id}.json", {})
    for source in (result, result.get("run_result") if isinstance(result, dict) else {}):
        if isinstance(source, dict) and source.get("elapsed_seconds") is not None:
            return _duration(source.get("elapsed_seconds"))
    return "--"


def _research_display_text(record: dict[str, Any]) -> str:
    proposal = record.get("proposal") if isinstance(record.get("proposal"), dict) else {}
    contract = proposal.get("build_contract") if isinstance(proposal.get("build_contract"), dict) else {}
    idea = (
        record.get("display_idea")
        or record.get("idea_summary")
        or proposal.get("innovation_name")
        or record.get("research_intent")
        or record.get("research_direction")
        or record.get("hypothesis")
        or contract.get("semantic_goal")
        or contract.get("hypothesis")
    )
    model_source = record.get("model_family") or contract.get("target_model")
    model = _model_label(model_source) if str(model_source or "").strip() else ""
    if idea and model and str(model) not in str(idea):
        return _compact_terminal_idea(str(idea), str(model))
    return _compact_terminal_idea(str(idea or model or proposal.get("fit_point") or record.get("fit_point") or "--"))


def _compact_terminal_idea(text: str, model: str = "") -> str:
    cleaned = " ".join(str(text or "").split())
    model = _model_label(model) if model else ""
    if not model and "|" in cleaned:
        left, right = cleaned.split("|", 1)
        model = _model_label(left)
        cleaned = right.strip()
    if not model:
        model, cleaned = _extract_model_from_text(cleaned)
    if model:
        title = _short_terminal_title(cleaned, max_words=7)
        return model if not title else f"{model} | {title}"
    return _short_terminal_title(cleaned, max_words=8) or "--"


def _short_terminal_title(text: str, *, max_words: int) -> str:
    cleaned = " ".join(str(text or "").split())
    words = cleaned.split()
    if len(words) > 1:
        return " ".join(words[:max(1, max_words)])
    if any("\u4e00" <= char <= "\u9fff" for char in cleaned):
        return cleaned[:16]
    return cleaned[:48]


def _counts_toward_formal_research(record: dict[str, Any]) -> bool:
    explicit = record.get("counts_toward_research_budget")
    if explicit is not None:
        return bool(explicit)
    return str(record.get("round_scope") or "research").strip().lower() == "research"


def _round_kind(record: dict[str, Any]) -> str:
    if _counts_toward_formal_research(record):
        return "Research"
    scope = str(record.get("round_scope") or "").strip().lower()
    if scope == "baseline_diagnosis":
        return "Ablation"
    return "Diagnostic"


class TerminalUI:
    """A two-pane terminal: upper dashboard and lower independently scrolled log."""

    RESET, DIM, BOLD = "\x1b[0m", "\x1b[2m", "\x1b[1m"
    RED, GREEN, YELLOW, BLUE = "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[36m"

    def __init__(self, base_dir: str, task_id: str) -> None:
        self.base_dir, self.task_id = base_dir, task_id
        self.log = TerminalLog(base_dir, task_id)
        self._stdin: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._enabled = self._closed = False
        self._terminal_fd: int | None = None
        self._terminal_attrs: Any = None
        self._signal_handlers: dict[int, Any] = {}
        self._follow, self._thinking_expanded = True, False
        self._focus_latest_thinking = False
        self._scroll = 0
        self._frozen_history: list[str] | None = None
        self._frozen_line_count = 0
        self._last_token_refresh, self._tokens = 0.0, 0
        self._render_lock, self._wake = threading.Lock(), threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"

    def isatty(self) -> bool:
        return bool(self._stdout and self._stdout.isatty())

    def fileno(self) -> int:
        return self._stdout.fileno() if self._stdout else -1

    def write(self, text: str) -> int:
        text = str(text)
        self.log.append(text)
        self._wake.set()
        if not self._enabled and self._stdout:
            self._stdout.write(text)
            self._stdout.flush()
        return len(text)

    def flush(self) -> None:
        self.log.flush()
        if not self._enabled and self._stdout:
            self._stdout.flush()

    def start(self) -> "TerminalUI":
        self._stdin = sys.stdin
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._enabled = self._can_use_dashboard()
        sys.stdout, sys.stderr = self, self  # type: ignore[assignment]
        if self._enabled:
            self._prepare_terminal()
        if self._enabled:
            self._thread = threading.Thread(target=self._event_loop, name="evocast-terminal-ui", daemon=True)
            self._thread.start()
        else:
            self.write(f"[terminal] raw transcript: {self.log.path}\n")
        return self

    def stop(self) -> None:
        # Paint once after the final report/status lines have been written so
        # task completion leaves a readable final console rather than a stale frame.
        if self._enabled:
            self.render()
        self._closed = True
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if sys.stdout is self and self._stdout:
            sys.stdout = self._stdout
        if sys.stderr is self and self._stderr:
            sys.stderr = self._stderr
        self._restore_terminal()
        self.log.close()
        if self._enabled and self._stdout:
            self._stdout.write("\x1b[?25h")
            self._stdout.flush()

    def detach(self) -> None:
        self._enabled = False
        self._restore_terminal()
        if self._stdout:
            view = self._task_view()
            message = f"EVOCAST terminal detached. Full log: {self.log.path}\n" if view["english"] else f"EVOCAST 控制台已退出。完整日志：{self.log.path}\n"
            self._stdout.write("\x1b[?25h\n" + message)
            self._stdout.flush()

    def _can_use_dashboard(self) -> bool:
        if not (self._stdin and self._stdout and self._stdin.isatty() and self._stdout.isatty()):
            return False
        return str(os.environ.get("TERM") or "").strip().lower() != "dumb"

    def _prepare_terminal(self) -> None:
        if os.name == "nt":
            self._enable_windows_vt()
            return
        if os.name != "posix" or not self._stdin:
            return
        try:
            import termios
            import tty

            fd = self._stdin.fileno()
            self._terminal_fd = fd
            self._terminal_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            self._install_terminal_signal_handlers()
        except Exception:
            self._restore_terminal()
            self._enabled = False

    def _restore_terminal(self) -> None:
        if os.name == "posix" and self._terminal_fd is not None and self._terminal_attrs is not None:
            try:
                import termios

                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_attrs)
            except Exception:
                pass
        self._terminal_fd, self._terminal_attrs = None, None
        self._restore_terminal_signal_handlers()

    def _install_terminal_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)):
            if sig is None or sig in self._signal_handlers:
                continue
            try:
                previous = signal.getsignal(sig)
                self._signal_handlers[sig] = previous

                def _handler(signum: int, frame: Any, *, previous_handler: Any = previous) -> None:
                    self._restore_terminal()
                    if callable(previous_handler):
                        previous_handler(signum, frame)
                    elif previous_handler == signal.SIG_DFL:
                        if signum == signal.SIGINT:
                            raise KeyboardInterrupt
                        raise SystemExit(128 + signum)

                signal.signal(sig, _handler)
            except Exception:
                self._signal_handlers.pop(sig, None)

    def _restore_terminal_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, previous in list(self._signal_handlers.items()):
            try:
                if signal.getsignal(sig) is not previous:
                    signal.signal(sig, previous)
            except Exception:
                pass
        self._signal_handlers.clear()

    def _enable_windows_vt(self) -> None:
        try:
            import ctypes
            handle, mode = ctypes.windll.kernel32.GetStdHandle(-11), ctypes.c_uint()
            if ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                ctypes.windll.kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    def _event_loop(self) -> None:
        while not self._closed and self._enabled:
            self._read_keys()
            self.render()
            self._wake.wait(0.2)
            self._wake.clear()

    def _read_keys(self) -> None:
        if os.name == "nt":
            self._read_windows_keys()
        elif os.name == "posix":
            self._read_posix_keys()

    def _handle_key(self, key: str) -> bool:
        if key in {"p", "P"}:
            set_pause_requested(self.base_dir, self.task_id, not pause_requested(self.base_dir, self.task_id))
        elif key in {"t", "T"}:
            self._thinking_expanded = not self._thinking_expanded
            if self._thinking_expanded:
                # T means "show the newest available reasoning", not
                # "keep an old snapshot forever".  Refresh before
                # locating the newest block, then enter browse mode.
                self._frozen_history = self.log.full_history()
                self._frozen_line_count = len(self._frozen_history)
                self._follow = False
                self._focus_latest_thinking = True
        elif key in {"q", "Q"}:
            self.detach()
            return True
        elif key in {"up", "\x1b[A"}:
            self._browse(1)
        elif key in {"down", "\x1b[B"}:
            self._browse(-1)
        elif key in {"page_up", "\x1b[5~"}:
            self._browse(12)
        elif key in {"page_down", "\x1b[6~"}:
            self._browse(-12)
        elif key in {"end", "\x1b[F", "\x1b[4~"}:
            self._resume_follow()
        return False

    def _read_windows_keys(self) -> None:
        try:
            import msvcrt
            while msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in {"\x00", "\xe0"}:
                    code = msvcrt.getwch()
                    if code == "H": self._handle_key("up")
                    elif code == "P": self._handle_key("down")
                    elif code == "I": self._handle_key("page_up")
                    elif code == "Q": self._handle_key("page_down")
                    elif code == "O": self._handle_key("end")
                elif key == "\x1b":
                    # Windows Terminal / modern PowerShell emits VT input
                    # (ESC [ A) instead of the legacy msvcrt scan codes.
                    sequence = ""
                    while msvcrt.kbhit() and len(sequence) < 4:
                        sequence += msvcrt.getwch()
                        if sequence[-1].isalpha() or sequence.endswith("~"):
                            break
                    self._handle_key("\x1b" + sequence)
                elif self._handle_key(key):
                    return
        except Exception:
            return

    def _read_posix_keys(self) -> None:
        if self._terminal_fd is None:
            return
        try:
            import select

            while True:
                readable, _, _ = select.select([self._terminal_fd], [], [], 0)
                if not readable:
                    return
                key = os.read(self._terminal_fd, 1).decode(errors="ignore")
                if key == "\x1b":
                    sequence = key + self._read_posix_escape_tail()
                    if self._handle_key(sequence):
                        return
                elif self._handle_key(key):
                    return
        except Exception:
            return

    def _read_posix_escape_tail(self) -> str:
        if self._terminal_fd is None:
            return ""
        try:
            import select

            sequence = ""
            deadline = time.monotonic() + 0.03
            while len(sequence) < 8 and time.monotonic() < deadline:
                timeout = max(0.0, deadline - time.monotonic())
                readable, _, _ = select.select([self._terminal_fd], [], [], timeout)
                if not readable:
                    break
                char = os.read(self._terminal_fd, 1).decode(errors="ignore")
                sequence += char
                if char.isalpha() or char == "~":
                    break
            return sequence
        except Exception:
            return ""

    def _freeze_history(self) -> None:
        if self._frozen_history is None:
            self._frozen_history = self.log.full_history()
            self._frozen_line_count = len(self._frozen_history)

    def _resume_follow(self) -> None:
        self._follow, self._scroll, self._frozen_history, self._frozen_line_count = True, 0, None, 0

    def _browse(self, delta: int) -> None:
        self._follow = False
        self._freeze_history()
        self._scroll = max(0, self._scroll + delta)

    def _task_view(self) -> dict[str, Any]:
        root = task_knowledge_dir(self.base_dir, self.task_id)
        config = load_task_config(self.base_dir, self.task_id)
        state, records, now = load_runtime_state(self.base_dir, self.task_id, auto_migrate=False), list_rounds(self.base_dir, self.task_id), datetime.now()
        research_state = _read_json(root / "research_state.json", {})
        action = research_state.get("current_action") if isinstance(research_state, dict) else None
        action = action if isinstance(action, dict) and str(action.get("status") or "").lower() not in {"", "completed", "failed", "rejected"} else None
        runtime_research = getattr(state, "research", None)
        active_id = (getattr(runtime_research, "current_research_id", "") if runtime_research else "") or ""
        if not active_id:
            candidates = sorted((path.name for path in (root / "rounds").glob("Research*") if path.is_dir()), reverse=True) if (root / "rounds").exists() else []
            active_id = candidates[0] if candidates else ""
        active_research = None
        if action:
            active_created_at = (
                action.get("created_at")
                or action.get("started_at")
                or research_state.get("created_at")
                or research_state.get("updated_at")
            )
            active_research = {
                "research_id": active_id or "Research001",
                "status": "running",
                "phase": "proposal",
                "research_intent": action.get("research_question") or action.get("hypothesis") or "--",
                "fit_point": action.get("primary_fit_point") or action.get("allowed_component_path") or "--",
                "created_at": active_created_at,
                "updated_at": research_state.get("updated_at"),
            }
        if time.monotonic() - self._last_token_refresh > 2.0:
            try: self._tokens = int((build_token_usage_summary(self.task_id, self.base_dir).get("totals") or {}).get("total_tokens") or 0)
            except Exception: pass
            self._last_token_refresh = time.monotonic()
        started = _parse_time(state.created_at) or _parse_time(config.get("created_at")) or now
        return {"root": root, "config": config, "state": state, "records": records, "active_research": active_research, "now": now, "objective": str(config.get("objective_metric") or state.objective_metric or "mse"), "elapsed": _duration((now - started).total_seconds()), "tokens": self._tokens, "english": str(config.get("language") or "zh").lower().startswith("en")}

    @staticmethod
    def _t(view: dict[str, Any], zh: str, en: str) -> str:
        return en if view["english"] else zh

    def _paint(self, text: str, colour: str) -> str:
        return f"{colour}{text}{self.RESET}" if text else text

    def _status_colour(self, status: str) -> str:
        return {
            "done": self.GREEN,
            "success": self.GREEN,
            "rejected": self.YELLOW,
            "running": self.YELLOW,
            "failed": self.RED,
        }.get(str(status or ""), "")

    def _metric_colour(self, metric: Any) -> str:
        return self.DIM if str(metric or "").strip() == "--" else ""

    def _status_text(self, view: dict[str, Any], value: Any) -> str:
        raw = str(value or "--")
        if view["english"]:
            return raw
        return {
            "running": "运行中", "completed": "已完成", "failed": "失败",
            "research": "研究", "baseline_search": "baseline 寻优",
            "baseline_diagnosis": "baseline 诊断", "research_round": "研究轮",
            "proposal": "方案生成", "design": "方案设计", "build": "构建变体",
            "verify": "验证", "experiment": "实验执行",
        }.get(raw.lower(), raw)

    def _baseline_duration(self, view: dict[str, Any]) -> str:
        summary = dict(getattr(view["state"], "baseline_search_progress", {}) or {})
        if summary.get("elapsed_seconds") is not None:
            return _duration(summary.get("elapsed_seconds"))
        values = [item.get("elapsed_seconds") for item in list(summary.get("run_results") or []) if isinstance(item, dict)]
        total = sum(float(value or 0) for value in values)
        if total:
            return _duration(total)
        # Old tasks did not persist search elapsed time.  Their archived trial
        # journal is the durable source for baseline start/end timestamps.
        journal = view["root"] / "archived_trial_journal.jsonl"
        spans: list[float] = []
        if journal.exists():
            for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
                try: node = json.loads(line)
                except Exception: continue
                if not isinstance(node, dict) or str(node.get("action_type")) != "baseline": continue
                start, end = _parse_time(node.get("created_at")), _parse_time(node.get("completed_at"))
                if start and end: spans.append((end - start).total_seconds())
        return _duration(sum(spans)) if spans else "--"

    def _ablation_rows(self, view: dict[str, Any]) -> list[dict[str, Any]]:
        objective, now = view["objective"], view["now"]
        rows: list[dict[str, Any]] = []
        # RoundStore is the authority for all round facts.  Legacy ablation
        # files are consumed only by the one-time migration in domain_store;
        # consulting them here would make the UI contradict a resumed task.
        for item in view["records"]:
            if _counts_toward_formal_research(item):
                continue
            rows.append({
                "id": item.get("research_id") or item.get("ablation_id") or f"Ablation{int(item.get('round_id') or 0):03d}",
                "kind": "Ablation",
                "what": item.get("mechanism_name") or item.get("target_name") or item.get("target_id") or _research_display_text(item),
                "metric": _metric(item, objective),
                "duration": _record_duration(item, now),
                "status": _research_result_state(item, objective),
            })
        return rows

    def _result_rows(self, view: dict[str, Any]) -> list[dict[str, Any]]:
        objective, state = view["objective"], view["state"]
        rows: list[dict[str, Any]] = []
        baseline = getattr(state, "baseline", None)
        if baseline and getattr(baseline, "candidate_id", ""):
            rows.append({"id": "Baseline", "kind": "Baseline", "what": _model_label(getattr(baseline, "display_name", "--")), "metric": _metric({"metrics": getattr(baseline, "metrics", {}) or {}}, objective), "duration": self._baseline_duration(view), "status": "done"})
        rows.extend(self._ablation_rows(view))
        for record in view["records"]:
            if not _counts_toward_formal_research(record):
                continue
            rows.append({"id": record.get("research_id") or f"Research{int(record.get('round_id') or 0):03d}", "kind": _round_kind(record), "what": _research_display_text(record), "metric": _metric(record, objective), "duration": _record_duration(record, view["now"]), "status": _research_result_state(record, objective)})
        active = view.get("active_research")
        row_ids = {str(row.get("id")) for row in rows}
        if active and str(active.get("research_id")) not in row_ids:
            rows.append({"id": active["research_id"], "kind": "Research", "what": _research_display_text(active), "metric": "--", "duration": _record_duration(active, view["now"]), "status": "running"})
        return rows

    def _top_lines(self, view: dict[str, Any], height: int, width: int) -> list[str]:
        state, records, objective = view["state"], view["records"], view["objective"]
        model = _model_label(getattr(state.baseline, "display_name", "") or view["config"].get("baseline_model") or "--")
        current = next((record for record in reversed(records) if str(record.get("status")) not in TERMINAL_ROUND_STATUSES), None) or view.get("active_research")
        current_name = ((current or {}).get("proposal") or {}).get("innovation_name") or ((current or {}).get("research_intent")) or "--"
        current_point = ((current or {}).get("proposal") or {}).get("fit_point") or ((current or {}).get("fit_point")) or "--"
        max_rounds = int(view["config"].get("max_rounds") or 0)
        formal_records = [record for record in records if _counts_toward_formal_research(record)]
        diagnostic_records = [record for record in records if not _counts_toward_formal_research(record)]
        closed = sum(1 for record in formal_records if str(record.get("status")) in TERMINAL_ROUND_STATUSES)
        diagnostic_closed = sum(1 for record in diagnostic_records if str(record.get("status")) in TERMINAL_ROUND_STATUSES)
        formal_current = next((record for record in reversed(formal_records) if str(record.get("status")) not in TERMINAL_ROUND_STATUSES), None)
        paused = pause_requested(self.base_dir, self.task_id)
        paused_text = self._t(view, "已暂停：等待当前调用结束", "PAUSED: waiting for current call") if paused else self._t(view, "运行中", "running")
        best = getattr(state.current_best, "metrics", {}) or getattr(state.baseline, "metrics", {}) or {}
        best_name = _model_label(getattr(state.current_best, "display_name", "") or getattr(state.baseline, "display_name", "") or "--")
        best_metric = _metric({"metrics": best if isinstance(best, dict) else {}}, objective)
        progress_parts = [f"[Baseline {'✓' if getattr(state.baseline, 'candidate_id', '') else '○'}]"]
        if diagnostic_records:
            progress_parts.append(
                f"[{self._t(view, '诊断消融', 'Diagnostic Ablation')} {diagnostic_closed}/{len(diagnostic_records)}]"
            )
        progress_parts.append(
            f"[{self._t(view, '正式 Research', 'Formal Research')} {closed}/{max_rounds or '?'} {'●' if formal_current else '○'}]"
        )
        lines = [
            self.BOLD + self._t(view, "EVOCAST · 实验控制台", "EVOCAST · Experiment Console") + self.RESET,
            _fit(f"{self._t(view, '任务', 'Task')} {self.task_id}  |  {self._t(view, '模型', 'Model')} {model}  |  {self._t(view, '目标', 'Objective')} {objective}↓", width),
            self.DIM + "─" * width + self.RESET,
            _fit(f"{self._t(view, '进度', 'Progress')} {' → '.join(progress_parts)}  |  {paused_text}", width),
            self.DIM + _fit(f"{self._t(view, '任务墙钟', 'Task wall-clock')} {view['elapsed']}  |  token {view['tokens']:,}  |  {self._t(view, '当前最佳', 'Current best')} {best_name} · {objective} {best_metric}", width) + self.RESET,
            _fit(f"{self._t(view, '当前', 'Current')} {(current or {}).get('research_id') or '--'} · {current_name}  |  {self._t(view, '改动点', 'Intervention')} {current_point}  |  {self._t(view, '步骤', 'Step')} {self._status_text(view, (current or {}).get('phase') or (current or {}).get('status'))}", width),
            self.DIM + "─" * width + self.RESET,
        ]
        labels = ("顺序", "类型", "模型 / idea", "耗时", "状态") if not view["english"] else ("Order", "Type", "Model / idea", "Elapsed", "Status")
        lines.append(self.BOLD + (_column(labels[0], 15) + _column(labels[1], 10) + _column(labels[2], max(14, width - 52)) + _column(objective, 12) + _column(labels[3], 9) + labels[4]) + self.RESET)
        result_rows = self._result_rows(view)
        slots = max(0, height - len(lines))
        shown = result_rows[-slots:] if slots else []
        hidden = max(0, len(result_rows) - len(shown))
        if hidden and shown:
            shown[0] = {**shown[0], "id": f"… {hidden} earlier" if view["english"] else f"…前 {hidden} 条"}
        for row in shown:
            status = {
                "done": self._t(view, "完成", "done"),
                "success": self._t(view, "成功", "success"),
                "rejected": self._t(view, "拒绝", "rejected"),
                "running": self._t(view, "运行中", "running"),
                "failed": self._t(view, "失败", "failed"),
            }[row["status"]]
            status = self._paint(status, self._status_colour(str(row["status"])))
            metric = _column(row["metric"], 12)
            metric = self._paint(metric, self._metric_colour(row["metric"]))
            lines.append(_column(row["id"], 15) + _column(row["kind"], 10) + _column(row["what"], max(14, width - 52)) + metric + _column(row["duration"], 9) + status)
        return (lines + [""] * height)[:height]

    @staticmethod
    def _reasoning_blocks(source: list[str]) -> list[tuple[int, int]]:
        """Find explicit reasoning events without swallowing ordinary logs.

        Only durable delimiters form multi-line blocks.  A JSON API record
        containing ``reasoning_content`` is an atomic one-line event: treating
        it as an open block previously hid every following training/tool line.
        """
        blocks: list[tuple[int, int]] = []
        index = 0
        marker = re.compile(r"^\s*\[reasoning(?::[^\]]*)?\]", re.IGNORECASE)
        while index < len(source):
            line = source[index]
            explicit_start = bool(marker.search(line)) or "<think" in line.lower()
            if explicit_start:
                end = index
                while end + 1 < len(source) and "[/reasoning]" not in source[end] and "</think" not in source[end].lower():
                    end += 1
                blocks.append((index, end))
                index = end + 1
                continue
            if "reasoning_content" in line:
                blocks.append((index, index))
            index += 1
        return blocks

    def _log_lines(self, height: int, width: int, view: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
        source = self.log.tail() if self._follow else list(self._frozen_history or self.log.full_history())
        blocks = self._reasoning_blocks(source)
        block_starts = {start: (number, end) for number, (start, end) in enumerate(blocks, start=1)}
        hidden_indices = {index for start, end in blocks for index in range(start, end + 1)}
        visible: list[tuple[int | None, str]] = []
        index = 0
        while index < len(source):
            if index in block_starts and not self._thinking_expanded:
                number, end = block_starts[index]
                count = end - index + 1
                visible.append((index, self._t(view, f"[LLM 思考 #{number}：{count} 行，按 T 展开]", f"[LLM thinking #{number}: {count} lines; press T]")))
                index = end + 1
                continue
            if index not in hidden_indices or self._thinking_expanded:
                visible.append((index, source[index]))
            index += 1

        wrapped: list[str] = []
        focus_start: int | None = None
        latest_start = blocks[-1][0] if blocks else None
        for source_index, line in visible:
            if self._focus_latest_thinking and source_index == latest_start:
                focus_start = len(wrapped)
            wrapped.extend(textwrap.wrap(line, width=max(8, width - 1), replace_whitespace=False, drop_whitespace=False) or [""])

        if self._focus_latest_thinking:
            self._focus_latest_thinking = False
            if focus_start is not None:
                self._scroll = max(0, len(wrapped) - (focus_start + max(1, height)))
            else:
                # No reasoning event is currently available: stay at the live
                # end instead of jumping to an arbitrary old transcript line.
                self._resume_follow()

        total = len(wrapped)
        if self._follow:
            self._scroll = 0
        self._scroll = min(max(0, self._scroll), max(0, total - height))
        end = total if self._follow else max(0, total - self._scroll)
        start = max(0, end - height)
        rows = wrapped[start:end]
        unseen = max(0, len(self.log.tail()) - self._frozen_line_count) if not self._follow else 0
        return rows, {
            "start": start + 1 if total else 0,
            "end": end,
            "total": total,
            "unseen": unseen,
            "thinking_blocks": len(blocks),
        }

    @staticmethod
    def _with_scrollbar(rows: list[str], *, height: int, width: int, start: int, total: int) -> list[str]:
        """Attach a compact visual scrollbar to the independently browsed log."""
        if height <= 0:
            return []
        row_count = max(1, len(rows))
        if total <= height:
            thumb_start, thumb_size = 0, height
        else:
            thumb_size = max(1, round(height * height / total))
            max_start = max(1, total - height)
            thumb_start = round(start / max_start * max(0, height - thumb_size))
        padded = rows + [""] * max(0, height - len(rows))
        content_width = max(1, width - 1)
        return [
            _fit(line, content_width) + ("█" if thumb_start <= index < thumb_start + thumb_size else "│")
            for index, line in enumerate(padded[:height])
        ]

    def _colour(self, line: str) -> str:
        lower = line.lower()
        if any(token in lower for token in ("traceback", "error", "exception", "failed")): return self.RED + line + self.RESET
        if any(token in lower for token in ("warning", "warn", "retry", "repair", "partial", "blocked")): return self.YELLOW + line + self.RESET
        if any(token in lower for token in ("success", "accepted", "completed", "done", "epoch", "train", "val", "test")): return self.GREEN + line + self.RESET
        if any(token in lower for token in ("api", "assistant", "reasoning", "<think", "tool", "read_source", "builder", "edit", "path")): return self.BLUE + line + self.RESET
        if any(token in lower for token in ("cache", "debug", "skipped", "raw transcript")): return self.DIM + line + self.RESET
        return line

    def render(self) -> None:
        if not self._enabled or not self._stdout: return
        with self._render_lock:
            width, height = shutil.get_terminal_size(fallback=(120, 40))
            top_height = max(1, height // 2)  # Strict 50/50 split.
            bottom_height = max(1, height - top_height)
            view = self._task_view()
            log_height = max(0, bottom_height - 2)
            raw_rows, position = self._log_lines(log_height, width, view)
            mode = self._t(view, "实时跟随" if self._follow else "历史浏览", "live follow" if self._follow else "history browse")
            position_text = self._t(view, f"第 {position['start']}–{position['end']} / {position['total']} 行", f"lines {position['start']}–{position['end']} / {position['total']}")
            unseen = self._t(view, f" · 新增 {position['unseen']} 行，按 End 返回最新", f" · {position['unseen']} new lines; End for latest") if position["unseen"] else ""
            hint = self._t(view, f"完整原始输出 · {mode} · {position_text}{unseen} | P 暂停/继续 | ↑↓/PgUp/PgDn 浏览 | End 跟随 | T 展开/折叠思考 | Q 退出界面", f"Complete raw output · {mode} · {position_text}{unseen} | P pause/resume | ↑↓/PgUp/PgDn browse | End follow | T toggle thinking | Q leave UI")
            lower_header = [self.BOLD + _fit(hint, width) + self.RESET, self.DIM + "─" * width + self.RESET]
            log_rows = self._with_scrollbar(raw_rows, height=log_height, width=width, start=max(0, position["start"] - 1), total=position["total"])
            top_rows = self._top_lines(view, top_height, width)
            screen = top_rows + lower_header + log_rows
            screen = (screen + [""] * height)[:height]
            # Paint fixed rows in place.  No full-screen clear and no newline at
            # the last row: terminal scrollback is never consumed by refreshes.
            raw_log_start = len(top_rows) + len(lower_header)
            payload = "".join(
                f"\x1b[{index + 1};1H\x1b[2K{self._colour(line) if index >= raw_log_start else line}"
                for index, line in enumerate(screen)
            )
            self._stdout.write("\x1b[?25l" + payload)
            self._stdout.flush()
