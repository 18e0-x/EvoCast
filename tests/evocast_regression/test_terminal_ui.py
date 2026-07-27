from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

from evocast.state.runtime.terminal_ui import (
    TerminalLog,
    TerminalUI,
    _metric,
    _research_result_state,
    pause_requested,
    set_pause_requested,
    terminal_control_path,
    wait_if_paused,
)
from evocast.state.runtime.store import sync_task_stage
import evocast.state.runtime.terminal_ui as terminal_ui


def test_terminal_log_persists_complete_raw_text(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    log = TerminalLog(str(base_dir), "terminal_fixture")
    log.append("LLM reasoning\nBuilder exact edit\ntraceback details\n")

    assert log.path.read_text(encoding="utf-8") == "LLM reasoning\nBuilder exact edit\ntraceback details\n"
    assert log.snapshot()[-3:] == ["LLM reasoning", "Builder exact edit", "traceback details"]


def test_terminal_pause_control_is_persistent_and_cooperative(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "pause_fixture"
    set_pause_requested(str(base_dir), task_id, True)
    assert pause_requested(str(base_dir), task_id) is True
    persisted = json.loads(terminal_control_path(str(base_dir), task_id).read_text(encoding="utf-8"))
    assert persisted["source"] == "terminal_ui"

    # Resume from another control writer; wait_if_paused must then return.
    import threading

    thread = threading.Thread(target=wait_if_paused, args=(str(base_dir), task_id))
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    set_pause_requested(str(base_dir), task_id, False)
    thread.join(timeout=1)
    assert not thread.is_alive()


class _TTYCapture:
    def __init__(self) -> None:
        self.output = ""

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.output += text
        return len(text)

    def flush(self) -> None:
        return None


class _TTYInput:
    def __init__(self, fd: int = 123) -> None:
        self.fd = fd

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self.fd


class _PlainCapture(_TTYCapture):
    def isatty(self) -> bool:
        return False


def test_terminal_dashboard_enables_on_posix_tty(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "posix_dashboard"
    (base_dir / "task_knowledge" / task_id).mkdir(parents=True)
    capture, stdin = _TTYCapture(), _TTYInput()
    monkeypatch.setattr(terminal_ui.sys, "stdin", stdin)
    monkeypatch.setattr(terminal_ui.sys, "stdout", capture)
    monkeypatch.setattr(terminal_ui.sys, "stderr", capture)
    prepared = {"value": False}

    def _prepare() -> None:
        prepared["value"] = True

    ui = TerminalUI(str(base_dir), task_id)
    monkeypatch.setattr(terminal_ui.os, "name", "posix")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(ui, "_prepare_terminal", _prepare)
    monkeypatch.setattr(ui, "_event_loop", lambda: None)
    monkeypatch.setattr(ui, "render", lambda: None)
    ui.start()
    ui.stop()

    assert prepared["value"] is True
    assert "[terminal] raw transcript" not in capture.output


def test_terminal_dashboard_stays_raw_for_non_interactive_output(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "raw_non_tty"
    capture, stdin = _PlainCapture(), _TTYInput()
    monkeypatch.setattr(terminal_ui.sys, "stdin", stdin)
    monkeypatch.setattr(terminal_ui.sys, "stdout", capture)
    monkeypatch.setattr(terminal_ui.sys, "stderr", capture)
    ui = TerminalUI(str(base_dir), task_id)
    monkeypatch.setattr(terminal_ui.os, "name", "posix")
    monkeypatch.setenv("TERM", "xterm-256color")
    ui.start()
    ui.stop()

    assert "[terminal] raw transcript" in capture.output


def test_terminal_posix_mode_sets_cbreak_and_restores_attrs(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / ".evocast"
    ui = TerminalUI(str(base_dir), "posix_restore")
    ui._stdin = _TTYInput(456)
    monkeypatch.setattr(terminal_ui.os, "name", "posix")
    monkeypatch.setattr(ui, "_install_terminal_signal_handlers", lambda: None)
    monkeypatch.setattr(ui, "_restore_terminal_signal_handlers", lambda: None)
    calls: list[tuple[str, int, object]] = []
    fake_termios = SimpleNamespace(
        TCSADRAIN=1,
        tcgetattr=lambda fd: ["original", fd],
        tcsetattr=lambda fd, when, attrs: calls.append(("tcsetattr", fd, attrs)),
    )
    fake_tty = SimpleNamespace(setcbreak=lambda fd: calls.append(("setcbreak", fd, None)))
    monkeypatch.setitem(sys.modules, "termios", fake_termios)
    monkeypatch.setitem(sys.modules, "tty", fake_tty)

    ui._prepare_terminal()
    ui._restore_terminal()

    assert calls == [("setcbreak", 456, None), ("tcsetattr", 456, ["original", 456])]
    assert ui._terminal_fd is None
    assert ui._terminal_attrs is None


def test_terminal_posix_keys_match_windows_controls(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "posix_keys"
    ui = TerminalUI(str(base_dir), task_id)
    ui._terminal_fd = 789
    set_pause_requested(str(base_dir), task_id, False)
    stream = iter([b"p", b"\x1b", b"[", b"A", b"\x1b", b"[", b"5", b"~", b"q"])

    def _select(readers, writers, errors, timeout):
        return ([789], [], []) if timeout == 0 else ([789], [], [])

    def _read(fd: int, count: int) -> bytes:
        try:
            return next(stream)
        except StopIteration:
            return b""

    monkeypatch.setitem(sys.modules, "select", SimpleNamespace(select=_select))
    monkeypatch.setattr(terminal_ui.os, "read", _read)
    monkeypatch.setattr(ui, "detach", lambda: setattr(ui, "_enabled", False))

    ui._read_posix_keys()

    assert pause_requested(str(base_dir), task_id) is True
    assert ui._follow is False
    assert ui._scroll == 13
    assert ui._enabled is False


def test_terminal_dashboard_uses_fixed_half_screen_and_keeps_raw_log(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_english"
    task_dir = base_dir / "task_knowledge" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "language": "en", "objective_metric": "mse_norm", "max_rounds": 3}),
        encoding="utf-8",
    )
    sync_task_stage(str(base_dir), task_id, stage="baseline_search", status="running")
    ui = TerminalUI(str(base_dir), task_id)
    capture = _TTYCapture()
    ui._stdout = capture
    ui._enabled = True
    ui.log.append("Builder generated exact edits\n")
    monkeypatch.setattr(terminal_ui.shutil, "get_terminal_size", lambda fallback: os.terminal_size((120, 40)))
    ui.render()

    assert "Experiment Console" in capture.output
    assert "Complete raw output" in capture.output
    assert "Builder generated exact edits" in capture.output
    assert "\x1b[2J" not in capture.output
    # Row 21 begins the lower panel in a 40-row terminal: strict 50/50.
    assert "\x1b[21;1H" in capture.output


def test_log_browsing_freezes_history_while_new_output_arrives(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_history"
    task_dir = base_dir / "task_knowledge" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    sync_task_stage(str(base_dir), task_id, stage="research", status="running")
    ui = TerminalUI(str(base_dir), task_id)
    ui.log.append("old line\n")
    ui._browse(1)
    frozen = list(ui._frozen_history or [])
    ui.log.append("new live line\n")

    assert frozen == ["old line"]
    assert ui._frozen_history == frozen


def test_expanding_thinking_focuses_the_latest_reasoning_block_start(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_thinking_focus"
    task_dir = base_dir / "task_knowledge" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "task_config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    sync_task_stage(str(base_dir), task_id, stage="research", status="running")
    ui = TerminalUI(str(base_dir), task_id)
    ui.log.append("before\n[reasoning:designer] first thought\nsecond thought\nthird thought\n[/reasoning]\nafter\n")
    ui._freeze_history()
    ui._follow = False
    ui._thinking_expanded = True
    ui._focus_latest_thinking = True

    rows, _ = ui._log_lines(3, 120, ui._task_view())

    assert rows[0] == "[reasoning:designer] first thought"
    assert ui._focus_latest_thinking is False
    assert ui._follow is False


def test_terminal_collapses_only_explicit_reasoning_blocks_and_keeps_following_logs(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_reasoning_boundaries"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    ui = TerminalUI(str(base_dir), task_id)
    ui.log.append("before\n[reasoning:builder] detailed thought\n[/reasoning]\nafter training epoch\n")

    rows, position = ui._log_lines(10, 100, ui._task_view())

    assert "before" in rows
    assert any("LLM 思考" in line for line in rows)
    assert "after training epoch" in rows
    assert position["thinking_blocks"] == 1


def test_terminal_expansion_refreshes_to_latest_reasoning_block(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_latest_reasoning"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    ui = TerminalUI(str(base_dir), task_id)
    ui.log.append("[reasoning:old] old\n[/reasoning]\nold tail\n")
    ui._freeze_history()
    ui.log.append("[reasoning:new] newest\n[/reasoning]\nnew tail\n")
    ui._thinking_expanded = True
    ui._frozen_history = ui.log.full_history()
    ui._frozen_line_count = len(ui._frozen_history)
    ui._follow = False
    ui._focus_latest_thinking = True

    rows, position = ui._log_lines(3, 100, ui._task_view())

    assert rows[0] == "[reasoning:new] newest"
    assert position["thinking_blocks"] == 2


def test_terminal_scrollbar_reports_history_position_and_unseen_output(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    task_id = "terminal_scrollbar"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(json.dumps({"task_id": task_id}), encoding="utf-8")
    ui = TerminalUI(str(base_dir), task_id)
    ui.log.append("\n".join(f"line {index}" for index in range(30)) + "\n")
    ui._browse(20)
    ui.log.append("new live line\n")

    rows, position = ui._log_lines(5, 80, ui._task_view())
    painted = ui._with_scrollbar(rows, height=5, width=80, start=position["start"] - 1, total=position["total"])

    assert position["unseen"] == 1
    assert position["start"] > 1
    assert len(painted) == 5
    assert any(line.endswith("█") for line in painted)


def test_terminal_reads_live_ablation_records_and_preproposal_state(tmp_path: Path) -> None:
    """The console must read the artifacts the live runners actually write."""
    base_dir, task_id = tmp_path / ".evocast", "mixed_layout"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 5}), encoding="utf-8")
    for name, status, metric, promoted in (("Ablation001", "success", 1.01, True), ("Ablation002", "failed_ablation_round", None, False)):
        folder = root / "rounds" / name
        folder.mkdir(parents=True)
        payload = {"ablation_id": name, "status": status, "mechanism_name": f"mechanism-{name}", "metrics": {"mse_norm": metric} if metric else {}, "seed_eval": {"promoted_to_current_best": promoted}}
        (folder / "ablation_record.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "rounds" / "Research001").mkdir(parents=True)
    (root / "research_state.json").write_text(json.dumps({"updated_at": "2026-07-17T01:00:00", "current_action": {"status": "selected", "research_question": "current executable question", "primary_fit_point": "self.encoder"}}), encoding="utf-8")
    sync_task_stage(str(base_dir), task_id, stage="baseline_diagnosis", status="running")

    ui = TerminalUI(str(base_dir), task_id)
    view, rows = ui._task_view(), None
    rows = ui._result_rows(view)
    by_id = {row["id"]: row for row in rows}

    assert by_id["Ablation001"]["metric"] == "1.010000"
    assert by_id["Ablation001"]["status"] == "success"
    assert by_id["Ablation002"]["status"] == "failed"
    assert by_id["Research001"]["status"] == "running"
    assert view["active_research"]["fit_point"] == "self.encoder"


def test_terminal_metric_display_uses_six_decimal_places() -> None:
    assert _metric({"metrics": {"mse_norm": 0.1}}, "mse_norm") == "0.100000"
    assert _metric({"candidate_metrics": {"mse_norm": 1.23456789}}, "mse_norm") == "1.234568"


def test_terminal_dashboard_uses_semantic_status_colours(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "dashboard_colours"
    root = base_dir / "task_knowledge" / task_id
    rounds = root / "rounds"
    rounds.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 2}),
        encoding="utf-8",
    )
    from evocast.state.runtime.store import sync_best_baseline

    sync_best_baseline(
        str(base_dir),
        task_id,
        {
            "candidate_id": "baseline_001",
            "display_name": "SparseTSF",
            "metrics": {"mse_norm": 1.0},
            "objective_metric": "mse_norm",
        },
    )
    for label, status, metrics in (
        ("Research001", "completed", {"mse_norm": 0.9}),
        ("Research002", "round_started", {}),
        ("Research003", "implementation_rejected", {}),
    ):
        folder = rounds / label
        folder.mkdir()
        (folder / "round.json").write_text(
            json.dumps(
                {
                    "round_id": int(label[-3:]),
                    "research_id": label,
                    "status": status,
                    "model_family": "SparseTSF",
                    "idea_summary": "short idea",
                    "metrics": metrics,
                }
            ),
            encoding="utf-8",
        )

    # Model an explicit one-time legacy migration after all historical files
    # exist; normal canonical reads never rescan them.
    from evocast.state.domain_store import domain_state_path
    domain_state_path(str(base_dir), task_id).unlink(missing_ok=True)
    sync_best_baseline(
        str(base_dir), task_id,
        {"candidate_id": "baseline_001", "display_name": "SparseTSF", "metrics": {"mse_norm": 1.0}, "objective_metric": "mse_norm"},
    )

    ui = TerminalUI(str(base_dir), task_id)
    top = "\n".join(ui._top_lines(ui._task_view(), height=20, width=140))

    assert f"{ui.GREEN}完成{ui.RESET}" in top
    assert f"{ui.YELLOW}运行中{ui.RESET}" in top
    assert f"{ui.RED}失败{ui.RESET}" in top
    assert f"{ui.DIM}--" in top


def test_terminal_research_result_state_uses_current_best_promotion_only() -> None:
    objective = "mse_norm"

    assert _research_result_state(
        {"status": "completed", "metrics": {objective: 0.9}, "gate_decision": "accept"},
        objective,
    ) == "failed"
    assert _research_result_state(
        {"status": "rejected", "metrics": {objective: 1.2}, "gate_event": {"gate": {"decision": "reject"}}},
        objective,
    ) == "failed"
    assert _research_result_state(
        {"status": "marginal_no_seed_eval", "metrics": {objective: 0.999}},
        objective,
    ) == "failed"
    assert _research_result_state(
        {
            "status": "scientific_rejected",
            "metrics": {objective: 0.95},
            "seed_eval": {"status": "completed", "significance_decision": {"decision": "reject"}},
        },
        objective,
    ) == "failed"
    assert _research_result_state(
        {
            "status": "rejected",
            "metrics": {objective: 0.95},
            "seed_eval": {
                "status": "completed",
                "significance_decision": {"decision": "accept"},
                "promoted_to_current_best": False,
            },
        },
        objective,
    ) == "failed"
    assert _research_result_state(
        {
            "status": "rejected",
            "metrics": {objective: 0.95},
            "seed_eval": {
                "status": "completed",
                "significance_decision": {"decision": "accept"},
                "promoted_to_current_best": True,
            },
        },
        objective,
    ) == "success"
    assert _research_result_state(
        {"status": "infra_failed", "metrics": {}},
        objective,
    ) == "failed"
    assert _research_result_state(
        {"status": "round_started", "metrics": {}},
        objective,
    ) == "running"


def test_terminal_research_rows_show_promotion_success_or_failed_labels(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "three_outcome_labels"
    root = base_dir / "task_knowledge" / task_id
    rounds = root / "rounds"
    rounds.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 4}),
        encoding="utf-8",
    )
    for label, status, metrics, extra in (
        ("Research001", "completed", {"mse_norm": 0.8}, {"gate_decision": "accept", "seed_eval": {"promoted_to_current_best": True}}),
        ("Research002", "rejected", {"mse_norm": 1.2}, {}),
        ("Research003", "marginal_no_seed_eval", {"mse_norm": 0.999}, {}),
        ("Research004", "infra_failed", {}, {}),
    ):
        folder = rounds / label
        folder.mkdir()
        (folder / "round.json").write_text(
            json.dumps(
                {
                    "round_id": int(label[-3:]),
                    "research_id": label,
                    "status": status,
                    "model_family": "Amplifier",
                    "idea_summary": "short idea",
                    "metrics": metrics,
                    **extra,
                }
            ),
            encoding="utf-8",
        )

    ui = TerminalUI(str(base_dir), task_id)
    rows = {item["id"]: item for item in ui._result_rows(ui._task_view())}
    top = "\n".join(ui._top_lines(ui._task_view(), height=20, width=150))

    assert rows["Research001"]["status"] == "success"
    assert rows["Research002"]["status"] == "failed"
    assert rows["Research003"]["status"] == "failed"
    assert rows["Research004"]["status"] == "failed"
    assert f"{ui.GREEN}成功{ui.RESET}" in top
    assert f"{ui.RED}失败{ui.RESET}" in top


def test_terminal_raw_log_uses_standard_keyword_colours(tmp_path: Path) -> None:
    base_dir = tmp_path / ".evocast"
    ui = TerminalUI(str(base_dir), "raw_log_colours")

    assert ui._colour("Traceback: failed") == f"{ui.RED}Traceback: failed{ui.RESET}"
    assert ui._colour("repair retry warning") == f"{ui.YELLOW}repair retry warning{ui.RESET}"
    assert ui._colour("completed accepted") == f"{ui.GREEN}completed accepted{ui.RESET}"
    assert ui._colour("api tool path") == f"{ui.BLUE}api tool path{ui.RESET}"
    assert ui._colour("cache debug skipped") == f"{ui.DIM}cache debug skipped{ui.RESET}"


def test_terminal_baseline_row_metric_uses_six_decimal_places(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "baseline_metric_display"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    from evocast.state.runtime.store import sync_best_baseline

    sync_best_baseline(
        str(base_dir),
        task_id,
        {
            "candidate_id": "baseline_001",
            "display_name": "ts_benchmark.baselines.sparsetsf.SparseTSF",
            "metrics": {"mse_norm": 1.2121407561166886},
            "objective_metric": "mse_norm",
        },
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Baseline"]

    assert row["what"] == "SparseTSF"
    assert row["metric"] == "1.212141"


def test_terminal_research_rows_prefer_recorded_idea_over_fit_point(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "research_idea_display"
    root = base_dir / "task_knowledge" / task_id
    round_dir = root / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "implementation_rejected",
                "fit_point": "PAttn",
                "model_family": "PAttn",
                "display_idea": "PAttn | zero-initialized residual gate over raw patch embedding",
                "idea_summary": "zero-initialized residual gate over raw patch embedding",
                "mechanism_summary": "Created a PAttn variant with a zero-initialized residual gate over raw patch embedding.",
                "research_intent": "Create a task-local forecasting research variant for PAttn.",
                "hypothesis": "A focused architecture change to PAttn can reduce mse_norm.",
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]

    assert row["what"] == "PAttn | zero-initialized residual gate over raw patch embedding"
    assert row["what"] != "PAttn"


def test_terminal_research_idea_is_capped_at_eight_words(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "research_idea_short"
    root = base_dir / "task_knowledge" / task_id
    round_dir = root / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "completed",
                "model_family": "PAttn",
                "display_idea": "PAttn | zero initialized residual gate over raw patch embedding with adaptive scaling",
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]

    words = [word for word in row["what"].replace("|", " ").split() if word]
    assert row["what"] == "PAttn | zero initialized residual gate over raw patch"
    assert len(words) <= 8


def test_terminal_research_cjk_idea_is_visually_short(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "research_idea_cjk_short"
    root = base_dir / "task_knowledge" / task_id
    round_dir = root / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "completed",
                "model_family": "PAttn",
                "display_idea": "零初始化残差门控结合自适应噪声抑制和多尺度补偿",
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]

    assert row["what"] == "PAttn | 零初始化残差门控结合自适应噪声抑"
    assert len(row["what"].split("|", 1)[1].strip()) == 16


def test_terminal_active_research_idea_is_compact(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "active_research_short"
    root = base_dir / "task_knowledge" / task_id
    root.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (root / "research_state.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-07-19T01:00:00",
                "current_action": {
                    "status": "selected",
                    "research_question": "zero initialized residual gate over raw patch embedding with adaptive scaling",
                    "primary_fit_point": "PAttn",
                },
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]

    assert row["what"] == "zero initialized residual gate over raw patch embedding"
    assert len(row["what"].split()) == 8


def test_terminal_research_semantic_goal_shortens_import_path_model(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "semantic_goal_import_path"
    root = base_dir / "task_knowledge" / task_id
    round_dir = root / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "round_started",
                "research_intent": "Modify the active ts_benchmark.baselines.sparsetsf.SparseTSF source to improve mse_norm",
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]
    words = [word for word in row["what"].replace("|", " ").split() if word]

    assert row["what"] == "SparseTSF | Modify the active source to improve mse_norm"
    assert len(words) <= 8
    assert "ts_benchmark" not in row["what"]


def test_terminal_research_rows_ignore_mechanism_summary_execution_noise(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "research_idea_ignores_execution_noise"
    root = base_dir / "task_knowledge" / task_id
    round_dir = root / "rounds" / "Research001"
    round_dir.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 1}),
        encoding="utf-8",
    )
    (round_dir / "round.json").write_text(
        json.dumps(
            {
                "round_id": 1,
                "research_id": "Research001",
                "status": "rejected",
                "fit_point": "FreTS",
                "model_family": "FreTS",
                "idea_summary": "Modified active source: F.relu -> torch.tanh; added self.alpha",
                "mechanism_summary": "coding backend exceeded max_tool_turns=10; workspace has changes for external audit",
                "execution_summary": "coding backend exceeded max_tool_turns=10; workspace has changes for external audit",
            }
        ),
        encoding="utf-8",
    )

    ui = TerminalUI(str(base_dir), task_id)
    row = {item["id"]: item for item in ui._result_rows(ui._task_view())}["Research001"]

    assert "torch.tanh" in row["what"]
    assert "max_tool_turns" not in row["what"]


def test_terminal_separates_diagnostic_ablations_from_formal_research_budget(tmp_path: Path) -> None:
    base_dir, task_id = tmp_path / ".evocast", "diagnostic_vs_formal"
    root = base_dir / "task_knowledge" / task_id
    rounds = root / "rounds"
    rounds.mkdir(parents=True)
    (root / "task_config.json").write_text(
        json.dumps({"task_id": task_id, "objective_metric": "mse_norm", "max_rounds": 3}),
        encoding="utf-8",
    )
    for idx in range(1, 7):
        is_diagnostic = idx <= 3
        label = f"Ablation{idx:03d}" if is_diagnostic else f"Research{idx - 3:03d}"
        round_dir = rounds / label
        round_dir.mkdir()
        (round_dir / "round.json").write_text(
            json.dumps(
                {
                    "round_id": idx if is_diagnostic else idx - 3,
                    "research_id": label,
                    **({"ablation_id": label} if is_diagnostic else {}),
                    "status": "rejected",
                    "round_scope": "baseline_diagnosis" if is_diagnostic else "research",
                    "counts_toward_research_budget": not is_diagnostic,
                    "model_family": "TimeBase",
                    "idea_summary": f"idea {idx}",
                    "created_at": "2026-07-19T01:00:00",
                    "closed_at": "2026-07-19T01:01:00",
                }
            ),
            encoding="utf-8",
        )

    ui = TerminalUI(str(base_dir), task_id)
    view = ui._task_view()
    rows = {item["id"]: item for item in ui._result_rows(view)}
    top = "\n".join(ui._top_lines(view, height=12, width=160))

    assert rows["Ablation001"]["kind"] == "Ablation"
    assert rows["Ablation003"]["kind"] == "Ablation"
    assert rows["Research001"]["kind"] == "Research"
    assert "正式 Research 3/3" in top
    assert "正式 Research 6/3" not in top
    assert "诊断消融 3/3" in top
    assert "任务墙钟" in top
    assert "累计耗时" not in top
