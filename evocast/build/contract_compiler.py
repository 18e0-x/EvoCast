from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evocast.build.contract import BuildContract
from evocast.build.source_snapshot import source_manifest
from evocast.domain.baseline_identity import source_binding_from_entry_file
from evocast.domain.effective_model_config import resolve_effective_model_config
from evocast.domain.execution_ids import format_ablation_id, format_research_id, parse_ablation_id
from evocast.domain.knowledge_paths import repo_root, task_knowledge_dir
from evocast.domain.task_identity import resolve_compiled_config_path
from evocast.harness import rounds as round_records
from evocast.policy.agent_control_policy import research_repair_budget
from evocast.policy.experiment_policy import task_build_mode
from evocast.research.ablation.exact_contract import compile_exact_ablation_target, exact_target_instruction
from evocast.runners.tfb_pipeline_runner import load_config_json
from evocast.state.runtime.store import load_runtime_state
from evocast.state.domain_store import load_task_config


def _task_config(base_dir: str, task_id: str) -> dict[str, Any]:
    return load_task_config(base_dir, task_id)


def _agent_ablation_mode(base_dir: str, task_id: str) -> str:
    value = str(_task_config(base_dir, task_id).get("agent_ablation") or "none").strip()
    if not value or value.lower() == "none":
        return "none"
    return value.upper()


def _repo_wide_paths(source_checkout: str | Path | None) -> tuple[list[str], list[str]]:
    if not source_checkout:
        return [], ["."]
    manifest = source_manifest(source_checkout)
    allowed_edit_files = [
        _norm(str(item.get("path") or ""))
        for item in list(manifest.get("files") or [])
        if _norm(str(item.get("path") or ""))
    ]
    return list(dict.fromkeys(allowed_edit_files)), ["."]


def next_research_id(base_dir: str, task_id: str) -> str:
    indices = [
        int(record.get("round_id") or 0)
        for record in round_records.list_rounds(base_dir, task_id)
        if round_records._round_counts_toward_research_budget(record)
    ]
    return format_research_id(max(indices or [0]) + 1)


def next_ablation_id(base_dir: str, task_id: str) -> str:
    root = task_knowledge_dir(base_dir, task_id) / "rounds"
    indices = [
        parsed
        for parsed in (parse_ablation_id(path.name) for path in root.glob("Ablation*"))
        if parsed is not None
    ]
    for record in round_records.list_rounds(base_dir, task_id):
        if str(record.get("round_scope") or "") == round_records.ROUND_SCOPE_BASELINE_DIAGNOSIS:
            indices.append(int(record.get("round_id") or 0))
    return format_ablation_id(max(indices or [0]) + 1)


def syntax_check_command(path: str) -> list[str]:
    return [
        "python",
        "-c",
        f"import ast, pathlib; ast.parse(pathlib.Path({path!r}).read_text(encoding='utf-8'))",
    ]


def model_entry_contract_command(path: str) -> list[str]:
    return [
        "python",
        "-c",
        (
            "import ast, pathlib; "
            f"ast.parse(pathlib.Path({path!r}).read_text(encoding='utf-8')); "
            "print('SOURCE_PARSE_OK')"
        ),
    ]


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


def _candidate_payload(candidate: Any) -> dict[str, Any]:
    return candidate.to_dict() if candidate and getattr(candidate, "candidate_id", "") else {}


def _normalize_evaluation_stage(value: Any, *, build_mode: bool = False) -> str:
    text = str(value or "").strip().lower()
    if text == "standard":
        text = "experiment"
    if text in {"smoke", "build_mode", "experiment", "seed_eval"}:
        return text
    return "smoke" if build_mode else "experiment"


def _repo_relative(path: str | Path, repo_dir: str | Path | None = None) -> str:
    root = Path(repo_dir or repo_root()).resolve()
    value = Path(path)
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return _norm(str(path))


def _load_source_binding(active: dict[str, Any], repo_dir: str | Path | None = None) -> dict[str, Any]:
    binding = dict(active.get("source_binding") or {})
    binding_ref = str(active.get("model_binding_ref") or "").strip()
    if binding_ref:
        path = Path(binding_ref)
        if path.is_file():
            try:
                persisted = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                persisted = {}
            if isinstance(persisted, dict):
                binding = dict(persisted.get("source_binding") or binding)
                if not binding and persisted.get("source_file"):
                    binding = _source_binding_from_entry(
                        str(persisted.get("source_file") or ""),
                        repo_dir=repo_dir,
                    )
                if binding and not binding.get("public_import_path") and persisted.get("public_import_path"):
                    binding["public_import_path"] = str(persisted.get("public_import_path") or "")
    if not binding and active.get("source_file"):
        binding = _source_binding_from_entry(str(active.get("source_file") or ""), repo_dir=repo_dir)
    return _canonical_source_binding(binding, repo_dir)


def _has_verified_source_binding(binding: dict[str, Any]) -> bool:
    return bool(str((binding or {}).get("entry_file") or "").strip() or list((binding or {}).get("source_files") or []))


def _source_binding_from_entry(entry_file: str, *, repo_dir: str | Path | None = None) -> dict[str, Any]:
    return source_binding_from_entry_file(entry_file, repo_dir=repo_dir or repo_root())


def _canonical_source_binding(binding: dict[str, Any], repo_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_dir or repo_root()).resolve()

    def existing(rel: str) -> str:
        normalized = _norm(rel)
        if not normalized:
            return ""
        target = root / normalized
        if target.is_file():
            return _repo_relative(target, root)
        return normalized

    entry = existing(str(binding.get("entry_file") or ""))
    if entry and entry.replace("\\", "/").startswith("ts_benchmark/baselines/time_series_library/models/"):
        refreshed = source_binding_from_entry_file(entry, repo_dir=root)
        return {
            **dict(binding or {}),
            **refreshed,
            "public_import_path": str(binding.get("public_import_path") or refreshed.get("public_import_path") or ""),
        }
    source_files = [existing(str(item)) for item in list(binding.get("source_files") or [])]
    core_files = [existing(str(item)) for item in list(binding.get("core_files") or [])]
    support_files = [existing(str(item)) for item in list(binding.get("support_files") or [])]
    ordered = [path for path in [entry, *core_files, *source_files, *support_files] if path]
    source_files = list(dict.fromkeys(ordered))
    core_files = [path for path in dict.fromkeys(core_files) if path and path in source_files]
    support_files = [path for path in dict.fromkeys(support_files) if path and path in source_files]
    return {
        **dict(binding or {}),
        "schema_version": str(binding.get("schema_version") or "source_binding_v1"),
        "entry_file": entry or (source_files[0] if source_files else ""),
        "source_files": source_files,
        "core_files": core_files,
        "support_files": support_files,
    }


def _new_file_roots_for_source_files(source_files: list[str]) -> list[str]:
    roots: list[str] = []
    for raw in source_files:
        path = _norm(raw)
        if not path or "/" not in path:
            continue
        parent = path.rsplit("/", 1)[0]
        if parent and parent not in roots:
            roots.append(parent)
    return roots


def active_source_ref(*, base_dir: str, task_id: str, repo_dir: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    active = _candidate_payload(state.current_best) or _candidate_payload(state.baseline)
    baseline_active = _candidate_payload(state.baseline)
    explicit = dict(active.get("source_ref") or {})
    baseline_snapshot = dict(active.get("baseline_snapshot") or {})
    snapshot_source_root = str(baseline_snapshot.get("source_root") or "").strip()
    snapshot_checkout = ""
    if snapshot_source_root:
        source_root_path = Path(snapshot_source_root)
        snapshot_checkout = str(source_root_path.parent.resolve()) if source_root_path.name == "ts_benchmark" else str(source_root_path.resolve())
    explicit_checkout = str(explicit.get("source_checkout") or active.get("source_checkout") or "").strip()
    source_checkout = explicit_checkout or snapshot_checkout
    if not source_checkout:
        raise RuntimeError(
            "active source snapshot is required before opening a BuildContract; "
            "expected current_best.source_ref.source_checkout or baseline.baseline_snapshot.source_root"
        )
    checkout_root = Path(source_checkout).resolve()
    if not checkout_root.is_dir():
        raise RuntimeError(f"active source checkout does not exist: {checkout_root}")
    manifest = source_manifest(checkout_root)
    snapshot_id = str(
        explicit.get("candidate_snapshot_id")
        or active.get("candidate_snapshot_id")
        or explicit.get("base_snapshot_id")
        or baseline_snapshot.get("snapshot_id")
        or manifest.get("snapshot_id")
        or ""
    ).strip()
    if not snapshot_id:
        raise RuntimeError("active source snapshot id could not be resolved")
    source_ref = {
        **explicit,
        "kind": "source_snapshot",
        "candidate_snapshot_id": snapshot_id,
        "base_snapshot_id": str(explicit.get("base_snapshot_id") or baseline_snapshot.get("snapshot_id") or snapshot_id),
        "source_checkout": str(checkout_root),
        "source_manifest_hash": str(manifest.get("manifest_hash") or ""),
        "candidate_id": str(active.get("candidate_id") or ""),
    }
    if baseline_snapshot:
        source_ref["baseline_snapshot"] = baseline_snapshot
    source_binding = _load_source_binding(active, repo_dir)
    if not _has_verified_source_binding(source_binding) and explicit.get("source_binding"):
        source_binding = _canonical_source_binding(dict(explicit.get("source_binding") or {}), repo_dir)
    if not _has_verified_source_binding(source_binding) and baseline_active and baseline_active != active:
        source_binding = _load_source_binding(baseline_active, repo_dir)
    if _has_verified_source_binding(source_binding):
        source_ref["source_binding"] = source_binding
    model_binding_ref = (
        explicit.get("model_binding_ref")
        or active.get("model_binding_ref")
        or (baseline_active.get("model_binding_ref") if baseline_active else "")
    )
    if model_binding_ref:
        source_ref["model_binding_ref"] = str(model_binding_ref)
    return active, source_ref


def _checkout_from_snapshot_root(snapshot_source_root: str) -> str:
    root = Path(str(snapshot_source_root or "").strip())
    if not root:
        return ""
    resolved = root.resolve()
    return str(resolved.parent) if resolved.name == "ts_benchmark" else str(resolved)


def _resolve_source_checkout_from_binding(
    *,
    baseline: dict[str, Any],
    source_binding: dict[str, Any],
    repo_dir: str | Path | None = None,
) -> str:
    root = Path(repo_dir or repo_root()).resolve()
    entry_file = _norm(str(source_binding.get("entry_file") or baseline.get("source_file") or ""))
    if entry_file and (root / entry_file).is_file():
        return str(root)
    source_files = [_norm(str(item)) for item in list(source_binding.get("source_files") or [])]
    for rel_path in source_files:
        if rel_path and (root / rel_path).is_file():
            return str(root)
    return ""


def baseline_diagnosis_source_ref(
    *,
    base_dir: str,
    task_id: str,
    baseline: dict[str, Any],
    repo_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_runtime_state(base_dir, task_id, auto_migrate=False)
    selected_baseline = dict(baseline or _candidate_payload(state.baseline) or {})
    if not selected_baseline:
        raise RuntimeError("baseline diagnosis requires a selected baseline with formal source authority")

    explicit = dict(selected_baseline.get("source_ref") or {})
    baseline_snapshot = dict(selected_baseline.get("baseline_snapshot") or {})
    source_binding = _load_source_binding(selected_baseline, repo_dir)
    if not _has_verified_source_binding(source_binding) and explicit.get("source_binding"):
        source_binding = _canonical_source_binding(dict(explicit.get("source_binding") or {}), repo_dir)

    explicit_checkout = str(explicit.get("source_checkout") or selected_baseline.get("source_checkout") or "").strip()
    snapshot_checkout = _checkout_from_snapshot_root(str(baseline_snapshot.get("source_root") or ""))
    binding_checkout = _resolve_source_checkout_from_binding(
        baseline=selected_baseline,
        source_binding=source_binding,
        repo_dir=repo_dir,
    )
    source_checkout = explicit_checkout or snapshot_checkout or binding_checkout
    if not source_checkout:
        raise RuntimeError(
            "baseline diagnosis requires formal baseline source authority; "
            "expected selected baseline source_ref/source_checkout, baseline_snapshot, or verified source_binding"
        )
    checkout_root = Path(source_checkout).resolve()
    if not checkout_root.is_dir():
        raise RuntimeError(f"baseline diagnosis source checkout does not exist: {checkout_root}")

    manifest = source_manifest(checkout_root)
    snapshot_id = str(
        explicit.get("candidate_snapshot_id")
        or explicit.get("base_snapshot_id")
        or baseline_snapshot.get("snapshot_id")
        or manifest.get("snapshot_id")
        or ""
    ).strip()
    if not snapshot_id:
        raise RuntimeError("baseline diagnosis source snapshot id could not be resolved")

    source_ref = {
        **explicit,
        "kind": "source_snapshot",
        "candidate_snapshot_id": snapshot_id,
        "base_snapshot_id": str(explicit.get("base_snapshot_id") or baseline_snapshot.get("snapshot_id") or snapshot_id),
        "source_checkout": str(checkout_root),
        "source_manifest_hash": str(manifest.get("manifest_hash") or ""),
        "candidate_id": str(selected_baseline.get("candidate_id") or ""),
    }
    if baseline_snapshot:
        source_ref["baseline_snapshot"] = baseline_snapshot
    if _has_verified_source_binding(source_binding):
        source_ref["source_binding"] = source_binding
    model_binding_ref = str(
        explicit.get("model_binding_ref")
        or selected_baseline.get("model_binding_ref")
        or ""
    ).strip()
    if model_binding_ref:
        source_ref["model_binding_ref"] = model_binding_ref
    return selected_baseline, source_ref


def _model_source_file(import_path: str, model_name: str, repo_dir: str | Path | None = None) -> str:
    root = Path(repo_dir or repo_root()).resolve()
    candidates: list[str] = []
    if import_path:
        if import_path.startswith("ts_benchmark.baselines.time_series_library."):
            name = import_path.rsplit(".", 1)[-1]
            candidates.extend(
                [
                    f"ts_benchmark/baselines/time_series_library/models/{name}.py",
                    f"ts_benchmark/baselines/time_series_library/patchs/{name}.py",
                ]
            )
        candidates.append(import_path.replace(".", "/") + ".py")
    if model_name:
        candidates.extend(
            [
                f"ts_benchmark/baselines/time_series_library/models/{model_name}.py",
                f"ts_benchmark/baselines/time_series_library/patchs/{model_name}.py",
            ]
        )
    for candidate in list(dict.fromkeys(_norm(item) for item in candidates if _norm(item))):
        if (root / candidate).is_file():
            return candidate
    if candidates:
        return _norm(candidates[0])
    raise RuntimeError("unable to resolve model source file")


def _first_existing_repo_file(paths: list[str], repo_dir: str | Path | None = None) -> str:
    root = Path(repo_dir or repo_root()).resolve()
    for path in paths:
        normalized = _norm(path)
        if normalized and (root / normalized).is_file():
            return normalized
    return ""


def _effective_model_config_for_contract(
    *,
    base_dir: str,
    task_id: str,
    baseline: dict[str, Any],
    requested_budget: str,
    smoke: bool,
) -> dict[str, Any]:
    model_config = dict(baseline.get("model_config") or {})
    if model_config.get("effective_model_hyper_params"):
        effective = dict(model_config.get("effective_model_hyper_params") or {})
        return {**model_config, "model_hyper_params": effective}
    try:
        config_path = resolve_compiled_config_path(task_id, base_dir)
        config_data = load_config_json(config_path)
    except FileNotFoundError:
        return model_config
    resolved = resolve_effective_model_config(
        config_data=config_data,
        base_dir=base_dir,
        task_id=task_id,
        model_entry={
            "model_key": baseline.get("display_name") or baseline.get("model_name") or model_config.get("model_key"),
            "model_name": model_config.get("model_name") or baseline.get("import_path") or baseline.get("model_name"),
            "adapter": model_config.get("adapter") if model_config.get("adapter") is not None else baseline.get("adapter"),
            "model_hyper_params": dict(model_config.get("explicit_model_hyper_params") or model_config.get("model_hyper_params") or {}),
        },
        baseline_model_config=model_config,
        explicit_model_hyper_params=dict(model_config.get("explicit_model_hyper_params") or model_config.get("model_hyper_params") or {}),
        requested_budget=requested_budget,
        smoke=smoke,
    )
    return resolved.entry


def write_contract_for_task(
    *,
    base_dir: str,
    task_id: str,
    contract: BuildContract,
    name: str | None = None,
) -> Path:
    stem = str(name or contract.research_id or "build_contract").strip()
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    path = task_knowledge_dir(base_dir, task_id) / "build_contracts" / f"{safe_stem}.json"
    return contract.write_json(path)


def build_research_contract(
    *,
    base_dir: str,
    task_id: str,
    baseline: dict[str, Any],
    objective_metric: str,
    api_config: str = "",
    repo_dir: str | Path | None = None,
    diagnosis: dict[str, Any] | None = None,
    research_id: str | None = None,
    research_direction: dict[str, Any] | None = None,
    repair_budget: int | None = None,
    timeout_seconds: int = 900,
) -> BuildContract:
    task_config = _task_config(base_dir, task_id)
    rid = research_id or next_research_id(base_dir, task_id)
    active, source_ref = active_source_ref(base_dir=base_dir, task_id=task_id, repo_dir=repo_dir)
    baseline = dict(active or baseline or {})
    model_name = str(
        baseline.get("display_name")
        or baseline.get("best_model_name")
        or baseline.get("model_name")
        or baseline.get("import_path")
        or "baseline"
    )
    import_path = str(baseline.get("import_path") or (baseline.get("model_config") or {}).get("model_name") or model_name)
    source_binding = _load_source_binding(baseline, repo_dir)
    if not _has_verified_source_binding(source_binding):
        source_binding = dict(source_ref.get("source_binding") or {})
    agent_ablation = _agent_ablation_mode(base_dir, task_id)
    execution_authority = "repo_wide" if agent_ablation == "A4" else "source_binding"
    if execution_authority == "repo_wide":
        allowed_edit_files, allowed_new_file_roots = _repo_wide_paths(source_ref.get("source_checkout") or repo_dir)
    else:
        allowed_edit_files = list(source_binding.get("source_files") or [])
        allowed_new_file_roots = _new_file_roots_for_source_files(allowed_edit_files)
    edit_file = str(source_binding.get("entry_file") or (allowed_edit_files[0] if allowed_edit_files else ""))
    if not allowed_edit_files:
        raise RuntimeError(
            "verified source_binding is required before opening a research BuildContract; "
            f"model={model_name}, import_path={import_path}"
        )
    diagnosis_hint = ""
    if diagnosis:
        useful = diagnosis.get("usable_ablation_count")
        status = diagnosis.get("status")
        diagnosis_hint = f"Baseline diagnosis status={status}, usable_ablation_count={useful}."
    direction = dict(research_direction or {})
    direction_title = str(direction.get("terminal_display_title") or direction.get("idea_title") or "").strip()
    direction_hypothesis = str(direction.get("hypothesis") or "").strip()
    direction_hints: list[str] = []
    if direction:
        direction_hints = [
            "Scientist-Critic selected research direction is authoritative for this Research round.",
            f"Selected idea title: {direction_title or '<missing>'}.",
            f"Round role: {direction.get('round_role') or '<missing>'}.",
            f"Hypothesis: {direction_hypothesis or '<missing>'}.",
            f"Target mechanism: {direction.get('target_mechanism') or '<missing>'}.",
            f"Target code region: {direction.get('target_code_region') or '<missing>'}.",
            f"Failure to avoid: {direction.get('failure_to_avoid') or '<missing>'}.",
            f"History usage: {direction.get('history_usage') or '<missing>'}.",
            f"What this round teaches: {direction.get('what_this_round_will_teach') or '<missing>'}.",
        ]
    semantic_goal = (
        f"Implement the selected Scientist-Critic research direction for {model_name} "
        f"to improve {objective_metric} under the unchanged EvoCast task protocol."
        if direction
        else (
            f"Modify the active {model_name} source to improve "
            f"{objective_metric} under the unchanged EvoCast task protocol."
        )
    )
    hypothesis = (
        direction_hypothesis
        if direction_hypothesis
        else (
            f"A focused architecture change to {model_name} can reduce {objective_metric} while preserving "
            "the existing dataset, horizon, optimizer, scheduler, metric, and adapter protocol."
        )
    )
    check_files = list(dict.fromkeys(source_binding.get("source_files") or [edit_file]))
    if not check_files:
        check_files = [edit_file] if edit_file else []
    return BuildContract(
        research_id=rid,
        base_snapshot_id=str(source_ref["candidate_snapshot_id"]),
        base_candidate_id=str(baseline.get("candidate_id") or ""),
        base_source_ref=source_ref,
        source_mode="patch_current_best",
        execution_authority=execution_authority,
        allowed_edit_files=allowed_edit_files,
        semantic_goal=semantic_goal,
        hypothesis=hypothesis,
        target_model=model_name,
        research_intent=str(task_config.get("research_intent") or "").strip(),
        likely_entrypoints=list(dict.fromkeys([edit_file, *allowed_edit_files])),
        discovery_hints=[
            "Inspect the active model implementation in the candidate worktree.",
            "Implement the research change by editing allowed source files or by adding new source files under allowed_new_file_roots.",
            diagnosis_hint,
            *direction_hints,
        ],
        protected_globs=[] if execution_authority == "repo_wide" else [
            "evocast/**",
            "tests/**",
            "config/**",
            "ts_benchmark/evaluation/**",
            "ts_benchmark/pipeline.py",
        ],
        allowed_new_file_roots=allowed_new_file_roots,
        forbidden_globs=[] if execution_authority == "repo_wide" else [
            "dataset/**",
            "datasets/**",
            "result/**",
            "providers/**",
        ],
        required_behavior=[
            (
                "Execution authority is repo-wide for this ablation arm: any repository file inside the candidate workspace may be edited."
                if execution_authority == "repo_wide"
                else "Edit one or more verified source-binding files: " + ", ".join(allowed_edit_files) + "."
            ),
            (
                "New files may be created anywhere inside the candidate workspace repository."
                if execution_authority == "repo_wide"
                else "If a new helper/module file is needed, create it only under allowed_new_file_roots: "
                + (", ".join(allowed_new_file_roots) if allowed_new_file_roots else "<none>")
                + "."
            ),
            "Preserve the baseline constructor, forward signature, and forecast output protocol.",
            *(
                [
                    "Implement the selected Scientist-Critic research direction; do not replace it with an unrelated idea.",
                    "If the selected direction is source-infeasible, report that infeasibility in finish metadata instead of inventing a different direction.",
                ]
                if direction
                else []
            ),
        ],
        forbidden_behavior=[
            "Do not add network access or external service calls.",
        ],
        internal_check_commands=[syntax_check_command(path) for path in check_files],
        external_verification_commands=[syntax_check_command(path) for path in check_files],
        implementation_constraints=[
            (
                "Execution authority is repo-wide: existing files and new files may be changed anywhere inside the candidate workspace repository."
                if execution_authority == "repo_wide"
                else "Existing files may be modified only when listed in allowed_edit_files: "
                + ", ".join(allowed_edit_files)
                + "."
            ),
            (
                "Repository-root file creation is allowed for this ablation arm."
                if execution_authority == "repo_wide"
                else "New files may be created only under allowed_new_file_roots: "
                + (", ".join(allowed_new_file_roots) if allowed_new_file_roots else "<none>")
                + "."
            ),
            "Use the active current_best source as the source of truth for constructor and forward signatures.",
        ],
        metric_protocol={
            "objective_metric": objective_metric,
            "evaluation_stage": "build_mode",
            "api_config": api_config,
            "source_mode": "patch_current_best",
            "base_candidate_id": str(baseline.get("candidate_id") or ""),
            "base_source_ref": source_ref,
            "source_binding": source_binding,
            "execution_authority": execution_authority,
            "model_config": _effective_model_config_for_contract(
                base_dir=base_dir,
                task_id=task_id,
                baseline=baseline,
                requested_budget="unified",
                smoke=False,
            ),
            "baseline": baseline,
            "research_direction": direction,
        },
        timeout_seconds=timeout_seconds,
        # Research 构建只使用 VariantForge max_attempts 作为实现修复边界。
        # BuildContract.repair_budget 保留为兼容字段，但 Research 主链不能再在
        # VariantForge 外层额外套一层 repair 循环。
        repair_budget=0,
        failure_semantics={
            "round_kind": "research",
            "failed_build_counts_as_research_round": True,
        },
    )


def build_ablation_contract(
    *,
    base_dir: str,
    task_id: str,
    target: dict[str, Any],
    baseline: dict[str, Any],
    objective_metric: str,
    repo_dir: str | Path | None = None,
    repair_budget: int | None = None,
    timeout_seconds: int = 900,
) -> BuildContract:
    rid = next_ablation_id(base_dir, task_id)
    baseline, source_ref = baseline_diagnosis_source_ref(
        base_dir=base_dir,
        task_id=task_id,
        baseline=baseline,
        repo_dir=repo_dir,
    )
    target_id = str(target.get("target_id") or target.get("ablation_id") or rid)
    mechanism = str(target.get("mechanism_name") or target.get("mechanism_id") or target_id)
    model_name = str(baseline.get("display_name") or baseline.get("best_model_name") or baseline.get("model_name") or "baseline")
    import_path = str(baseline.get("import_path") or (baseline.get("model_config") or {}).get("model_name") or model_name)
    exact_target = compile_exact_ablation_target(
        target,
        repo_dir=repo_dir or repo_root(),
        source_checkout=str(source_ref.get("source_checkout") or ""),
    )
    edit_spec = dict(target.get("edit_spec") or {})
    evidence_files = [str(item) for item in list(target.get("evidence_files") or []) if str(item).strip()]
    if edit_spec.get("target_file"):
        evidence_files.insert(0, str(edit_spec.get("target_file")))
    edit_file = str(exact_target["target_file"])
    source_binding = _load_source_binding(baseline, repo_dir)
    if not _has_verified_source_binding(source_binding):
        source_binding = dict(source_ref.get("source_binding") or {})
    allowed_edit_files = list(source_binding.get("source_files") or [])
    if edit_file and edit_file not in allowed_edit_files:
        allowed_edit_files.insert(0, edit_file)
    allowed_edit_files = list(dict.fromkeys([path for path in allowed_edit_files if str(path).strip()]))
    if not allowed_edit_files:
        raise RuntimeError(
            "verified source_binding is required before opening an ablation BuildContract; "
            f"model={model_name}, import_path={import_path}, target_file={edit_file}"
        )
    evaluation_stage = _normalize_evaluation_stage(
        target.get("evaluation_stage"),
        build_mode=task_build_mode(base_dir, task_id),
    )
    ablation_intent = str(
        exact_target.get("ablation_intent")
        or exact_target.get("replacement_intent")
        or target.get("exact_edit_intent")
        or target.get("causal_variable")
        or mechanism
    ).strip()
    model_config = _effective_model_config_for_contract(
        base_dir=base_dir,
        task_id=task_id,
        baseline=baseline,
        requested_budget="unified",
        smoke=evaluation_stage == "smoke",
    )
    return BuildContract(
        research_id=rid,
        base_snapshot_id=str(source_ref["candidate_snapshot_id"]),
        base_candidate_id=str(baseline.get("candidate_id") or ""),
        base_source_ref=source_ref,
        source_mode="patch_current_best",
        allowed_edit_files=allowed_edit_files,
        semantic_goal=f"Build mechanism ablation {target_id}: {mechanism}.",
        hypothesis=str(target.get("diagnosis_question") or target.get("question") or target.get("exact_edit_intent") or mechanism),
        target_model=model_name,
        research_intent=str(_task_config(base_dir, task_id).get("research_intent") or "").strip(),
        likely_entrypoints=list(dict.fromkeys(evidence_files + [edit_file, *allowed_edit_files])),
        discovery_hints=[
            str(target.get("causal_variable") or ""),
            ablation_intent,
            "Implement the ablated model by editing one or more verified source-binding files.",
            "All source files provided to the coding backend are editable. Preserve runtime protocols even when an optional replacement hint is unsafe.",
            exact_target_instruction(exact_target),
        ],
        protected_globs=[
            "evocast/**",
            "tests/**",
            "config/**",
            "ts_benchmark/evaluation/**",
            "ts_benchmark/pipeline.py",
        ],
        allowed_new_file_roots=[],
        forbidden_globs=["dataset/**", "datasets/**", "result/**", "providers/**"],
        required_behavior=[
            "Edit one or more verified source-binding files: " + ", ".join(allowed_edit_files) + ".",
            "The ablation removes or bypasses the named causal mechanism while preserving the task protocol.",
            "Modify code at or near anchor_text; replacement_pseudocode is only an optional hint, not a literal requirement.",
            "Preserve constructor signatures, forward input signatures, adapter return protocol, tensor shapes, dtype/device, and state keys consumed later in forward; canonical smoke verifies the live runtime path.",
        ],
        forbidden_behavior=[
            "Do not change dataset, horizon, metric, optimizer, scheduler, or training policy.",
            "Do not edit tests or EvoCast runtime policy code.",
        ],
        internal_check_commands=[syntax_check_command(path) for path in allowed_edit_files] + [model_entry_contract_command(edit_file)],
        external_verification_commands=[syntax_check_command(path) for path in allowed_edit_files] + [model_entry_contract_command(edit_file)],
        implementation_constraints=[
            "Edit only these verified source-binding files unless a future contract lists more files: "
            + ", ".join(allowed_edit_files)
            + ".",
            f"Target mechanism: {mechanism}.",
            ablation_intent,
            str(edit_spec.get("shape_invariant_argument") or ""),
            f"Intent-anchor target file: {exact_target['target_file']}",
            f"Anchor text identifying the local edit area:\n{exact_target['anchor_text']}",
            "Do not remove auxiliary return values that adapters consume.",
            "Do not make paired state dictionaries inconsistent across normalize/denormalize or encode/decode stages.",
            "Prefer the smallest local patch that implements the ablation and passes canonical smoke/build-mode metric execution.",
        ],
        metric_protocol={
            "objective_metric": objective_metric,
            "evaluation_stage": evaluation_stage,
            "source_mode": "patch_current_best",
            "base_candidate_id": str(baseline.get("candidate_id") or ""),
            "base_source_ref": source_ref,
            "source_binding": source_binding,
            "model_config": model_config,
            "baseline": baseline,
            "target": target,
            "exact_ablation_target": exact_target,
            "ablation_contract_mode": "intent_anchor_runtime_protocol",
        },
        timeout_seconds=timeout_seconds,
        repair_budget=research_repair_budget(base_dir) if repair_budget is None else int(repair_budget),
        failure_semantics={
            "round_kind": "baseline_diagnosis_ablation",
            "ablation_id": target_id,
            "exact_ablation_target": exact_target,
            "failed_build_counts_as_research_round": True,
        },
    )
