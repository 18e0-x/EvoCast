"""Deterministic model-structure understanding tools.

These tools produce source-grounded facts for the lead agent.  They are not a
replacement for scientific judgment; they provide the cockpit instruments.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import pandas as pd
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evocast.domain.knowledge_paths import shared_knowledge_dir, task_knowledge_dir
from evocast.domain.atomic_io import atomic_write_json
from evocast.research.objective_boundary import discover_objective_boundaries
from evocast.research.model_registry import build_registry
from evocast.state.runtime.store import load_runtime_state
from evocast.state.domain_store import load_task_config
from evocast.harness.permissions import PROJECT_ROOT, assert_variant_path, normalize_repo_path
from evocast.harness.session import AgentSession
from evocast.runners.tfb_pipeline_runner import build_run_configs


class ModelStructureError(ValueError):
    """Raised when a model structure request is invalid."""


def _purge_workspace_shadowed_repo_modules() -> None:
    """Drop cached benchmark imports before baseline structure analysis.

    Workspace variants intentionally shadow ``ts_benchmark`` while a round entry
    is imported.  Those package modules can remain cached after the workspace
    path is removed, so later baseline structure analysis may import a partial
    workspace package instead of the repository package.  Structure analysis is
    always about the canonical baseline unless variant_path is explicit.
    """

    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "ts_benchmark" or name.startswith("ts_benchmark."):
            sys.modules.pop(name, None)


@dataclass
class SourceClass:
    class_name: str
    module: str
    file_path: str
    source: str
    tree: ast.ClassDef
    bases: List[str] = field(default_factory=list)


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_json(path, payload, ensure_ascii=False, default=str)


def _hash_file(path: str) -> str:
    try:
        data = Path(path).read_bytes()
    except Exception:
        return ""
    return hashlib.sha1(data).hexdigest()


def _model_key_aliases(value: str) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    aliases: List[str] = []

    def add(item: str) -> None:
        item = str(item or "").strip()
        if item and item not in aliases:
            aliases.append(item)

    add(raw)
    if raw.lower().startswith("baseline_001_"):
        add(raw[len("baseline_001_"):])
    if raw.lower().startswith("baseline_") and raw.count("_") >= 2:
        add(raw.split("_", 2)[-1])
    return aliases


def _resolve_model(session: AgentSession, args: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    variant_path = str(args.get("variant_path") or "").strip()
    if variant_path:
        rel = assert_variant_path(variant_path)
        module = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel.replace("/", ".")
        return Path(rel).stem, f"{module}.Model", args.get("adapter")

    model_name = str(args.get("model_name") or "").strip()
    model_key = str(args.get("model_key") or "").strip()
    if model_name:
        if "." in model_name:
            return model_key or model_name.rsplit(".", 1)[-1], model_name, args.get("adapter")
        requested = model_name.lower()
        for spec in build_registry(verify=False):
            spec_key = str(spec.get("model_key") or "")
            import_path = str(spec.get("import_path") or "")
            aliases = {spec_key.lower(), import_path.rsplit(".", 1)[-1].lower() if import_path else ""}
            if requested in aliases or (model_key and spec_key.lower() == model_key.lower()):
                return spec_key, import_path, spec.get("adapter")
        if model_key:
            requested_key = model_key.lower()
            for spec in build_registry(verify=False):
                spec_key = str(spec.get("model_key") or "")
                if spec_key.lower() == requested_key:
                    return spec_key, str(spec.get("import_path") or ""), spec.get("adapter")
        raise ModelStructureError(
            f"Cannot resolve dotless model_name {model_name!r}. "
            "Use model_key for a registered baseline or provide a fully qualified import path."
        )
    if not model_name and not model_key:
        try:
            state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
        except Exception:
            state = None
        for candidate in ((state.current_best, state.baseline, state.provisional_best) if state else ()):
            if not candidate:
                continue
            config = dict(candidate.model_config or {})
            candidate_model_name = str(config.get("model_name") or candidate.model_name or "").strip()
            candidate_key = str(candidate.display_name or candidate.model_name or candidate.candidate_id or "").strip()
            if candidate_model_name or candidate_key:
                return _resolve_model(
                    session,
                    {
                        "model_key": candidate_key,
                        "model_name": candidate_model_name,
                        "adapter": config.get("adapter") if config.get("adapter") is not None else candidate.adapter,
                    },
                )

    if model_key:
        requested_aliases = {item.lower() for item in _model_key_aliases(model_key)}
        for spec in build_registry(verify=False):
            spec_key = str(spec.get("model_key") or "")
            import_path = str(spec.get("import_path") or "")
            aliases = {spec_key.lower()}
            if import_path:
                aliases.add(import_path.lower())
                aliases.add(import_path.rsplit(".", 1)[-1].lower())
            if aliases & requested_aliases:
                return spec_key, str(spec.get("import_path") or ""), spec.get("adapter")
        state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
        for candidate in (state.baseline, state.current_best, state.provisional_best):
            if not candidate or not candidate.candidate_id:
                continue
            config = dict(candidate.model_config or {})
            aliases = {
                str(candidate.candidate_id or ""),
                str(candidate.display_name or ""),
                str(candidate.model_name or ""),
                str(config.get("model_name") or ""),
            }
            if set(aliases) & set(_model_key_aliases(model_key)):
                return (
                    model_key,
                    str(config.get("model_name") or candidate.model_name or ""),
                    config.get("adapter") if config.get("adapter") is not None else candidate.adapter,
                )
    raise ModelStructureError("requires model_name, model_key, or variant_path")


def _import_class(import_path: str) -> Tuple[Any, Any]:
    normalized = import_path
    if normalized.startswith("global."):
        normalized = normalized[len("global.") :]
    if "." not in normalized:
        for spec in build_registry(verify=False):
            if str(spec.get("model_key") or "").lower() == normalized.lower():
                normalized = str(spec.get("import_path") or "")
                if normalized:
                    break
        if "." not in normalized:
            raise ModelStructureError(
                f"Cannot import from dotless path {import_path!r}. "
                f"Provide a fully qualified import path like 'ts_benchmark.baselines.dtaf.DTAF'."
            )
    module_name, class_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return module, getattr(module, class_name)


def _resolve_ast_ref(module: Any, node: ast.AST) -> Optional[Any]:
    """Resolve a constructor reference from an adapter module's globals."""
    if isinstance(node, ast.Name):
        return getattr(module, node.id, None)
    if isinstance(node, ast.Attribute):
        parent = _resolve_ast_ref(module, node.value)
        return getattr(parent, node.attr, None) if parent is not None else None
    return None


def _init_model_constructors(module: Any, classes: List[SourceClass]) -> List[Dict[str, Any]]:
    """Find inner model constructors returned by adapter _init_model methods.

    TFB adapters usually construct the real torch module inside _init_model(),
    while the registry points at the adapter wrapper.  This function handles the
    common patterns:
      return Model(self.config)
      inner = DUETModel(self.config); return inner
    """
    constructors: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(call: ast.Call, item: SourceClass, via: str) -> None:
        obj = _resolve_ast_ref(module, call.func)
        if obj is None or not inspect.isclass(obj):
            return
        key = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__name__', '')}"
        if not key or key in seen:
            return
        seen.add(key)
        constructors.append(
            {
                "constructor": _unparse(call.func),
                "class_path": key,
                "class_object": obj,
                "defined_in": item.class_name,
                "source_file": _repo_rel(item.file_path),
                "via": via,
                "arguments": [_unparse(arg) for arg in call.args],
                "keywords": [kw.arg for kw in call.keywords if kw.arg],
            }
        )

    for item in classes:
        methods = [n for n in item.tree.body if isinstance(n, ast.FunctionDef) and n.name == "_init_model"]
        for fn in methods:
            assigned_calls: Dict[str, ast.Call] = {}
            for stmt in ast.walk(fn):
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            assigned_calls[target.id] = stmt.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    assigned_calls[stmt.target.id] = stmt.value

            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Return):
                    continue
                if isinstance(stmt.value, ast.Call):
                    add(stmt.value, item, "return_call")
                elif isinstance(stmt.value, ast.Name) and stmt.value.id in assigned_calls:
                    add(assigned_calls[stmt.value.id], item, f"return_variable:{stmt.value.id}")
    return constructors


def _class_from_source(cls: Any) -> Optional[SourceClass]:
    try:
        source = inspect.getsource(cls)
        file_path = inspect.getsourcefile(cls) or inspect.getfile(cls)
        module = getattr(cls, "__module__", "")
    except Exception:
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    class_node = next((node for node in tree.body if isinstance(node, ast.ClassDef)), None)
    if class_node is None:
        return None
    return SourceClass(
        class_name=getattr(cls, "__name__", class_node.name),
        module=module,
        file_path=str(Path(file_path).resolve()),
        source=source,
        tree=class_node,
        bases=[_unparse(base) for base in class_node.bases],
    )


def _source_chain(cls: Any) -> List[SourceClass]:
    items: List[SourceClass] = []
    for base in inspect.getmro(cls):
        if base is object:
            continue
        item = _class_from_source(base)
        if item and item.file_path.startswith(str(PROJECT_ROOT)):
            if item.file_path not in {existing.file_path for existing in items} or item.class_name not in {existing.class_name for existing in items}:
                items.append(item)
    return items


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _attr_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Attribute):
        parent = _attr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return None


def _is_self_attr(node: ast.AST) -> Optional[str]:
    if not isinstance(node, ast.Attribute):
        return None
    parts: List[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "self":
        return ".".join(reversed(parts))
    return None


def _assigned_self_attr(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            attr = _is_self_attr(target)
            if attr:
                return attr
    if isinstance(node, ast.AnnAssign):
        return _is_self_attr(node.target)
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _attr_name(node.func) or ""
    return ""


def _constructor_class_from_value(module: Any, value: Optional[ast.AST]) -> Optional[Any]:
    if not isinstance(value, ast.Call):
        return None
    obj = _resolve_ast_ref(module, value.func)
    if obj is None or not inspect.isclass(obj):
        return None
    item = _class_from_source(obj)
    if item is None or not item.file_path.startswith(str(PROJECT_ROOT)):
        return None
    return obj


def _extract_components(
    classes: List[SourceClass],
    prefix: str = "",
    *,
    recursive_depth: int = 0,
    max_recursive_depth: int = 2,
    visited_classes: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    components: Dict[str, Dict[str, Any]] = {}
    visited = set(visited_classes or set())
    for item in reversed(classes):
        try:
            item_module = importlib.import_module(item.module)
        except Exception:
            item_module = None
        for fn in [n for n in item.tree.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
            for node in ast.walk(fn):
                attr = _assigned_self_attr(node)
                if not attr:
                    continue
                name = f"{prefix}{attr}" if prefix else attr
                value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
                call = _call_name(value) if value is not None else ""
                constructor = call or (_unparse(value) if value is not None else "")
                kind = "attribute"
                if "nn." in call or call.startswith("torch.nn"):
                    kind = "nn_module"
                if call.endswith("Parameter") or "nn.Parameter" in call:
                    kind = "parameter"
                if call.endswith("ModuleList") or call.endswith("Sequential"):
                    kind = "container"
                if constructor and constructor[:1].isupper() and constructor not in {"False", "True", "None"} and kind == "attribute":
                    kind = "module_like"
                components[name] = {
                    "name": name,
                    "local_name": attr,
                    "constructor": constructor,
                    "kind": kind,
                    "defined_in": item.class_name,
                    "source_file": _repo_rel(item.file_path),
                }
                if item_module is None or recursive_depth >= max_recursive_depth:
                    continue
                child_cls = _constructor_class_from_value(item_module, value)
                if child_cls is None:
                    continue
                child_key = f"{getattr(child_cls, '__module__', '')}.{getattr(child_cls, '__name__', '')}"
                if not child_key or child_key in visited:
                    continue
                child_classes = _source_chain(child_cls)
                if not child_classes:
                    continue
                child_components = _extract_components(
                    child_classes,
                    prefix=f"{name}.",
                    recursive_depth=recursive_depth + 1,
                    max_recursive_depth=max_recursive_depth,
                    visited_classes=visited | {child_key},
                )
                for child in child_components:
                    components.setdefault(str(child.get("name") or ""), child)
    return list(components.values())


def _extract_hparams(classes: List[SourceClass]) -> Dict[str, Any]:
    reads: Dict[str, Dict[str, Any]] = {}
    constructor_params: Dict[str, Any] = {}
    for item in classes:
        for fn in [n for n in item.tree.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]:
            for arg in fn.args.args:
                if arg.arg != "self":
                    constructor_params.setdefault(arg.arg, None)
            defaults = list(fn.args.defaults)
            args = list(fn.args.args)
            for arg, default in zip(args[-len(defaults):] if defaults else [], defaults):
                if arg.arg != "self":
                    constructor_params[arg.arg] = _unparse(default)
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute):
                    text = _unparse(node)
                    if text.startswith("configs."):
                        field = text.split(".", 1)[1]
                        reads.setdefault(field, {"field": field, "access": text, "source": item.class_name})
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                    receiver = _unparse(node.func.value)
                    if receiver in {"kwargs", "model_hyper_params", "hparams"} and node.args:
                        key = _literal_string(node.args[0])
                        if key:
                            reads.setdefault(key, {"field": key, "access": f"{receiver}.get", "source": item.class_name})
    required = sorted(set(reads) | {k for k in constructor_params if constructor_params[k] is None})
    return {
        "constructor_params": constructor_params,
        "hparam_reads": sorted(reads.values(), key=lambda x: x["field"]),
        "required_or_observed": required,
        "task_aligned_fields": [f for f in required if f in {"seq_len", "pred_len", "horizon", "enc_in", "c_out", "patch_len", "input_chunk_length", "output_chunk_length"}],
    }


def _literal_string(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _extract_forward(classes: List[SourceClass], prefix: str = "") -> Dict[str, Any]:
    methods: Dict[str, ast.FunctionDef] = {}
    method_sources: Dict[str, str] = {}
    for item in classes:
        for fn in [n for n in item.tree.body if isinstance(n, ast.FunctionDef)]:
            methods[fn.name] = fn
            method_sources[fn.name] = item.class_name

    visited: set[str] = set()
    calls: List[Dict[str, Any]] = []

    def visit_method(name: str) -> None:
        if name in visited or name not in methods:
            return
        visited.add(name)
        for node in ast.walk(methods[name]):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            self_attr = _is_self_attr(fn)
            call_text = _unparse(fn)
            if self_attr:
                calls.append({"from_method": name, "call": f"{prefix}{self_attr}" if prefix else self_attr, "source": method_sources.get(name)})
                if self_attr in methods:
                    visit_method(self_attr)
            elif call_text.startswith(("torch.", "F.", "nn.")):
                calls.append({"from_method": name, "call": call_text, "source": method_sources.get(name), "kind": "external_op"})

    visit_method("forward")
    signature = ""
    if "forward" in methods:
        signature = f"forward({_signature_from_ast(methods['forward'])})"
    return {"forward_signature": signature, "called_components": calls, "parsed_methods": sorted(visited)}


def _signature_from_ast(fn: ast.FunctionDef) -> str:
    parts = [arg.arg for arg in fn.args.args]
    if fn.args.vararg:
        parts.append("*" + fn.args.vararg.arg)
    parts.extend(arg.arg for arg in fn.args.kwonlyargs)
    if fn.args.kwarg:
        parts.append("**" + fn.args.kwarg.arg)
    return ", ".join(parts)


def _baseline_hparams_for_model(session: AgentSession, model_key: str, import_path: str) -> Dict[str, Any]:
    try:
        state = load_runtime_state(session.base_dir, session.task_id, auto_migrate=False)
    except Exception:
        return {}
    aliases = {str(item).strip().lower() for item in _model_key_aliases(model_key)}
    if import_path:
        aliases.add(str(import_path).strip().lower())
        aliases.add(str(import_path).rsplit(".", 1)[-1].strip().lower())
    for candidate in (state.baseline, state.current_best):
        if not candidate or not candidate.candidate_id:
            continue
        config = dict(candidate.model_config or {})
        candidate_aliases = {
            str(candidate.candidate_id or "").strip().lower(),
            str(candidate.display_name or "").strip().lower(),
            str(candidate.model_name or "").strip().lower(),
            str(config.get("model_name") or "").strip().lower(),
        }
        if aliases and not (aliases & candidate_aliases):
            continue
        hparams = dict(config.get("model_hyper_params") or {})
        if hparams:
            return hparams
    return {}


def _task_config(session: AgentSession) -> Dict[str, Any]:
    return load_task_config(session.base_dir, session.task_id)


def _compiled_config(session: AgentSession, task_config: Dict[str, Any]) -> Dict[str, Any]:
    config_path = str(task_config.get("config_path") or "").strip()
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _infer_task_value_columns(task_config: Dict[str, Any]) -> List[str]:
    semantics = dict(task_config.get("task_semantics") or {})
    target_columns = list(semantics.get("target_columns") or [])
    if target_columns:
        return [str(col) for col in target_columns]
    dataset_path = str(task_config.get("dataset_path") or semantics.get("dataset_path") or "").strip()
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            columns = list(pd.read_csv(path, nrows=1).columns)
            time_col = str(semantics.get("time_col") or "date")
            value_columns = [col for col in columns if str(col) != time_col]
            if value_columns:
                return [str(col) for col in value_columns]
        except Exception:
            pass
    return []


def _infer_probe_channels(session: AgentSession, task_config: Dict[str, Any]) -> int:
    del session
    value_columns = _infer_task_value_columns(task_config)
    return max(1, len(value_columns))


def _runtime_task_contract(session: AgentSession, model_key: str, import_path: str, args: Dict[str, Any]) -> Dict[str, Any]:
    task_config = _task_config(session)
    semantics = dict(task_config.get("task_semantics") or {})
    hparams = dict(args.get("model_hyper_params") or {})
    value_columns = _infer_task_value_columns(task_config)
    runtime_channels = max(1, len(value_columns))
    baseline_hparams = _baseline_hparams_for_model(session, model_key, import_path)
    static_channels = {
        key: _safe_int_scalar(baseline_hparams.get(key), runtime_channels)
        for key in ("enc_in", "dec_in", "c_out")
        if baseline_hparams.get(key) is not None
    }
    effective_channels = {
        key: _safe_int_scalar(hparams.get(key), runtime_channels)
        for key in ("enc_in", "dec_in", "c_out")
        if hparams.get(key) is not None
    }
    mismatch_keys = [
        key
        for key, value in static_channels.items()
        if value != runtime_channels
    ]
    seq_len = _safe_int_scalar(task_config.get("seq_len") or hparams.get("seq_len"), 0)
    horizon = _safe_int_scalar(task_config.get("horizon") or hparams.get("horizon") or hparams.get("pred_len"), 0)
    warnings = []
    if mismatch_keys:
        warnings.append(
            "Static baseline hyperparameters disagree with runtime task channels. "
            "Forecasting adapters may mutate config.enc_in/dec_in/c_out from train_data.shape[1] before _init_model; "
            "derive channel dimensions from runtime input x.shape[-1] or task_contract.runtime_channels, not from default source constants."
        )
    return {
        "task_mode": semantics.get("task_mode"),
        "dataset_path": str(task_config.get("dataset_path") or semantics.get("dataset_path") or ""),
        "time_col": semantics.get("time_col"),
        "value_columns": value_columns,
        "runtime_channels": runtime_channels,
        "seq_len": seq_len or None,
        "horizon": horizon or None,
        "expected_model_input_shape": [None, seq_len or "seq_len", runtime_channels],
        "expected_model_output_shape": [None, horizon or "horizon", runtime_channels],
        "baseline_static_channels": static_channels,
        "effective_probe_channels": effective_channels,
        "channel_source": "dataset_value_columns_or_target_columns",
        "warnings": warnings,
    }


def _safe_int_scalar(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if not isinstance(item, (list, tuple, dict))), None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _align_hparams_for_structure_probe(session: AgentSession, model_key: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    aligned = dict(hparams or {})
    task_config = _task_config(session)
    compiled = _compiled_config(session, task_config)
    try:
        spec = next(
            (item for item in build_registry(verify=False) if str(item.get("model_key") or "").lower() == str(model_key or "").lower()),
            {},
        )
        if spec:
            from evocast.runners.baseline_runner import align_model_hparams_to_task

            aligned, _ = align_model_hparams_to_task(spec, compiled, aligned)
    except Exception:
        pass
    seq_len = task_config.get("seq_len") or task_config.get("horizon")
    pred_len = task_config.get("horizon")
    channels = _infer_probe_channels(session, task_config)
    if seq_len is not None:
        aligned.setdefault("seq_len", seq_len)
    if pred_len is not None:
        aligned.setdefault("pred_len", pred_len)
        aligned.setdefault("horizon", pred_len)
    aligned.setdefault("label_len", max(1, _safe_int_scalar(aligned.get("seq_len"), 2) // 2))
    aligned.setdefault("enc_in", channels)
    aligned.setdefault("dec_in", channels)
    aligned.setdefault("c_out", channels)
    aligned.setdefault("use_gpu", False)
    return aligned


def _merged_probe_hparams(args: Dict[str, Any]) -> Dict[str, Any]:
    probe_args = dict(args.get("shape_probe") or {})
    hparams = dict(args.get("model_hyper_params") or {})
    if not probe_args:
        return hparams
    seq_len = probe_args.get("seq_len") or probe_args.get("input_chunk_length")
    pred_len = probe_args.get("pred_len") or probe_args.get("horizon") or probe_args.get("output_chunk_length")
    channels = probe_args.get("channels") or probe_args.get("enc_in") or probe_args.get("c_out")
    defaults = {
        "seq_len": seq_len,
        "horizon": pred_len,
        "pred_len": pred_len,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "num_epochs": 1,
        "batch_size": probe_args.get("batch") or 2,
        "use_gpu": False,
    }
    for key, value in defaults.items():
        if value is not None:
            hparams.setdefault(key, value)
    hparams.update(dict(probe_args.get("model_hyper_params") or {}))
    return hparams


def _structure_args_with_runtime_hparams(session: AgentSession, args: Dict[str, Any], model_key: str, import_path: str) -> Dict[str, Any]:
    merged = dict(args or {})
    explicit = dict(merged.get("model_hyper_params") or {})
    baseline_hparams = _baseline_hparams_for_model(session, model_key, import_path)
    combined = {**baseline_hparams, **explicit}
    merged["model_hyper_params"] = _align_hparams_for_structure_probe(session, model_key, combined)
    return merged


def _runtime_introspection(cls: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    # Instantiating arbitrary forecasting models is fragile; keep this best-effort.
    hparams = _merged_probe_hparams(args)
    result: Dict[str, Any] = {
        "status": "not_run",
        "modules": [],
        "parameters": [],
        "buffers": [],
        "active_path_probe": {"status": "not_run"},
        "error": "",
        "task_name": str(hparams.get("task_name") or ""),
    }
    if not hparams:
        return result
    try:
        model = cls(**hparams)
        result["status"] = "ok"
        result["modules"] = [
            _runtime_module_record(name, module)
            for name, module in list(model.named_modules())[:300]
            if name
        ]
        result["parameters"] = [
            {"name": name, "shape": list(param.shape), "requires_grad": bool(param.requires_grad)}
            for name, param in list(model.named_parameters())[:300]
        ]
        result["buffers"] = [
            {"name": name, "shape": list(buf.shape)}
            for name, buf in list(model.named_buffers())[:100]
        ]
        try:
            import torch

            probe_args = dict(args.get("shape_probe") or {})
            requested_device = str(probe_args.get("device") or "cpu")
            device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
            if hasattr(model, "to"):
                model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            result["active_path_probe"] = _runtime_active_path_probe(model, model, hparams, device, torch)
        except Exception as probe_exc:
            result["active_path_probe"] = {"status": "error", "error": f"{type(probe_exc).__name__}: {probe_exc}"}
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _runtime_introspection_from_adapter(cls: Any, args: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "not_run",
        "modules": [],
        "parameters": [],
        "buffers": [],
        "active_path_probe": {"status": "not_run"},
        "error": "",
    }
    try:
        hparams = _merged_probe_hparams(args)
        result["task_name"] = str(hparams.get("task_name") or "")
        adapter = cls(**hparams)
        probe_args = dict(args.get("shape_probe") or {})
        requested_device = str(probe_args.get("device") or "cpu")
        try:
            import torch

            device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
            if hasattr(adapter, "device"):
                adapter.device = device
        except Exception:
            device = None
        if not hasattr(adapter, "model") and hasattr(adapter, "_init_model"):
            adapter.model = adapter._init_model()
        inner = getattr(adapter, "model", None)
        if inner is None:
            result["status"] = "error"
            result["error"] = "adapter has no self.model after construction/_init_model"
            return result
        if device is not None and hasattr(inner, "to"):
            inner.to(device)
        if hasattr(inner, "eval"):
            inner.eval()
        result["status"] = "ok"
        result["adapter_inner_model"] = {
            "type": type(inner).__name__,
            "module": getattr(type(inner), "__module__", ""),
        }
        result["inner_modules"] = []
        result["inner_parameters"] = []
        result["inner_buffers"] = []
        if hasattr(inner, "named_modules"):
            result["inner_modules"] = [
                _runtime_module_record(f"model.{name}", module)
                for name, module in list(inner.named_modules())[:300]
                if name
            ]
        if hasattr(inner, "named_parameters"):
            result["inner_parameters"] = [
                {"name": f"model.{name}", "shape": list(param.shape), "requires_grad": bool(param.requires_grad)}
                for name, param in list(inner.named_parameters())[:300]
            ]
        if hasattr(inner, "named_buffers"):
            result["inner_buffers"] = [
                {"name": f"model.{name}", "shape": list(buf.shape)}
                for name, buf in list(inner.named_buffers())[:100]
            ]
        if device is not None:
            try:
                import torch

                active = _runtime_active_path_probe(adapter, inner, hparams, device, torch)
                active_names = [str(name or "") for name in list(active.get("called_module_names") or [])]
                active["called_inner_module_names"] = ["model." + name for name in active_names if name]
                result["active_path_probe"] = active
            except Exception as probe_exc:
                result["active_path_probe"] = {"status": "error", "error": f"{type(probe_exc).__name__}: {probe_exc}"}
        result["modules"].extend(result["inner_modules"])
        result["parameters"].extend(result["inner_parameters"])
        result["buffers"].extend(result["inner_buffers"])
    except Exception as exc:
        result["inner_introspection_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _tfb_config_for_structure_probe(session: AgentSession) -> Dict[str, Any]:
    task_config = _task_config(session)
    compiled = _compiled_config(session, task_config)
    return compiled if isinstance(compiled, dict) else {}


def _model_entry_for_structure_probe(import_path: str, adapter: Optional[str], args: Dict[str, Any]) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "model_name": import_path,
        "model_hyper_params": dict(args.get("model_hyper_params") or {}),
    }
    if adapter is not None:
        entry["adapter"] = adapter
    return entry


def _tensor_shape_tree(value: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_tensor_shape_tree(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_tensor_shape_tree(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(key): _tensor_shape_tree(item) for key, item in value.items()}}
    return {"kind": type(value).__name__}


def _runtime_module_record(name: str, module: Any) -> Dict[str, Any]:
    module_cls = type(module)
    class_name = getattr(module_cls, "__name__", "")
    class_module = getattr(module_cls, "__module__", "")
    source_file = ""
    try:
        source_path = inspect.getsourcefile(module_cls) or inspect.getfile(module_cls)
        if source_path:
            source_file = _repo_rel(source_path)
    except Exception:
        source_file = ""
    return {
        "name": name,
        "type": class_name,
        "class_name": class_name,
        "class_module": class_module,
        "class_path": ".".join(part for part in (class_module, class_name) if part),
        "source_file": source_file,
    }


def _runtime_active_path_probe(model: Any, target: Any, hparams: Dict[str, Any], device: Any, torch_module: Any) -> Dict[str, Any]:
    """Observe which runtime modules are actually called for the current task probe."""
    if target is None or not hasattr(target, "named_modules"):
        return {"status": "not_run", "reason": "target has no named_modules"}
    observed: Dict[str, Dict[str, Any]] = {}
    handles = []

    def make_pre_hook(name: str):
        def pre_hook(_module: Any, inputs: Tuple[Any, ...]) -> None:
            item = observed.setdefault(name, {"name": name, "calls": 0})
            item["calls"] = int(item.get("calls") or 0) + 1
            item["input"] = _tensor_shape_tree(inputs)

        return pre_hook

    def make_hook(name: str):
        def hook(_module: Any, _inputs: Tuple[Any, ...], output: Any) -> None:
            item = observed.setdefault(name, {"name": name, "calls": 0})
            item["output"] = _tensor_shape_tree(output)

        return hook

    try:
        for name, module in list(target.named_modules())[:300]:
            if not name:
                continue
            handles.append(module.register_forward_pre_hook(make_pre_hook(str(name))))
            handles.append(module.register_forward_hook(make_hook(str(name))))
        batch = _safe_int_scalar(hparams.get("batch_size"), 2)
        seq_len = _safe_int_scalar(hparams.get("seq_len") or hparams.get("input_chunk_length"), 96)
        pred_len = _safe_int_scalar(hparams.get("pred_len") or hparams.get("horizon") or hparams.get("output_chunk_length"), 96)
        label_len = _safe_int_scalar(hparams.get("label_len"), max(1, seq_len // 2))
        channels = _safe_int_scalar(hparams.get("enc_in") or hparams.get("c_out") or hparams.get("dec_in"), 1)
        target_len = label_len + pred_len
        x = torch_module.zeros(batch, seq_len, channels, device=device)
        target_tensor = torch_module.zeros(batch, target_len, channels, device=device)
        input_mark = torch_module.zeros(batch, seq_len, 4, device=device)
        target_mark = torch_module.zeros(batch, target_len, 4, device=device)
        with torch_module.no_grad():
            if hasattr(model, "_process"):
                output = model._process(x, target_tensor, input_mark, target_mark)
                output = output.get("output") if isinstance(output, dict) else output
            elif hasattr(model, "forward"):
                output = model(x)
            else:
                return {"status": "error", "error": "model has no _process or forward method"}
        if isinstance(output, tuple):
            output = output[0]
        called = sorted(observed.values(), key=lambda item: str(item.get("name") or ""))
        return {
            "status": "ok",
            "called_modules": called[:300],
            "called_module_count": len(called),
            "called_module_names": [str(item.get("name") or "") for item in called[:300]],
            "input_shape": list(x.shape),
            "output_shape": list(output.shape) if hasattr(output, "shape") else str(type(output).__name__),
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


def _runtime_introspection_via_tfb_loader(
    session: AgentSession,
    *,
    import_path: str,
    adapter: Optional[str],
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Instantiate through the same TFB model loader used by experiments.

    Registered models with adapters often expose a raw torch class in the
    registry. Direct construction with flattened benchmark hparams is not the
    real runtime contract; the TFB loader first applies adapter-required hparam
    mapping and then constructs the adapter object. This probe mirrors that
    path so safe_fit_points are grounded in the current task instance.
    """
    hparams = _merged_probe_hparams(args)
    result: Dict[str, Any] = {
        "status": "not_run",
        "modules": [],
        "parameters": [],
        "buffers": [],
        "error": "",
        "task_name": str(hparams.get("task_name") or ""),
    }
    tfb_config = _tfb_config_for_structure_probe(session)
    if not tfb_config:
        result["status"] = "error"
        result["error"] = "compiled_config unavailable for TFB loader structure probe"
        return result
    try:
        import torch
        from ts_benchmark.models.model_loader import get_models

        _, model_config, _ = build_run_configs(
            tfb_config,
            [_model_entry_for_structure_probe(import_path, adapter, args)],
            save_path="EvoCast_structure_probe",
            seed=2021,
        )
        factory = get_models(model_config)[0]
        hparams = dict(getattr(factory, "model_hyper_params", {}) or hparams)
        result["task_name"] = str(hparams.get("task_name") or result.get("task_name") or "")
        model = factory()
        requested_device = str((args.get("shape_probe") or {}).get("device") or "cpu")
        device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
        if hasattr(model, "device"):
            model.device = device
        if not hasattr(model, "model") and hasattr(model, "_init_model"):
            model.model = model._init_model()
        inner = getattr(model, "model", None)
        target = inner if inner is not None else model
        if hasattr(target, "to"):
            target.to(device)
        if hasattr(target, "eval"):
            target.eval()
        result["status"] = "ok"
        result["loader_model_name"] = getattr(factory, "model_name", "")
        result["loader_model_hyper_params"] = {
            key: hparams.get(key)
            for key in sorted(hparams)
            if key in {
                "task_name",
                "seq_len",
                "pred_len",
                "horizon",
                "input_chunk_length",
                "output_chunk_length",
                "enc_in",
                "dec_in",
                "c_out",
                "patch_len",
                "seg_len",
                "stride",
            }
        }
        if inner is not None:
            result["adapter_inner_model"] = {
                "type": type(inner).__name__,
                "module": getattr(type(inner), "__module__", ""),
            }
        if hasattr(target, "named_modules"):
            result["modules"] = [
                _runtime_module_record(name, module)
                for name, module in list(target.named_modules())[:300]
                if name
            ]
        if hasattr(target, "named_parameters"):
            result["parameters"] = [
                {"name": name, "shape": list(param.shape), "requires_grad": bool(param.requires_grad)}
                for name, param in list(target.named_parameters())[:300]
            ]
        if hasattr(target, "named_buffers"):
            result["buffers"] = [
                {"name": name, "shape": list(buf.shape)}
                for name, buf in list(target.named_buffers())[:100]
            ]
        result["active_path_probe"] = _runtime_active_path_probe(model, target, hparams, device, torch)
        if inner is not None:
            result["inner_modules"] = [
                {
                    **dict(item),
                    "name": "model." + str(item.get("name") or ""),
                }
                for item in result["modules"]
            ]
            result["inner_parameters"] = [
                {"name": "model." + str(item.get("name") or ""), "shape": item.get("shape"), "requires_grad": item.get("requires_grad")}
                for item in result["parameters"]
            ]
            result["inner_buffers"] = [{"name": "model." + str(item.get("name") or ""), "shape": item.get("shape")} for item in result["buffers"]]
            active = dict(result.get("active_path_probe") or {})
            active_names = [str(name or "") for name in list(active.get("called_module_names") or [])]
            active["called_inner_module_names"] = ["model." + name for name in active_names if name]
            result["active_path_probe"] = active
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _probe_shape_contract(cls: Any, args: Dict[str, Any], has_inner_model: bool) -> Dict[str, Any]:
    probe_args = dict(args.get("shape_probe") or {})
    enabled = bool(probe_args) or bool(args.get("run_shape_probe"))
    if not enabled:
        return {
            "status": "not_probed",
            "reason": "Set run_shape_probe=true or provide shape_probe to execute a conservative forward probe.",
        }
    try:
        import torch
    except Exception as exc:
        return {"status": "error", "error": f"torch_unavailable: {type(exc).__name__}: {exc}"}

    batch = _safe_int_scalar(probe_args.get("batch"), 2)
    seq_len = _safe_int_scalar(probe_args.get("seq_len") or probe_args.get("input_chunk_length"), 96)
    pred_len = _safe_int_scalar(probe_args.get("pred_len") or probe_args.get("horizon") or probe_args.get("output_chunk_length"), 96)
    channels = _safe_int_scalar(probe_args.get("channels") or probe_args.get("enc_in") or probe_args.get("c_out"), 1)
    requested_device = str(probe_args.get("device") or "cpu")
    device = torch.device(requested_device if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    hparams = {
        "seq_len": seq_len,
        "horizon": pred_len,
        "pred_len": pred_len,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "num_epochs": 1,
        "batch_size": batch,
        "use_gpu": False,
        **_merged_probe_hparams(args),
        **dict(probe_args.get("model_hyper_params") or {}),
    }
    try:
        model = cls(**hparams)
        if hasattr(model, "device"):
            model.device = device
        if has_inner_model and not hasattr(model, "model"):
            model.model = model._init_model()
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
        if hasattr(model, "model") and hasattr(model.model, "eval"):
            model.model.eval()
        x = torch.zeros(batch, seq_len, channels, device=device)
        target = torch.zeros(batch, pred_len, channels, device=device)
        input_mark = torch.zeros(batch, seq_len, 4, device=device)
        target_mark = torch.zeros(batch, pred_len, 4, device=device)
        with torch.no_grad():
            if hasattr(model, "_process"):
                out = model._process(x, target, input_mark, target_mark)
                output = out.get("output") if isinstance(out, dict) else out
            elif hasattr(model, "forward"):
                output = model(x)
            else:
                return {"status": "error", "error": "no _process or forward method available for probing"}
        if isinstance(output, tuple):
            output = output[0]
        return {
            "status": "ok",
            "probe_input_shape": list(x.shape),
            "probe_target_shape": list(target.shape),
            "output_shape": list(output.shape) if hasattr(output, "shape") else str(type(output).__name__),
            "device": str(device),
            "hparams": {key: hparams[key] for key in sorted(hparams) if key in {"seq_len", "pred_len", "horizon", "enc_in", "dec_in", "c_out", "patch_len"}},
        }
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "hparams": hparams}


def _repo_rel(path: str) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path


def _model_structure_cache_dir(session: AgentSession) -> Path:
    path = shared_knowledge_dir(session.base_dir) / "model_structure_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_cache_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value or "model"))


def _source_fingerprint(source_files: List[Dict[str, Any]]) -> str:
    pairs = [
        f"{item.get('path')}:{item.get('sha1')}"
        for item in list(source_files or [])
        if item.get("path") and item.get("sha1")
    ]
    return hashlib.sha1("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()


def _resolve_source_imports(source_files: Iterable[str]) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in source_files:
        file_path = Path(str(source)).resolve()
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = str(node.module or "")
            names = [alias.name for alias in node.names]
            import_text = f"from {'.' * int(node.level or 0)}{module} import {', '.join(names)}"
            candidates: List[Path] = []
            if node.level:
                base = file_path.parent
                for _ in range(max(int(node.level) - 1, 0)):
                    base = base.parent
                if module:
                    module_path = base.joinpath(*module.split("."))
                    candidates.extend([module_path.with_suffix(".py"), module_path / "__init__.py"])
                else:
                    candidates.append(base / "__init__.py")
            elif module.startswith("ts_benchmark") or module.startswith("evocast"):
                module_path = PROJECT_ROOT.joinpath(*module.split("."))
                candidates.extend([module_path.with_suffix(".py"), module_path / "__init__.py"])
            resolved_path = ""
            for candidate in candidates:
                if candidate.exists():
                    resolved_path = _repo_rel(str(candidate))
                    break
            key = (_repo_rel(str(file_path)), import_text, resolved_path)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(
                {
                    "source_file": _repo_rel(str(file_path)),
                    "import": import_text,
                    "module": module,
                    "level": int(node.level or 0),
                    "imported_names": names,
                    "resolved_path": resolved_path,
                    "status": "resolved" if resolved_path else "unresolved",
                }
            )
    return sorted(resolved, key=lambda item: (item.get("source_file") or "", item.get("import") or ""))


def _cache_model_structure(session: AgentSession, analysis: Dict[str, Any]) -> str:
    model_key = _safe_cache_name(str(analysis.get("model_key") or "model"))
    payload = {
        "schema_version": "model_structure_cache_v1",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "status": analysis.get("status"),
        "model_key": analysis.get("model_key"),
        "import_path": analysis.get("import_path"),
        "adapter": analysis.get("adapter"),
        "source_files": analysis.get("source_files") or [],
        "objective_boundaries": analysis.get("objective_boundaries") or [],
        "source_fingerprint": _source_fingerprint(list(analysis.get("source_files") or [])),
        "analysis": analysis,
    }
    path = _model_structure_cache_dir(session) / f"{model_key}.json"
    _json_write(path, payload)
    return str(path)


def _effective_runtime_introspection(runtime: Dict[str, Any]) -> Dict[str, Any]:
    if str(runtime.get("status") or "") == "ok":
        return runtime
    fallback = dict(runtime.get("fallback_runtime_introspection") or {})
    if str(fallback.get("status") or "") == "ok":
        fallback["_selected_from"] = "fallback_runtime_introspection"
        return fallback
    return runtime


def _runtime_analysis_path(name: str) -> str:
    normalized = str(name or "").strip().lstrip(".")
    if not normalized:
        return ""
    return normalized if normalized.startswith("self.") else f"self.{normalized}"

def _wildcard_runtime_path(name: str) -> str:
    parts = [part for part in str(name or "").strip().lstrip(".").split(".") if part]
    if not parts:
        return ""
    result: List[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            result.append(f"{part}[*]")
            index += 2
            continue
        if not part.isdigit():
            result.append(part)
        index += 1
    return ".".join(result)


def _runtime_parent_path(name: str) -> str:
    parts = [part for part in str(name or "").strip().lstrip(".").split(".") if part]
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


def _runtime_local_component_path(name: str) -> str:
    parts = [part for part in str(name or "").strip().lstrip(".").split(".") if part]
    return parts[-1] if parts else ""


def _runtime_index_trace(name: str) -> List[int]:
    return [int(part) for part in str(name or "").strip().lstrip(".").split(".") if part.isdigit()]


def _project_source_path(path: str) -> str:
    candidate = str(path or "").replace("\\", "/").strip()
    if not candidate:
        return ""
    try:
        return _repo_rel(str(Path(candidate).resolve()))
    except Exception:
        return candidate


def _source_in_project(path: str) -> bool:
    candidate = str(path or "").strip()
    if not candidate:
        return False
    try:
        Path(candidate).resolve().relative_to(PROJECT_ROOT)
        return True
    except Exception:
        try:
            normalize_repo_path(candidate)
            return True
        except Exception:
            return False


def _canonical_component_path(item: Dict[str, Any]) -> str:
    canonical = str(
        item.get("canonical_component_path")
        or item.get("path")
        or item.get("component_path")
        or item.get("name")
        or ""
    ).strip()
    if canonical.startswith("self.") and not canonical.startswith("self.model."):
        suffix = canonical[len("self.") :]
        for runtime_path in list(item.get("expanded_runtime_paths") or []):
            normalized_runtime = str(runtime_path or "").strip()
            if normalized_runtime.startswith("self.model.") and normalized_runtime.endswith(suffix):
                return normalized_runtime
        owner_runtime_path = str(item.get("owner_runtime_path") or "").strip()
        if owner_runtime_path.startswith("self.model."):
            return f"{owner_runtime_path}.{str(item.get('local_component_path') or '').strip()}".rstrip(".")
    return canonical


def _display_component_path(path: str) -> str:
    normalized = str(path or "").strip()
    if normalized.startswith("self.model."):
        return normalized.replace("self.model.", "self.", 1)
    return normalized


def _candidate_patchable(item: Dict[str, Any]) -> bool:
    path = _canonical_component_path(item)
    if not path or not bool(item.get("forward_reachable")):
        return False
    if "[*]" in path and not list(item.get("expanded_runtime_paths") or []):
        return False
    needs_owner_mapping = bool("[*]" in path or path.count(".") >= 2 or str(item.get("granularity") or "") == "mechanism")
    if not needs_owner_mapping:
        return True
    if not str(item.get("owner_class") or "").strip():
        return False
    if not str(item.get("owner_source_file") or "").strip():
        return False
    if not _source_in_project(str(item.get("owner_source_file") or "")):
        return False
    if not str(item.get("local_component_path") or "").strip():
        return False
    if not bool(item.get("owner_local_component_found", True)):
        return False
    return True


def _annotate_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    annotated = dict(item)
    canonical = _canonical_component_path(annotated)
    annotated["canonical_component_path"] = canonical
    annotated["display_component_path"] = _display_component_path(canonical)
    if canonical:
        annotated["path"] = canonical
        annotated["component_path"] = canonical
    annotated["owner_source_in_project"] = _source_in_project(
        str(annotated.get("owner_source_file") or annotated.get("source_file") or "")
    )
    annotated["patchable"] = _candidate_patchable(annotated)
    return annotated


def _runtime_parameter_count(name: str, runtime_params: Iterable[Dict[str, Any]]) -> int:
    target = str(name or "").strip()
    total = 0
    for item in runtime_params:
        param_name = str(item.get("name") or "").strip()
        shape = list(item.get("shape") or [])
        if not target or not param_name or not (param_name == target or param_name.startswith(target + ".")):
            continue
        count = 1
        for dim in shape:
            try:
                count *= int(dim)
            except Exception:
                count = 0
                break
        total += count
    return total


def _mechanism_shape(shape_tree: Any) -> Any:
    if isinstance(shape_tree, dict):
        if shape_tree.get("kind") == "tensor":
            return list(shape_tree.get("shape") or [])
        if shape_tree.get("kind") in {"tuple", "list"}:
            items = list(shape_tree.get("items") or [])
            return _mechanism_shape(items[0]) if items else None
        if shape_tree.get("kind") == "dict":
            items = dict(shape_tree.get("items") or {})
            for value in items.values():
                candidate = _mechanism_shape(value)
                if candidate is not None:
                    return candidate
    return None


def _classify_runtime_mechanism(name: str, module_record: Dict[str, Any]) -> Dict[str, str]:
    lowered = str(name or "").lower()
    module_type = str(module_record.get("type") or "").lower()
    if lowered.endswith(".time_attention"):
        return {
            "mechanism_family": "temporal_attention",
            "mechanism_role": "time-dimension attention over segments/time steps",
            "granularity": "mechanism",
        }
    if lowered.endswith(".dim_sender"):
        return {
            "mechanism_family": "variable_attention",
            "mechanism_role": "variable-dimension sender attention for inter-variable aggregation",
            "granularity": "mechanism",
        }
    if lowered.endswith(".dim_receiver"):
        return {
            "mechanism_family": "variable_attention",
            "mechanism_role": "variable-dimension receiver attention for inter-variable redistribution",
            "granularity": "mechanism",
        }
    if lowered.endswith(".self_attention"):
        return {
            "mechanism_family": "decoder_self_attention",
            "mechanism_role": "decoder self-attention mechanism",
            "granularity": "mechanism",
        }
    if lowered.endswith(".cross_attention"):
        return {
            "mechanism_family": "decoder_cross_attention",
            "mechanism_role": "decoder cross-attention over encoder outputs",
            "granularity": "mechanism",
        }
    if lowered.endswith(".mlp1") or lowered.endswith(".mlp2") or module_type == "sequential":
        return {
            "mechanism_family": "feedforward",
            "mechanism_role": "feedforward transformation block",
            "granularity": "mechanism",
        }
    if lowered.endswith(".linear_pred"):
        return {
            "mechanism_family": "prediction_head",
            "mechanism_role": "linear prediction head",
            "granularity": "mechanism",
        }
    if ".norm" in lowered or module_type == "layernorm":
        return {
            "mechanism_family": "normalization",
            "mechanism_role": "normalization layer",
            "granularity": "mechanism",
        }
    if "patch" in lowered or module_type == "patchembedding":
        return {
            "mechanism_family": "patching",
            "mechanism_role": "patch/segment embedding",
            "granularity": "mechanism",
        }
    if "merge" in lowered or module_type == "segmerging":
        return {
            "mechanism_family": "segment_merging",
            "mechanism_role": "segment merging/downsampling",
            "granularity": "mechanism",
        }
    if "attention" in lowered or module_type in {"attentionlayer", "fullattention", "twostageattentionlayer"}:
        return {
            "mechanism_family": "attention",
            "mechanism_role": "attention mechanism",
            "granularity": "mechanism",
        }
    if module_type in {"encoderlayer", "decoderlayer"}:
        return {
            "mechanism_family": "transformer_block",
            "mechanism_role": "encoder/decoder block",
            "granularity": "mechanism",
        }
    return {
        "mechanism_family": "component_modification",
        "mechanism_role": "generic runtime module",
        "granularity": "coarse_component",
    }


def _is_mechanism_level_module(name: str, module_record: Dict[str, Any]) -> bool:
    lowered = str(name or "").lower()
    module_type = str(module_record.get("type") or "").lower()
    excluded_terms = (
        "query_projection",
        "key_projection",
        "value_projection",
        "out_projection",
        "dropout",
        "gelu",
    )
    if any(term in lowered for term in excluded_terms):
        return False
    if ".inner_attention" in lowered:
        return False
    if lowered.endswith(".router"):
        return False
    if _runtime_local_component_path(name).isdigit():
        return False
    if module_type in {"dropout", "gelu", "relu", "linear"} and not lowered.endswith(".linear_pred"):
        return False
    meta = _classify_runtime_mechanism(name, module_record)
    return str(meta.get("granularity") or "") == "mechanism"


def _runtime_mechanism_candidates(runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
    effective_runtime = _effective_runtime_introspection(runtime)
    active_probe = dict(effective_runtime.get("active_path_probe") or {})
    if str(active_probe.get("status") or "") != "ok":
        return []

    module_records: Dict[str, Dict[str, Any]] = {}
    for item in list(effective_runtime.get("modules") or []) + list(effective_runtime.get("inner_modules") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            module_records[name] = {**dict(item), "name": name, "source_file": _project_source_path(str(item.get("source_file") or ""))}
    active_records: Dict[str, Dict[str, Any]] = {}
    for item in list(active_probe.get("called_modules") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            active_records[name] = {**dict(item), "name": name}
    runtime_params = list(effective_runtime.get("parameters") or []) + list(effective_runtime.get("inner_parameters") or [])
    grouped: Dict[str, Dict[str, Any]] = {}
    active_runtime_names = (
        list(active_probe.get("called_inner_module_names") or [])
        if list(active_probe.get("called_inner_module_names") or [])
        else list(active_probe.get("called_module_names") or [])
    )
    has_inner_runtime_modules = any(name.startswith("model.") for name in module_records)
    for concrete_name in active_runtime_names:
        concrete_name = str(concrete_name or "").strip()
        if has_inner_runtime_modules and concrete_name and not concrete_name.startswith("model."):
            model_prefixed = f"model.{concrete_name}"
            if model_prefixed in module_records:
                concrete_name = model_prefixed
        module_record = dict(module_records.get(concrete_name) or {})
        if not concrete_name or not module_record or not _is_mechanism_level_module(concrete_name, module_record):
            continue
        wildcard_name = _wildcard_runtime_path(concrete_name)
        if not wildcard_name:
            continue
        active_record = dict(active_records.get(concrete_name) or {})
        if not active_record and concrete_name.startswith("model."):
            active_record = dict(active_records.get(concrete_name[len("model.") :]) or {})
        parent_name = _runtime_parent_path(concrete_name)
        parent_record = dict(module_records.get(parent_name) or {})
        meta = _classify_runtime_mechanism(concrete_name, module_record)
        item = grouped.setdefault(
            wildcard_name,
            {
                "name": wildcard_name,
                "path": _runtime_analysis_path(wildcard_name),
                "source_file": str(module_record.get("source_file") or ""),
                "owner_class": str(parent_record.get("class_name") or ""),
                "owner_source_file": _project_source_path(str(parent_record.get("source_file") or "")),
                "owner_runtime_path": _runtime_analysis_path(_wildcard_runtime_path(parent_name) or parent_name),
                "local_component_path": _runtime_local_component_path(concrete_name),
                "runtime_type": str(module_record.get("type") or ""),
                "mechanism_family": str(meta.get("mechanism_family") or "component_modification"),
                "mechanism_role": str(meta.get("mechanism_role") or "generic runtime module"),
                "forward_reachable": True,
                "same_input_observable": True,
                "replaceability": "direct",
                "recommended_exact_edit_strategy": "replace_with_simple_baseline"
                if str(meta.get("mechanism_family") or "") == "normalization"
                else "remove_mechanism_and_route_residual",
                "recommended_plan_type": "exact_edits",
                "granularity": str(meta.get("granularity") or "mechanism"),
                "runtime_input_shape": _mechanism_shape(active_record.get("input")),
                "runtime_output_shape": _mechanism_shape(active_record.get("output")),
                "parameter_count": 0,
                "evidence": ["called by current-task runtime forward probe"],
                "risks": [],
                "confidence": 0.9,
                "apply_scope": "owner_class_all_instances" if "[*]" in wildcard_name else "owner_instance",
                "expanded_runtime_paths": [],
                "instance_indices": [],
                "owner_local_component_found": True,
                "runtime_candidate": True,
            },
        )
        item["expanded_runtime_paths"].append(_runtime_analysis_path(concrete_name))
        item["instance_indices"].append(_runtime_index_trace(concrete_name))
        item["parameter_count"] += _runtime_parameter_count(concrete_name, runtime_params)
        if not item.get("source_file"):
            item["source_file"] = str(module_record.get("source_file") or "")
        if not item.get("owner_class"):
            item["owner_class"] = str(parent_record.get("class_name") or "")
        if not item.get("owner_source_file"):
            item["owner_source_file"] = _project_source_path(str(parent_record.get("source_file") or ""))
        if not item.get("runtime_input_shape"):
            item["runtime_input_shape"] = _mechanism_shape(active_record.get("input"))
        if not item.get("runtime_output_shape"):
            item["runtime_output_shape"] = _mechanism_shape(active_record.get("output"))
    candidates: List[Dict[str, Any]] = []
    for item in grouped.values():
        expanded = sorted(set(str(path) for path in list(item.get("expanded_runtime_paths") or [])))
        item["expanded_runtime_paths"] = expanded
        item["evidence"] = list(item.get("evidence") or []) + [f"runtime expansion count={len(expanded)}"]
        item["owner_local_component_found"] = bool(item.get("local_component_path")) and not str(item.get("local_component_path")).isdigit()
        item["owner_source_in_project"] = _source_in_project(str(item.get("owner_source_file") or ""))
        candidates.append(item)
    return sorted(candidates, key=lambda candidate: (candidate.get("path") or ""))


def _classify_fit_points(model_key: str, components: List[Dict[str, Any]], forward: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    effective_runtime = _effective_runtime_introspection(runtime)
    called_paths = {str(item.get("call") or "") for item in list(forward.get("called_components") or [])}
    runtime_ok = str(effective_runtime.get("status") or "") == "ok"
    runtime_module_paths = {
        str(m.get("name") or "")
        for m in list(effective_runtime.get("modules") or []) + list(effective_runtime.get("inner_modules") or [])
        if m.get("name")
    }
    runtime_param_paths = {
        str(p.get("name") or "")
        for p in list(effective_runtime.get("parameters") or []) + list(effective_runtime.get("inner_parameters") or [])
        if p.get("requires_grad")
    }
    runtime_buffer_paths = {
        str(b.get("name") or "")
        for b in list(effective_runtime.get("buffers") or []) + list(effective_runtime.get("inner_buffers") or [])
        if b.get("name")
    }
    active_probe = dict(effective_runtime.get("active_path_probe") or {})
    active_probe_ok = str(active_probe.get("status") or "") == "ok"
    active_module_paths = {
        str(name or "")
        for name in list(active_probe.get("called_module_names") or []) + list(active_probe.get("called_inner_module_names") or [])
        if str(name or "")
    }
    protected_terms = {"optimizer", "scheduler", "loss", "criterion", "train", "loader", "metric", "early"}
    protected_hparam_terms = {
        "alpha",
        "beta",
        "d_model",
        "d_ff",
        "dropout",
        "factor",
        "head_dropout",
        "label_len",
        "lr",
        "n_heads",
        "patch_len",
        "pred_len",
        "seq_len",
        "stride",
        "top_p",
    }
    safe: List[Dict[str, Any]] = []
    risky: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []
    unknown: List[Dict[str, Any]] = []
    task_name = str(effective_runtime.get("task_name") or effective_runtime.get("task") or "").lower()
    patchtst_forecast = task_name in {"long_term_forecast", "short_term_forecast"}
    # ── Structural parent detection ──
    # A component is a "parent" if any other component's name starts with
    # "this_name." — i.e., it has declared sub-components in __init__.
    # Parents have internal forward logic; they need exact_edits.
    _all_component_names = {str(c.get("name") or "") for c in components}
    _parent_components = {
        n for n in _all_component_names
        if any(o.startswith(n + ".") for o in _all_component_names if o != n)
    }
    for comp in components:
        name = str(comp.get("name") or "")
        root = name.split(".", 1)[0]
        lowered = name.lower()
        item = {
            "name": name,
            "path": "self." + name,
            "constructor": comp.get("constructor"),
            "source_file": comp.get("source_file"),
            "defined_in": comp.get("defined_in"),
            "owner_class": comp.get("defined_in"),
            "owner_source_file": comp.get("source_file"),
            "owner_runtime_path": _runtime_analysis_path(name.rsplit(".", 1)[0]) if "." in name else "self",
            "local_component_path": comp.get("local_name") or name.rsplit(".", 1)[-1],
            "apply_scope": "owner_instance",
            "expanded_runtime_paths": ["self." + name] if name else [],
            "instance_indices": [],
            "owner_local_component_found": True,
            "evidence": [],
            "risks": [],
            "confidence": 0.45,
            # ── P0-1: structured fit-point eligibility fields ──
            "forward_reachable": False,
            "same_input_observable": False,
            "replaceability": "unknown",
            "recommended_plan_type": "exact_edits",
            "rejection_reason": "",
            "mechanism_family": _mechanism_type("self." + name, str(comp.get("constructor") or "")),
            "mechanism_role": "",
            "granularity": "coarse_component",
        }
        if name in _parent_components:
            item["evidence"].append("has declared child components")
        if patchtst_forecast and name in {"flatten", "projection"}:
            item["category"] = "protected"
            item["risks"].append("PatchTST forecast path does not construct this classification-only module")
            item["evidence"].append("classification-only PatchTST component")
            protected.append(item)
            continue
        if any(term in lowered for term in protected_terms):
            item["category"] = "protected"
            item["evidence"].append("name suggests training/evaluation policy")
            protected.append(item)
            continue
        if name.rsplit(".", 1)[-1].lower() in protected_hparam_terms:
            item["category"] = "protected"
            item["evidence"].append("name matches a protected hyperparameter/config token")
            item["risks"].append("modifying this trainable object is indistinguishable from hyperparameter tuning")
            protected.append(item)
            continue
        if comp.get("kind") == "parameter":
            item["category"] = "risky_candidate"
            item["evidence"].append("declared as parameter")
            item["risks"].append("parameter-only fit points require custom replacement and are not safe for generic module wrappers")
            risky.append(item)
            continue
        module_like = comp.get("kind") in {"nn_module", "container", "module_like"}
        runtime_present = (
            name in runtime_module_paths
            or name in runtime_param_paths
            or name in runtime_buffer_paths
            or any(param.startswith(name + ".") for param in runtime_param_paths)
            or any(buf.startswith(name + ".") for buf in runtime_buffer_paths)
        )
        if runtime_ok and module_like and not runtime_present:
            item["category"] = "risky_candidate"
            item["risks"].append("declared in source but absent in the instantiated runtime model for the current config")
            if name in called_paths or any(call.startswith(name + ".") for call in called_paths):
                item["evidence"].append("appears in parsed source forward graph, likely behind dynamic control flow")
            item["confidence"] = min(item["confidence"], 0.4)
            risky.append(item)
            continue
        active_present = (
            name in active_module_paths
            or any(active.startswith(name + ".") for active in active_module_paths)
            or any(name.startswith(active + ".") for active in active_module_paths if active)
        )
        if runtime_ok and active_probe_ok and module_like and runtime_present and not active_present:
            item["category"] = "risky_candidate"
            item["evidence"].append("present in instantiated runtime model")
            item["risks"].append("present but not called by the current task forward probe")
            item["confidence"] = min(item["confidence"], 0.45)
            risky.append(item)
            continue
        has_trainable_params = any(param == name or param.startswith(name + ".") for param in runtime_param_paths)
        observed_in_forward = name in called_paths or any(call.startswith(name + ".") for call in called_paths)
        if runtime_present:
            item["evidence"].append("present in instantiated runtime model")
            item["confidence"] += 0.15
        if has_trainable_params:
            item["evidence"].append("has trainable runtime parameters")
            item["confidence"] += 0.2
        if observed_in_forward:
            item["evidence"].append("observed in parsed forward call graph")
            item["confidence"] += 0.15
        if active_present:
            item["evidence"].append("called by current-task runtime forward probe")
            item["confidence"] += 0.2
        if comp.get("kind") in {"nn_module", "container", "parameter", "module_like"}:
            item["evidence"].append(f"declared as {comp.get('kind')}")
            item["confidence"] += 0.1
        # ── P0-1: populate structured eligibility fields ──
        item["forward_reachable"] = bool(observed_in_forward or active_present)
        item["same_input_observable"] = bool(module_like and runtime_present and active_present)
        if comp.get("kind") in {"nn_module", "container", "module_like"}:
            item["replaceability"] = "direct"
        elif comp.get("kind") == "parameter":
            item["replaceability"] = "none"
        elif module_like:
            item["replaceability"] = "indirect"
        else:
            item["replaceability"] = "unknown"
        # New architecture: all executable fit points are implemented by
        # exact edits in the round workspace.  Structural facts still drive
        # eligibility/risk, not materialization mode.
        kind = comp.get("kind") or ""
        if kind in {"container", "nn_module", "module_like", "parameter"}:
            item["recommended_plan_type"] = "exact_edits"
        item["recommended_exact_edit_strategy"] = (
            "replace_with_simple_baseline"
            if item["mechanism_family"] == "normalization"
            else "remove_mechanism_and_route_residual"
        )
        if not item["forward_reachable"]:
            item["rejection_reason"] = "not reachable in current-task forward graph"
        elif not item["same_input_observable"]:
            item["rejection_reason"] = "not observable under same-input forward probe"
        elif item["replaceability"] in ("none", "unknown"):
            item["rejection_reason"] = f"replaceability={item['replaceability']}"
        if comp.get("kind") in {"attribute"}:
            # P2 fix: attribute-kind components (lists, ints, strings) are NEVER
            # safe fit_points — _wrap_target requires nn.Module.  Route to risky
            # BEFORE the evidence/confidence check so high-confidence attributes
            # (e.g. period_len list at 0.65) cannot leak into safe_fit_points.
            item["category"] = "risky_candidate"
            item["risks"].append("not clearly an nn.Module/Parameter")
            risky.append(item)
        elif item["evidence"] and item["confidence"] >= 0.6 and (not module_like or runtime_present or not runtime_ok):
            # P0-1: points without forward reachability go to unknown, not safe
            if not item["forward_reachable"]:
                item["category"] = "unknown"
                item["risks"].append("not observed in parsed or runtime forward graph — demoted from safe")
                unknown.append(item)
            else:
                item["category"] = "safe_candidate"
                safe.append(item)
        else:
            item["category"] = "unknown"
            item["risks"].append("insufficient runtime/source evidence")
            unknown.append(item)
    # ── Model-specific promotions ────────────────────────────────────────
    _apply_model_specific_promotions(model_key, safe, risky, unknown, forward, effective_runtime)
    _apply_task_observability_demotions(model_key=model_key, safe=safe, risky=risky, runtime=effective_runtime)

    runtime_mechanisms = _runtime_mechanism_candidates(runtime)
    existing_by_path = {
        str(item.get("path") or item.get("name") or ""): item
        for item in safe + risky + unknown + protected
        if str(item.get("path") or item.get("name") or "")
    }
    safe_paths = {str(item.get("path") or item.get("name") or "") for item in safe}
    for candidate in runtime_mechanisms:
        path = str(candidate.get("path") or candidate.get("name") or "").strip()
        if not path:
            continue
        existing = existing_by_path.get(path)
        if existing is None:
            safe.append(candidate)
            existing_by_path[path] = candidate
            safe_paths.add(path)
            continue
        existing.update(
            {
                "source_file": candidate.get("source_file") or existing.get("source_file"),
                "owner_class": candidate.get("owner_class") or existing.get("owner_class"),
                "owner_source_file": candidate.get("owner_source_file") or existing.get("owner_source_file"),
                "owner_runtime_path": candidate.get("owner_runtime_path") or existing.get("owner_runtime_path"),
                "local_component_path": candidate.get("local_component_path") or existing.get("local_component_path"),
                "apply_scope": candidate.get("apply_scope") or existing.get("apply_scope"),
                "expanded_runtime_paths": list(candidate.get("expanded_runtime_paths") or existing.get("expanded_runtime_paths") or []),
                "instance_indices": list(candidate.get("instance_indices") or existing.get("instance_indices") or []),
                "mechanism_family": candidate.get("mechanism_family") or existing.get("mechanism_family"),
                "mechanism_role": candidate.get("mechanism_role") or existing.get("mechanism_role"),
                "granularity": candidate.get("granularity") or existing.get("granularity"),
                "runtime_type": candidate.get("runtime_type") or existing.get("runtime_type"),
                "runtime_input_shape": candidate.get("runtime_input_shape") or existing.get("runtime_input_shape"),
                "runtime_output_shape": candidate.get("runtime_output_shape") or existing.get("runtime_output_shape"),
                "parameter_count": candidate.get("parameter_count") or existing.get("parameter_count"),
                "forward_reachable": True,
                "same_input_observable": True,
                "replaceability": "direct",
                "recommended_exact_edit_strategy": candidate.get("recommended_exact_edit_strategy") or existing.get("recommended_exact_edit_strategy"),
                "owner_local_component_found": bool(candidate.get("owner_local_component_found", True)),
            }
        )
        merged_evidence = list(existing.get("evidence") or []) + [item for item in list(candidate.get("evidence") or []) if item not in list(existing.get("evidence") or [])]
        existing["evidence"] = merged_evidence
        existing["confidence"] = max(float(existing.get("confidence") or 0.0), float(candidate.get("confidence") or 0.0))
        if path not in safe_paths:
            safe.append(existing)
            safe_paths.add(path)

    # ── P0-1: unified candidate_fit_points — the primary API ──
    mechanism_candidates = [_annotate_candidate(item) for item in safe if str(item.get("granularity") or "") == "mechanism"]
    coarse_candidates = [_annotate_candidate(item) for item in safe if str(item.get("granularity") or "") != "mechanism"]
    candidate_fit_points = [_annotate_candidate(item) for item in safe + risky + unknown + protected]

    return {
        # === Primary API (P0-1) ===
        "candidate_fit_points": candidate_fit_points,
        "mechanism_candidates": mechanism_candidates,
        "coarse_component_candidates": coarse_candidates,
        # === Deprecated: legacy bucket fields ===
        # Retained for backward compatibility with external consumers
        # (wizard.py and tfb_ablation.py prompts).
        # New code MUST use candidate_fit_points with structured eligibility
        # fields (forward_reachable, same_input_observable, replaceability).
        # These buckets will be removed in a future cleanup pass once all
        # consumers have been migrated.
        "safe_fit_points": [_annotate_candidate(item) for item in safe],
        "risky_fit_points": [_annotate_candidate(item) for item in risky],
        "protected_fit_points": [_annotate_candidate(item) for item in protected],
        "unknown_fit_points": [_annotate_candidate(item) for item in unknown],
    }


# ── Model-specific safe-fit-point promotions ───────────────────────────────
# Some models have critical components (e.g., ModuleList submodules) that the
# generic classifier marks as risky despite being safe, well-defined targets.
MODEL_SAFE_PROMOTIONS: Dict[str, List[str]] = {
    "TSMixer": ["model"],   # nn.ModuleList of ResBlocks — safe to wrap/replace
}


def _apply_model_specific_promotions(
    model_key: str,
    safe: List[Dict[str, Any]],
    risky: List[Dict[str, Any]],
    unknown: List[Dict[str, Any]],
    forward: Dict[str, Any],
    runtime: Dict[str, Any],
) -> None:
    """Promote model-specific components from risky/unknown to safe."""
    names = MODEL_SAFE_PROMOTIONS.get(model_key, [])
    if not names:
        return
    name_set = set(names)
    for collection in (risky, unknown):
        promoted: List[int] = []
        for i, item in enumerate(collection):
            if item.get("name") in name_set:
                item["category"] = "safe_candidate"
                item["confidence"] = max(item.get("confidence", 0.5), 0.65)
                item["evidence"].append("promoted by model-specific safe_fit_point override")
                item["risks"].append("originally classified as risky; promoted due to known-safe architecture pattern")
                safe.append(item)
                promoted.append(i)
        # Remove in reverse to keep indices valid
        for i in reversed(promoted):
            collection.pop(i)


def _apply_task_observability_demotions(
    *,
    model_key: str,
    safe: List[Dict[str, Any]],
    risky: List[Dict[str, Any]],
    runtime: Dict[str, Any],
) -> None:
    """Demote fit points whose current-task operation is deterministically no-op."""

    hparams = dict(runtime.get("loader_model_hyper_params") or {})
    model_key_l = str(model_key or "").lower()
    demoted: List[int] = []
    if model_key_l == "crossformer":
        seq_len = _safe_int_scalar(hparams.get("seq_len") or hparams.get("input_chunk_length"), 0)
        seg_len = _safe_int_scalar(hparams.get("seg_len") or hparams.get("patch_len"), 0)
        padding = 0
        if seq_len > 0 and seg_len > 0:
            from math import ceil
            padding = int(ceil(1.0 * seq_len / seg_len) * seg_len - seq_len)
        for index, item in enumerate(safe):
            if str(item.get("name") or "") == "enc_value_embedding.padding_patch_layer" and padding == 0:
                item["category"] = "risky_candidate"
                item["rejection_reason"] = "current task has Crossformer padding=0; padding layer is a deterministic identity"
                item["risks"].append("current-task same-input behavior cannot change because padding length is zero")
                item["same_input_observable"] = False
                risky.append(item)
                demoted.append(index)
    for index in reversed(demoted):
        safe.pop(index)


def analyze_model_structure(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    # This tool always recomputes source facts; force_refresh is accepted so
    # callers can be explicit when replacing stale or empty caches.
    _ = bool(args.get("force_refresh"))
    model_key, import_path, adapter = _resolve_model(session, args)
    args = _structure_args_with_runtime_hparams(session, args, model_key, import_path)
    if not str(args.get("variant_path") or "").strip():
        _purge_workspace_shadowed_repo_modules()
    try:
        module, cls = _import_class(import_path)
    except Exception as exc:
        result = {
            "status": "error",
            "error_type": "model_import_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "model_key": model_key,
            "import_path": import_path,
            "adapter": adapter,
            "source_files": [],
            "inner_models": [],
            "safe_fit_points": [],
            "risky_fit_points": [],
            "protected_fit_points": [],
            "unknown_fit_points": [],
            "uncertainties": [
                "Model import failed before source analysis. Do not write architecture variants for this model until dependencies/imports are fixed.",
            ],
        }
        result["cache_path"] = _cache_model_structure(session, result)
        result["source_fingerprint"] = _source_fingerprint([])
        return result
    if not isinstance(cls, type):
        result = {
            "status": "error",
            "error_type": "non_class_model_info",
            "error": f"{import_path} resolves to {type(cls).__name__}, not a model class.",
            "model_key": model_key,
            "import_path": import_path,
            "adapter": adapter,
            "source_files": [],
            "inner_models": [],
            "safe_fit_points": [],
            "risky_fit_points": [],
            "protected_fit_points": [],
            "unknown_fit_points": [],
            "uncertainties": [
                "Factory/dictionary models are not safe Python architecture-variant targets without a specialized adapter-aware transformation.",
            ],
        }
        result["cache_path"] = _cache_model_structure(session, result)
        result["source_fingerprint"] = _source_fingerprint([])
        return result
    defining_module = importlib.import_module(getattr(cls, "__module__", module.__name__))
    classes = _source_chain(cls)
    inner_specs = _init_model_constructors(defining_module, classes)
    components = _extract_components(classes)
    hparams = _extract_hparams(classes)
    forward = _extract_forward(classes)
    inner_models: List[Dict[str, Any]] = []
    for spec in inner_specs:
        inner_cls = spec["class_object"]
        inner_classes = _source_chain(inner_cls)
        prefix = "model."
        inner_components = _extract_components(inner_classes, prefix=prefix)
        inner_hparams = _extract_hparams(inner_classes)
        inner_forward = _extract_forward(inner_classes, prefix=prefix)
        inner_source_files = sorted({item.file_path for item in inner_classes})
        components.extend(inner_components)
        forward.setdefault("called_components", []).extend(inner_forward.get("called_components") or [])
        inner_models.append(
            {
                "constructor": spec.get("constructor"),
                "class_path": spec.get("class_path"),
                "defined_in": spec.get("defined_in"),
                "via": spec.get("via"),
                "arguments": spec.get("arguments"),
                "keywords": spec.get("keywords"),
                "source_files": [{"path": _repo_rel(path), "sha1": _hash_file(path)} for path in inner_source_files],
                "inheritance_chain": [
                    {"class_name": item.class_name, "module": item.module, "file_path": _repo_rel(item.file_path), "bases": item.bases}
                    for item in inner_classes
                ],
                "components": inner_components,
                "hparam_schema": inner_hparams,
                "forward": inner_forward,
            }
        )
    if adapter:
        runtime = _runtime_introspection_via_tfb_loader(
            session,
            import_path=import_path,
            adapter=adapter,
            args=args,
        )
        if runtime.get("status") != "ok":
            fallback = _runtime_introspection_from_adapter(cls, args) if inner_models else _runtime_introspection(cls, args)
            runtime["fallback_runtime_introspection"] = fallback
        elif not inner_models and runtime.get("adapter_inner_model"):
            inner_info = dict(runtime.get("adapter_inner_model") or {})
            inner_path = ".".join(
                item
                for item in [str(inner_info.get("module") or ""), str(inner_info.get("type") or "")]
                if item
            )
            inner_models.append(
                {
                    "constructor": "TFB model loader adapter",
                    "class_path": inner_path,
                    "defined_in": str(adapter or ""),
                    "via": "runtime_adapter_loader",
                    "arguments": [],
                    "keywords": [],
                    "source_files": [],
                    "inheritance_chain": [],
                    "components": [],
                    "hparam_schema": {},
                    "forward": {},
                }
            )
    else:
        runtime = _runtime_introspection_from_adapter(cls, args) if inner_models else _runtime_introspection(cls, args)
    shape_contract = _probe_shape_contract(cls, args, has_inner_model=bool(inner_models))
    fit_points = _classify_fit_points(model_key, components, forward, runtime)
    source_files = sorted({item.file_path for item in classes} | {Path(src["path"]).resolve().as_posix() if Path(src["path"]).is_absolute() else str((PROJECT_ROOT / src["path"]).resolve()) for model in inner_models for src in model.get("source_files", [])})
    source_file_rows = [{"path": _repo_rel(path), "sha1": _hash_file(path)} for path in source_files]
    resolved_imports = _resolve_source_imports(source_files)
    objective_boundaries = discover_objective_boundaries(
        source_files=source_file_rows,
        project_root=PROJECT_ROOT,
        inner_models=inner_models,
    )
    result = {
        "status": "ok",
        "model_key": model_key,
        "import_path": import_path,
        "adapter": adapter,
        "class_name": getattr(cls, "__name__", ""),
        "module": getattr(module, "__name__", ""),
        "source_files": source_file_rows,
        "resolved_imports": resolved_imports,
        "objective_boundaries": objective_boundaries,
        "inheritance_chain": [
            {"class_name": item.class_name, "module": item.module, "file_path": _repo_rel(item.file_path), "bases": item.bases}
            for item in classes
        ],
        "inner_models": inner_models,
        "components": components,
        "hparam_schema": hparams,
        "forward": forward,
        "runtime_introspection": runtime,
        "task_contract": _runtime_task_contract(session, model_key, import_path, args),
        "shape_contract": shape_contract,
        **fit_points,
        "uncertainties": [
            "Forward graph is approximate for dynamic control flow.",
            "Runtime introspection requires valid constructor hparams and may be unavailable for some TFB adapters.",
        ],
    }
    cache_path = _cache_model_structure(session, result)
    result["cache_path"] = cache_path
    result["source_fingerprint"] = _source_fingerprint(list(result.get("source_files") or []))
    return result


def _model_card_dir(session: AgentSession) -> Path:
    path = shared_knowledge_dir(session.base_dir) / "model_cards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_model_card(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    card = {
        "schema_version": "v3_model_card_1",
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "analysis_version": "v3.0-initial",
        **analysis,
    }
    model_key = str(card.get("model_key") or "model")
    json_path = _model_card_dir(session) / f"{model_key}.json"
    md_path = _model_card_dir(session) / f"{model_key}.md"
    _json_write(json_path, card)
    md_lines = [
        f"# {model_key} Model Card",
        "",
        f"- Import path: `{card.get('import_path')}`",
        f"- Adapter: `{card.get('adapter')}`",
        f"- Class: `{card.get('class_name')}`",
        f"- Safe fit points: {len(card.get('safe_fit_points') or [])}",
        f"- Risky fit points: {len(card.get('risky_fit_points') or [])}",
        "",
        "## Safe Fit Points",
    ]
    for fp in list(card.get("safe_fit_points") or [])[:20]:
        md_lines.append(f"- `{fp.get('path')}` confidence={fp.get('confidence')}: {', '.join(fp.get('evidence') or [])}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"status": "ok", "model_card": card, "json_path": str(json_path), "markdown_path": str(md_path)}


def read_model_card(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    model_key = str(args.get("model_key") or "").strip()
    path_arg = str(args.get("path") or "").strip()
    path = Path(path_arg) if path_arg else (_model_card_dir(session) / f"{model_key}.json")
    if not path.is_absolute():
        path = PROJECT_ROOT / normalize_repo_path(path.as_posix())
    if not path.exists():
        raise ModelStructureError(f"model card not found: {path}")
    card = json.loads(path.read_text(encoding="utf-8"))
    stale = False
    stale_files: List[str] = []
    for item in list(card.get("source_files") or []):
        rel = item.get("path")
        expected = item.get("sha1")
        if rel and expected:
            current = _hash_file(str(PROJECT_ROOT / rel))
            if current and current != expected:
                stale = True
                stale_files.append(rel)
    return {"status": "ok", "path": str(path), "stale": stale, "stale_files": stale_files, "model_card": card}


def extract_forward_signature(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    return {
        "status": "ok",
        "model_key": analysis.get("model_key"),
        "outer_forward": analysis.get("forward"),
        "inner_forwards": [
            {
                "class_path": model.get("class_path"),
                "constructor": model.get("constructor"),
                "forward": model.get("forward"),
            }
            for model in analysis.get("inner_models") or []
        ],
    }


def extract_hparam_schema(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    return {
        "status": "ok",
        "model_key": analysis.get("model_key"),
        "outer_hparam_schema": analysis.get("hparam_schema"),
        "inner_hparam_schemas": [
            {
                "class_path": model.get("class_path"),
                "constructor": model.get("constructor"),
                "hparam_schema": model.get("hparam_schema"),
            }
            for model in analysis.get("inner_models") or []
        ],
    }


def extract_adapter_contract(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    inner_models = analysis.get("inner_models") or []
    adapter = analysis.get("adapter")
    entrypoint_pattern = "class_subclassing_wrapper" if inner_models else "direct_model_class"
    return {
        "status": "ok",
        "model_key": analysis.get("model_key"),
        "import_path": analysis.get("import_path"),
        "adapter": adapter,
        "entrypoint_class": analysis.get("class_name"),
        "entrypoint_pattern": entrypoint_pattern,
        "inner_model_required": bool(inner_models),
        "inner_models": [
            {
                "constructor": model.get("constructor"),
                "class_path": model.get("class_path"),
                "via": model.get("via"),
                "arguments": model.get("arguments"),
            }
            for model in inner_models
        ],
        "variant_guidance": _variant_guidance(analysis),
    }


def extract_shape_contract(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    shape_contract = dict(analysis.get("shape_contract") or {})
    shape_contract["runtime_modules_observed"] = len((analysis.get("runtime_introspection") or {}).get("modules") or [])
    return {
        "status": "ok",
        "model_key": analysis.get("model_key"),
        "shape_contract": shape_contract,
        "forward": analysis.get("forward"),
        "inner_forwards": [
            {"class_path": model.get("class_path"), "forward": model.get("forward")}
            for model in analysis.get("inner_models") or []
        ],
        "uncertainties": analysis.get("uncertainties") or [],
    }


def _component_lookup(analysis: Dict[str, Any], path: str) -> Dict[str, Any]:
    requested = str(path or "").replace("\\", ".").strip()
    requested = requested[5:] if requested.startswith("self.") else requested
    if not requested:
        return {}
    candidates: List[Dict[str, Any]] = []
    for comp in list(analysis.get("components") or []):
        name = str(comp.get("name") or comp.get("path") or "")
        normalized = name[5:] if name.startswith("self.") else name
        if normalized == requested or normalized.endswith("." + requested) or requested.endswith("." + normalized):
            candidates.append(dict(comp))
            continue
        stripped_requested = requested[6:] if requested.startswith("model.") else requested
        stripped_normalized = normalized[6:] if normalized.startswith("model.") else normalized
        if stripped_normalized == stripped_requested or stripped_normalized.endswith("." + stripped_requested) or stripped_requested.endswith("." + stripped_normalized):
            candidates.append(dict(comp))
    return candidates[0] if candidates else {}


def _path_shape_facts(analysis: Dict[str, Any], component_path: str) -> Dict[str, Any]:
    component = _component_lookup(analysis, component_path)
    name = str(component.get("name") or component_path or "")
    stripped = name[5:] if name.startswith("self.") else name
    runtime = dict(analysis.get("runtime_introspection") or {})
    params = []
    buffers = []
    modules = []
    for item in list(runtime.get("parameters") or []):
        pname = str(item.get("name") or "")
        if stripped and (pname == stripped or pname.startswith(stripped + ".") or pname.endswith("." + stripped)):
            params.append(item)
    for item in list(runtime.get("buffers") or []):
        bname = str(item.get("name") or "")
        if stripped and (bname == stripped or bname.startswith(stripped + ".") or bname.endswith("." + stripped)):
            buffers.append(item)
    for item in list(runtime.get("modules") or []):
        mname = str(item.get("name") or "")
        if stripped and (mname == stripped or mname.startswith(stripped + ".") or mname.endswith("." + stripped)):
            modules.append(item)
    return {
        "requested_path": component_path,
        "component": component,
        "runtime_module_count": len(modules),
        "parameter_shapes": [
            {"name": item.get("name"), "shape": item.get("shape")}
            for item in params[:20]
        ],
        "buffer_shapes": [
            {"name": item.get("name"), "shape": item.get("shape")}
            for item in buffers[:20]
        ],
        "forward": analysis.get("forward") or {},
        "shape_contract": analysis.get("shape_contract") or {},
    }


def _tensor_convention_hints(analysis: Dict[str, Any], facts: Dict[str, Any]) -> List[str]:
    text = json.dumps(
        {
            "model_key": analysis.get("model_key"),
            "component": facts.get("component"),
            "forward": facts.get("forward"),
            "shape_contract": facts.get("shape_contract"),
        },
        ensure_ascii=False,
        default=str,
    ).lower()
    hints: List[str] = []
    if "[b, t, c]" in text or "batch, seq" in text or "batch, time" in text:
        hints.append("likely_batch_time_channel")
    if "[b, c, t]" in text or "conv1d" in text or "permute" in text:
        hints.append("may_use_batch_channel_time")
    if "fft" in text or "rfft" in text or "frequency" in text:
        hints.append("frequency_domain_path")
    if "decomp" in text or "trend" in text or "seasonal" in text:
        hints.append("decomposition_path")
    if not hints:
        hints.append("unknown_tensor_convention")
    return sorted(set(hints))


def _shape_family(shapes: List[Dict[str, Any]]) -> List[int]:
    for item in shapes:
        shape = item.get("shape")
        if isinstance(shape, list) and shape:
            return [int(x) for x in shape if isinstance(x, int)]
    return []


def precheck_mechanism_transfer(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured compatibility evidence before borrowing a mechanism."""
    source_args = {
        "model_key": args.get("source_model_key"),
        "model_name": args.get("source_model_name"),
        "model_hyper_params": dict(args.get("source_model_hyper_params") or {}),
        "run_shape_probe": bool(args.get("run_shape_probe")),
        "shape_probe": dict(args.get("shape_probe") or {}),
    }
    target_args = {
        "model_key": args.get("target_model_key"),
        "model_name": args.get("target_model_name"),
        "model_hyper_params": dict(args.get("target_model_hyper_params") or {}),
        "run_shape_probe": bool(args.get("run_shape_probe")),
        "shape_probe": dict(args.get("shape_probe") or {}),
    }
    source = analyze_model_structure(session, source_args)
    target = analyze_model_structure(session, target_args)
    source_facts = _path_shape_facts(source, str(args.get("source_component_path") or ""))
    target_facts = _path_shape_facts(target, str(args.get("target_component_path") or ""))
    source_hints = _tensor_convention_hints(source, source_facts)
    target_hints = _tensor_convention_hints(target, target_facts)
    source_shape = _shape_family(list(source_facts.get("parameter_shapes") or []))
    target_shape = _shape_family(list(target_facts.get("parameter_shapes") or []))
    notes: List[str] = []
    suggestions: List[str] = []
    verdict = "unknown"
    if source.get("status") != "ok" or target.get("status") != "ok":
        verdict = "blocked_until_structure_ok"
        notes.append("source or target model structure analysis failed")
    elif not source_facts.get("component") and args.get("source_component_path"):
        verdict = "source_component_unresolved"
        notes.append("source component path was not found in analyzed source components")
    elif not target_facts.get("component") and args.get("target_component_path"):
        verdict = "target_component_unresolved"
        notes.append("target component path was not found in analyzed source components")
    else:
        verdict = "compatible_with_adapters"
        notes.append("both models have usable structure analysis")
    if source_shape and target_shape and source_shape != target_shape:
        notes.append(f"representative parameter shape differs: source={source_shape}, target={target_shape}")
        suggestions.append("derive adapter dimensions from target runtime modules instead of copying source constructor constants")
    if source_hints != target_hints:
        notes.append(f"tensor convention hints differ: source={source_hints}, target={target_hints}")
        suggestions.append("consider explicit transpose/permute boundaries and assert output returns to the target convention")
    if any("frequency_domain_path" in hints for hints in (source_hints, target_hints)):
        suggestions.append("for frequency mechanisms, check real/complex dtype handling and horizon/sequence length restoration")
    if any("decomposition_path" in hints for hints in (source_hints, target_hints)):
        suggestions.append("for decomposition mechanisms, keep trend/seasonal tensors aligned to the target time axis")
    suggestions.append("run exact-edit materialization compile/import checks and runtime contract probe before treating this as evidence of success")
    return {
        "status": "ok",
        "verdict": verdict,
        "mechanism_summary": str(args.get("mechanism_summary") or ""),
        "source": {
            "model_key": source.get("model_key"),
            "component_path": args.get("source_component_path"),
            "source_files": source.get("source_files") or [],
            "facts": source_facts,
            "tensor_convention_hints": source_hints,
        },
        "target": {
            "model_key": target.get("model_key"),
            "component_path": args.get("target_component_path"),
            "source_files": target.get("source_files") or [],
            "facts": target_facts,
            "tensor_convention_hints": target_hints,
        },
        "compatibility_notes": notes,
        "adaptation_suggestions": sorted(set(suggestions)),
        "audit": {
            "borrowed_from_candidate": [
                item.get("path")
                for item in list(source.get("source_files") or [])
                if item.get("path")
            ],
            "task_semantics_unchanged_required": True,
            "training_policy_unchanged_required": True,
        },
    }


def propose_ablation_targets(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis_payload = args.get("analysis")
    if isinstance(analysis_payload, dict):
        analysis = analysis_payload
    else:
        analysis = analyze_model_structure(session, args)
    max_targets = max(1, min(int(args.get("max_targets") or 6), 12))
    targets: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    preferred_candidates = _rank_ablation_candidates(
        list(analysis.get("mechanism_candidates") or [])
        + list(analysis.get("safe_fit_points") or [])
    )
    for fp in preferred_candidates:
        if len(targets) >= max_targets:
            break
        path = str(fp.get("path") or fp.get("name") or "")
        mechanism = _mechanism_type(path, str(fp.get("constructor") or ""))
        mechanism = str(fp.get("mechanism_family") or mechanism)
        safety = _ablation_safety_metadata(
            path=path,
            constructor=str(fp.get("constructor") or ""),
            mechanism=mechanism,
            analysis=analysis,
        )
        targets.append(
            {
                "target_id": f"{str(analysis.get('model_key') or 'model').lower()}.{path.replace('self.', '').replace('.', '_')}",
                "component_path": path,
                "canonical_component_path": path,
                "display_component_path": _display_component_path(path),
                "hypothesis": _hypothesis_for(mechanism),
                "mechanism_type": mechanism,
                "mechanism_role": fp.get("mechanism_role"),
                "granularity": fp.get("granularity") or "coarse_component",
                "owner_class": fp.get("owner_class"),
                "owner_source_file": fp.get("owner_source_file"),
                "owner_runtime_path": fp.get("owner_runtime_path"),
                "local_component_path": fp.get("local_component_path"),
                "apply_scope": fp.get("apply_scope"),
                "expanded_runtime_paths": list(fp.get("expanded_runtime_paths") or []),
                "runtime_input_shape": fp.get("runtime_input_shape"),
                "runtime_output_shape": fp.get("runtime_output_shape"),
                "forward_reachable": bool(fp.get("forward_reachable")),
                "patchable": bool(fp.get("patchable", True)),
                "expected_shape_preservation": bool(safety.get("ablation_safe")),
                **safety,
                "identity_validation": {
                    "variant_pattern": "wrap or subclass while preserving component output exactly before applying any change",
                    "expected_metric_delta": 0.0,
                    "required_checks": ["import_check", "forward_shape_check", "identity_smoke_against_baseline"],
                },
                "ablation_plan": {
                    "toggle_off": "remove or bypass the proposed mechanism while keeping all unrelated code fixed",
                    "compare_to": "baseline, candidate, and ablated candidate",
                    "success_condition": "candidate improves over baseline and ablated candidate loses the gain",
                },
                "risks": list(fp.get("risks") or []),
                "confidence": fp.get("confidence", 0.5),
                "evidence": list(fp.get("evidence") or []),
            }
        )
    for fp in list(analysis.get("protected_fit_points") or []) + list(analysis.get("risky_fit_points") or []):
        rejected.append({"component_path": fp.get("path"), "reason": "; ".join(list(fp.get("risks") or fp.get("evidence") or ["not safe by structural heuristic"]))})
    return {
        "status": "ok",
        "targets": targets,
        "rejected_targets": rejected,
        "ranked_candidate_paths": [item.get("component_path") for item in targets],
        "mechanism_candidates": list(analysis.get("mechanism_candidates") or []),
        "coarse_candidates": list(analysis.get("coarse_component_candidates") or []),
        "source_model_key": analysis.get("model_key"),
        "analysis_source": "fresh_source_analysis" if not isinstance(analysis_payload, dict) else "provided_analysis",
        "source_files": analysis.get("source_files") or [],
        "variant_guidance": _variant_guidance(analysis),
    }


def _variant_guidance(analysis: Dict[str, Any]) -> Dict[str, Any]:
    import_path = str(analysis.get("import_path") or "")
    class_name = str(analysis.get("class_name") or "Model")
    class_module = ""
    inheritance = list(analysis.get("inheritance_chain") or [])
    if inheritance:
        class_module = str((inheritance[0] or {}).get("module") or "")
    guidance = {
        "preserve_training_policy": True,
        "write_under": "round_sources/<task_id>/ResearchNNN/round_entry.py",
        "entrypoint": "class Model",
        "model_key": analysis.get("model_key"),
        "adapter": analysis.get("adapter"),
    }
    if analysis.get("inner_models"):
        wrapper_module = import_path
        wrapper_class = str(analysis.get("class_name") or "BaseModel")
        guidance.update(
            {
                "pattern": "subclass wrapper entrypoint and override _init_model to return a modified inner model",
                "must_preserve": ["_init_model construction side effects", "outer _process contract"],
                "base_wrapper_import": f"from {wrapper_module.rsplit('.', 1)[0]} import {wrapper_module.rsplit('.', 1)[-1]} as BaseModel" if "." in wrapper_module else "",
                "example_skeleton": (
                    f"from {wrapper_module.rsplit('.', 1)[0]} import {wrapper_class} as BaseModel\n\n"
                    "class Model(BaseModel):\n"
                    "    def _init_model(self):\n"
                    "        # preserve any baseline side effects before returning the modified inner model\n"
                    "        return super()._init_model()\n"
                ) if "." in wrapper_module else "",
            }
        )
        if import_path.endswith(".TimeFilter") or analysis.get("model_key") == "TimeFilter":
            guidance["must_preserve"].append("self.masks = self._get_mask() before returning the inner model")
    else:
        import_module = class_module or import_path
        guidance.update(
            {
                "pattern": "subclass the analyzed class; do not invent alternate source modules",
                "base_class_import": f"from {import_module} import {class_name} as BaseModel" if import_module else "",
                "example_skeleton": (
                    f"from {import_module} import {class_name} as BaseModel\n\n"
                    "class Model(BaseModel):\n"
                    "    def __init__(self, configs):\n"
                    "        super().__init__(configs)\n"
                ) if import_module else "",
            }
        )
    return guidance


MECHANISM_FAMILY_PRIORITY: Dict[str, int] = {
    "temporal_attention": 0,
    "variable_attention": 1,
    "decoder_self_attention": 2,
    "decoder_cross_attention": 3,
    "feedforward": 4,
    "prediction_head": 5,
    "patching": 6,
    "normalization": 7,
    "segment_merging": 8,
    "transformer_block": 9,
    "attention": 10,
    "component_modification": 11,
}


def _candidate_path_signature(path: str) -> str:
    return str(path or "").strip()


def _candidate_is_runtime_noise(item: Dict[str, Any]) -> bool:
    path = _candidate_path_signature(str(item.get("path") or item.get("component_path") or item.get("name") or ""))
    local_component_path = str(item.get("local_component_path") or "").strip()
    if not path:
        return True
    if ".inner_attention" in path:
        return True
    if local_component_path.isdigit():
        return True
    if path.endswith("[*]"):
        return True
    return False


def _candidate_priority_score(item: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
    path = _candidate_path_signature(str(item.get("path") or item.get("component_path") or item.get("name") or ""))
    family = str(item.get("mechanism_family") or item.get("mechanism_type") or "component_modification")
    score = 0
    if str(item.get("granularity") or "") == "mechanism":
        score += 300
    score += max(0, 120 - 10 * int(MECHANISM_FAMILY_PRIORITY.get(family, 12)))
    if "encoder.encode_blocks" in path and family in {"temporal_attention", "variable_attention", "feedforward", "segment_merging"}:
        score += 120
    if "decoder.decode_layers" in path and family in {"decoder_self_attention", "decoder_cross_attention", "feedforward", "prediction_head"}:
        score += 110
    if ".self_attention." in path and family in {"temporal_attention", "variable_attention", "feedforward", "normalization"}:
        score -= 70
    if ".norm" in path:
        score -= 20
    if _candidate_is_runtime_noise(item):
        score -= 300
    if not _source_in_project(str(item.get("owner_source_file") or item.get("source_file") or "")):
        score -= 200
    return (
        -score,
        int(MECHANISM_FAMILY_PRIORITY.get(family, 12)),
        0 if "encoder.encode_blocks" in path else 1,
        0 if "decoder.decode_layers" in path else 1,
        path,
    )


def _rank_ablation_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        normalized = _annotate_candidate(candidate)
        path = _candidate_path_signature(_canonical_component_path(normalized))
        if not path or _candidate_is_runtime_noise(normalized) or not bool(normalized.get("patchable")):
            continue
        existing = deduped.get(path)
        if existing is None or _candidate_priority_score(normalized) < _candidate_priority_score(existing):
            deduped[path] = normalized
    ranked = sorted(deduped.values(), key=_candidate_priority_score)
    selected: List[Dict[str, Any]] = []
    used_paths: set[str] = set()
    first_pass_families = (
        "temporal_attention",
        "variable_attention",
        "decoder_self_attention",
        "decoder_cross_attention",
        "feedforward",
        "prediction_head",
        "patching",
        "normalization",
    )
    for family in first_pass_families:
        for candidate in ranked:
            path = str(candidate.get("path") or "")
            if path in used_paths:
                continue
            if str(candidate.get("mechanism_family") or candidate.get("mechanism_type") or "") != family:
                continue
            selected.append(candidate)
            used_paths.add(path)
            break
    for candidate in ranked:
        path = str(candidate.get("path") or "")
        if path in used_paths:
            continue
        selected.append(candidate)
        used_paths.add(path)
    return selected


def _ablation_safety_metadata(*, path: str, constructor: str, mechanism: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{path} {constructor} {mechanism}".lower()
    reasons: List[str] = []
    shape_terms = [
        "upsample",
        "downsample",
        "interpolate",
        "interpolation",
        "freq",
        "frequency",
        "fft",
        "ifft",
        "rfft",
        "irfft",
    ]
    matched_terms = sorted(term for term in shape_terms if term in text)
    if matched_terms:
        reasons.append("length_transform_frequency_module")
    if "projection" in text and any(term in text for term in ("length", "len", "seq", "pred", "horizon")):
        reasons.append("projection_length_transform")
    calls = [
        str(item.get("call") or "").lower()
        for item in list((analysis.get("forward") or {}).get("called_components") or [])
        if isinstance(item, dict)
    ]
    if any(any(op in call for op in ("fft", "ifft", "rfft", "irfft")) for call in calls) and any(
        term in text for term in ("freq", "filter", "upsample", "projection")
    ):
        reasons.append("forward_frequency_transform_dependency")
    if reasons:
        return {
            "ablation_safe": False,
            "reason": reasons[0],
            "reasons": sorted(set(reasons)),
            "matched_risk_terms": matched_terms,
            "recommended_exact_edit_strategy": "replace_with_simple_baseline",
            "zero_ablation_unsafe": True,
        }
    if mechanism == "normalization":
        return {
            "ablation_safe": True,
            "reason": "normalization_or_scale_module",
            "reasons": [],
            "matched_risk_terms": [],
            "recommended_exact_edit_strategy": "replace_with_simple_baseline",
            "zero_ablation_unsafe": False,
        }
    return {
        "ablation_safe": True,
        "reason": "generic_feature_module",
        "reasons": [],
        "matched_risk_terms": [],
        "recommended_exact_edit_strategy": "remove_mechanism_and_route_residual",
        "zero_ablation_unsafe": False,
    }


def _mechanism_type(path: str, constructor: str) -> str:
    text = f"{path} {constructor}".lower()
    if "norm" in text or "revin" in text:
        return "normalization"
    if "head" in text or "proj" in text or "linear" in text:
        return "output_or_projection"
    if "attn" in text or "attention" in text:
        return "attention"
    if "fft" in text or "freq" in text or "filter" in text:
        return "frequency_filter"
    if "patch" in text:
        return "patching"
    if "drop" in text:
        return "regularization"
    return "component_modification"


def _hypothesis_for(mechanism: str) -> str:
    return {
        "normalization": "Changing normalization may improve distribution alignment while preserving temporal structure.",
        "output_or_projection": "Changing the projection/head may improve calibration or channel mixing without altering the full backbone.",
        "attention": "Changing attention may improve temporal dependency selection.",
        "temporal_attention": "Changing temporal attention may reveal whether time-axis dependency modeling is actually helping the baseline.",
        "variable_attention": "Changing variable attention may reveal whether cross-variable message passing is materially useful.",
        "decoder_self_attention": "Changing decoder self-attention may reveal whether the decoder needs its own internal temporal mixing.",
        "decoder_cross_attention": "Changing decoder cross-attention may reveal whether decoder access to encoder states is critical.",
        "feedforward": "Changing the feedforward block may reveal whether nonlinear channel mixing contributes beyond attention alone.",
        "prediction_head": "Changing the prediction head may reveal whether performance depends on the final readout rather than upstream representation quality.",
        "frequency_filter": "Changing frequency filtering may improve signal/noise separation.",
        "patching": "Changing patching may improve local temporal representation.",
        "regularization": "Changing regularization may improve generalization but requires seed verification.",
    }.get(mechanism, "Changing this localized component may improve the model while preserving the surrounding contract.")


def validate_model_contract(session: AgentSession, args: Dict[str, Any]) -> Dict[str, Any]:
    analysis = analyze_model_structure(session, args)
    forward_signature = bool(analysis.get("forward", {}).get("forward_signature")) or any(
        bool(model.get("forward", {}).get("forward_signature")) for model in analysis.get("inner_models") or []
    )
    ok = bool(analysis.get("source_files")) and forward_signature
    return {
        "status": "ok" if ok else "warning",
        "contract_ok": ok,
        "analysis": analysis,
        "warnings": [] if ok else ["Could not establish source files and forward signature."],
    }
