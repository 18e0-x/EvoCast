from __future__ import annotations

import ast
import inspect
import traceback
from pathlib import Path
from typing import Any, Dict, List


def _shape(value: Any) -> Any:
    if hasattr(value, "shape"):
        try:
            return list(value.shape)
        except Exception:
            return "<shape_unavailable>"
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _shape(item) for key, item in value.items()}
    return type(value).__name__


def _repo_path(path: str | Path | None) -> str:
    if not path:
        return ""
    raw = Path(str(path)).resolve()
    try:
        return raw.relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return raw.as_posix()


def _is_editable_file(path: str | Path | None) -> bool:
    text = str(path or "").replace("\\", "/")
    return "/ts_benchmark/baselines/" in text or text.startswith("ts_benchmark/baselines/")


def _read_line(path: str | Path, line: int) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        if 1 <= int(line) <= len(lines):
            return lines[int(line) - 1].strip()
    except Exception:
        pass
    return ""


def _source_info(obj: Any) -> Dict[str, Any]:
    try:
        file = inspect.getsourcefile(type(obj)) or ""
        _, line = inspect.getsourcelines(type(obj))
    except Exception:
        file, line = "", 0
    return {
        "file": _repo_path(file),
        "abs_file": str(Path(file).resolve()) if file else "",
        "line": line,
        "editable": _is_editable_file(file),
    }


def _call_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _find_attr_sites(parent: Any, attr: str) -> List[Dict[str, Any]]:
    try:
        file = inspect.getsourcefile(type(parent)) or ""
        lines, start = inspect.getsourcelines(type(parent))
        tree = ast.parse("".join(lines))
    except Exception:
        return []
    sites: List[Dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == attr
                ):
                    line = start + node.lineno - 1
                    sites.append({
                        "kind": "assign",
                        "file": _repo_path(file),
                        "abs_file": str(Path(file).resolve()),
                        "line": line,
                        "code": _read_line(file, line),
                        "editable": _is_editable_file(file),
                    })
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in {"append", "extend"}:
                    base = func.value
                    if (
                        isinstance(base, ast.Attribute)
                        and isinstance(base.value, ast.Name)
                        and base.value.id == "self"
                        and base.attr == attr
                    ):
                        line = start + node.lineno - 1
                        sites.append({
                            "kind": func.attr,
                            "file": _repo_path(file),
                            "abs_file": str(Path(file).resolve()),
                            "line": line,
                            "code": _read_line(file, line),
                            "editable": _is_editable_file(file),
                        })
                if func.attr == "add_module" and node.args:
                    receiver = _call_name(func.value)
                    first = node.args[0]
                    literal = first.value if isinstance(first, ast.Constant) else ""
                    if receiver == f"self.{attr}" or literal == attr:
                        line = start + node.lineno - 1
                        sites.append({
                            "kind": "add_module",
                            "file": _repo_path(file),
                            "abs_file": str(Path(file).resolve()),
                            "line": line,
                            "code": _read_line(file, line),
                            "editable": _is_editable_file(file),
                        })
            self.generic_visit(node)

    Visitor().visit(tree)
    return sites


def _map_editable_sites(module_path: str, modules: Dict[str, Any], root: Any) -> List[Dict[str, Any]]:
    parts = [part for part in str(module_path or "").split(".") if part]
    if not parts:
        info = _source_info(root)
        return [{
            "module_path": "<root>",
            "mapped_parent": "",
            "mapped_attr": "<root>",
            "kind": "class",
            **info,
            "code": _read_line(info.get("abs_file"), info.get("line")) if info.get("abs_file") else "",
        }]
    for cut in range(len(parts), 0, -1):
        attr = parts[cut - 1]
        if attr.isdigit():
            continue
        parent_path = ".".join(parts[: cut - 1])
        parent = modules.get(parent_path) if parent_path else root
        if parent is None:
            continue
        sites = _find_attr_sites(parent, attr)
        if sites:
            for site in sites:
                site["module_path"] = module_path
                site["mapped_parent"] = parent_path or "<root>"
                site["mapped_attr"] = attr
            return sites
    info = _source_info(modules.get(module_path) or root)
    return [{
        "module_path": module_path,
        "mapped_parent": "",
        "mapped_attr": "",
        "kind": "class",
        **info,
        "code": _read_line(info.get("abs_file"), info.get("line")) if info.get("abs_file") else "",
    }]


def _traceback_frames(exc: BaseException | None) -> List[Dict[str, Any]]:
    if exc is None:
        return []
    frames: List[Dict[str, Any]] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        frames.append({
            "file": _repo_path(frame.filename),
            "abs_file": str(Path(frame.filename).resolve()),
            "line": frame.lineno,
            "function": frame.name,
            "code": frame.line or "",
            "editable": _is_editable_file(frame.filename),
        })
    return frames


def _traceback_text_frames(traceback_text: str) -> List[Dict[str, Any]]:
    text = str(traceback_text or "")
    if not text.strip():
        return []
    frames: List[Dict[str, Any]] = []
    lines = text.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith('File "') and '", line ' in line:
            try:
                file_part, rest = line.split('", line ', 1)
                filename = file_part[len('File "') :]
                line_part, func_part = rest.split(", in ", 1)
                lineno = int(line_part.strip())
                code = ""
                if idx + 1 < len(lines):
                    code = lines[idx + 1].strip()
                frames.append({
                    "file": _repo_path(filename),
                    "abs_file": str(Path(filename).resolve()),
                    "line": lineno,
                    "function": func_part.strip(),
                    "code": code,
                    "editable": _is_editable_file(filename),
                })
            except Exception:
                pass
        idx += 1
    return frames


class RuntimeFactTracer:
    def __init__(self, model: Any):
        self.model = model
        self.root = self._resolve_root(model)
        self.status = "ok" if self.root is not None else "not_applicable"
        self.reason = "" if self.root is not None else "non_local_or_non_torch_model"
        self.modules: Dict[str, Any] = {}
        self.module_names: Dict[int, str] = {}
        self.events: List[Dict[str, Any]] = []
        self._handles: List[Any] = []
        self._next_event_id = 1
        if self.root is not None:
            self.modules = {name: module for name, module in self.root.named_modules()}
            self.module_names = {id(module): (name or "<root>") for name, module in self.modules.items()}

    @staticmethod
    def _resolve_root(model: Any) -> Any:
        try:
            import torch.nn as nn
        except Exception:
            return None
        candidates = [model, getattr(model, "model", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, nn.DataParallel):
                candidate = candidate.module
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    @property
    def applicable(self) -> bool:
        return self.root is not None

    def attach(self) -> None:
        if not self.applicable:
            return
        for name, module in self.root.named_modules():
            module_name = name or "<root>"
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(module_name, module)))
            self._handles.append(module.register_forward_hook(self._post_hook(module_name, module)))

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def _pre_hook(self, module_name: str, module: Any):
        def hook(_module: Any, inputs: Any) -> None:
            event_id = self._next_event_id
            self._next_event_id += 1
            self.events.append({
                "event_id": event_id,
                "kind": "pre",
                "module_name": module_name,
                "module_type": type(module).__name__,
                "input_shapes": _shape(inputs),
                "completed": False,
            })
        return hook

    def _post_hook(self, module_name: str, module: Any):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            for event in reversed(self.events):
                if event.get("kind") == "pre" and event.get("module_name") == module_name and not event.get("completed"):
                    event["completed"] = True
                    event["output_shapes"] = _shape(output)
                    break
            self.events.append({
                "event_id": self._next_event_id,
                "kind": "post",
                "module_name": module_name,
                "module_type": type(module).__name__,
                "output_shapes": _shape(output),
            })
            self._next_event_id += 1
        return hook

    def _failed_event(self) -> Dict[str, Any]:
        for event in reversed(self.events):
            if event.get("kind") == "pre" and not event.get("completed"):
                return dict(event)
        return {}

    def module_io_trace(self) -> List[Dict[str, Any]]:
        trace: List[Dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for event in self.events:
            if event.get("kind") != "pre" or not event.get("completed"):
                continue
            item = {
                "module_name": event.get("module_name"),
                "module_type": event.get("module_type"),
                "input_shapes": event.get("input_shapes"),
                "output_shapes": event.get("output_shapes"),
            }
            key = (item["module_name"], str(item["input_shapes"]), str(item["output_shapes"]))
            if key in seen:
                continue
            seen.add(key)
            trace.append(item)
        return trace

    def _parameter_owners(self, focus_module: str = "") -> List[Dict[str, Any]]:
        owners: List[Dict[str, Any]] = []
        if not self.applicable:
            return owners
        for module_name, module in self.root.named_modules():
            if focus_module and not (module_name == focus_module or focus_module.startswith(module_name + ".") or module_name.startswith(focus_module + ".")):
                continue
            try:
                params = list(module.named_parameters(recurse=False))
            except Exception:
                params = []
            for param_name, param in params:
                try:
                    shape = list(param.shape)
                except Exception as exc:
                    shape = f"unavailable: {type(exc).__name__}: {exc}"
                owners.append({
                    "module_name": module_name or "<root>",
                    "parameter_name": param_name,
                    "qualified_name": f"{module_name}.{param_name}" if module_name else param_name,
                    "shape": shape,
                    "requires_grad": bool(getattr(param, "requires_grad", False)),
                })
        return owners

    def build(
        self,
        *,
        exception: BaseException | None = None,
        output: Any = None,
        input_context: Dict[str, Any] | None = None,
        failure_traceback_text: str = "",
        failure_message: str = "",
    ) -> Dict[str, Any]:
        if not self.applicable:
            return {
                "status": "not_applicable",
                "model_kind": "non_local_or_non_torch",
                "reason": self.reason,
            }
        failed = exception is not None or bool(str(failure_traceback_text or "").strip()) or bool(str(failure_message or "").strip())
        failed_event = self._failed_event() if failed else {}
        failed_module_name = str(failed_event.get("module_name") or "")
        failed_sites = _map_editable_sites("" if failed_module_name == "<root>" else failed_module_name, self.modules, self.root) if failed_module_name else []
        trace_frames = _traceback_frames(exception) if exception is not None else _traceback_text_frames(failure_traceback_text)
        editable_frames = [frame for frame in trace_frames if frame.get("editable")]
        focus = "" if failed_module_name == "<root>" else failed_module_name
        ambiguities: List[str] = []
        if failed_sites and len(failed_sites) > 1:
            ambiguities.append("multiple_editable_sites_for_module_path")
        if any(part.isdigit() for part in focus.split(".")):
            ambiguities.append("numeric_container_child")
        confidence = "high"
        if ambiguities:
            confidence = "medium"
        if failed and not failed_event:
            confidence = "low"
            ambiguities.append("failed_module_not_observed")
        return {
            "status": "failed" if failed else "ok",
            "model_kind": "local_torch",
            "input_context": dict(input_context or {}),
            "failure_message": str(failure_message or ""),
            "output_shape": _shape(output) if not failed else None,
            "module_trace": self.module_io_trace(),
            "module_trace_count": len(self.module_io_trace()),
            "failed_module": {
                "module_name": failed_module_name,
                "module_type": failed_event.get("module_type"),
                "input_shapes": failed_event.get("input_shapes"),
                "editable_sites": failed_sites,
            } if failed_event else {},
            "traceback_frames": trace_frames,
            "editable_traceback_frames": editable_frames,
            "parameter_owners": self._parameter_owners(focus)[:80],
            "confidence": confidence,
            "ambiguities": ambiguities,
        }
