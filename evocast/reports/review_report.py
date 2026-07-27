from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from evocast.domain.knowledge_paths import task_knowledge_dir
from evocast.harness.api_client import create_task_client
from evocast.reports.review_fact_pack import attach_stage_timing_summary, attach_token_usage_summary, write_review_fact_pack
from evocast.reports.review_html import write_review_html
from evocast.reports.review_narrative import write_review_narrative
from evocast.reports.view_model import build_report_view_model
from evocast.state.cost_ledger import tracked_stage


def _write_report_error(*, task_id: str, base_dir: str, exc: BaseException, fact_pack_path: Optional[Path] = None) -> Path:
    path = task_knowledge_dir(base_dir, task_id) / "review_report_error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "review_report_error_v1",
        "task_id": task_id,
        "status": "error",
        "stage": "review_report",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "fact_pack_path": str(fact_pack_path) if fact_pack_path else "",
        "created_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


@tracked_stage(
    "review_report",
    lambda *, task_id, base_dir, client=None, api_config=None: (str(base_dir), str(task_id), "", ""),
)
def generate_review_report(
    *,
    task_id: str,
    base_dir: str,
    client: Any | None = None,
    api_config: Optional[Path] = None,
) -> Dict[str, Any]:
    fact_pack_path: Optional[Path] = None
    try:
        fact_pack, fact_pack_path = write_review_fact_pack(task_id=task_id, base_dir=base_dir)
        effective_client = client or create_task_client(
            base_dir=base_dir,
            task_id=task_id,
            explicit_config=api_config,
        )
        narrative, narrative_path = write_review_narrative(
            task_id=task_id,
            base_dir=base_dir,
            client=effective_client,
            fact_pack=fact_pack,
        )
        fact_pack = attach_token_usage_summary(fact_pack, task_id=task_id, base_dir=base_dir)
        fact_pack = attach_stage_timing_summary(fact_pack, task_id=task_id, base_dir=base_dir)
        if fact_pack_path is not None:
            fact_pack_path.write_text(
                json.dumps(fact_pack, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        view_model = build_report_view_model(fact_pack=fact_pack, narrative=narrative)
        locale_validation_path = task_knowledge_dir(base_dir, task_id) / "review_locale_validation.json"
        locale_validation_path.write_text(
            json.dumps(view_model.get("locale_validation") or {}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        html_path = write_review_html(
            task_id=task_id,
            base_dir=base_dir,
            fact_pack=fact_pack,
            narrative=narrative,
        )
        return {
            "status": "ok",
            "task_id": task_id,
            "fact_pack_path": str(fact_pack_path),
            "narrative_path": str(narrative_path),
            "locale_validation_path": str(locale_validation_path),
            "html_path": str(html_path),
        }
    except BaseException as exc:
        error_path = _write_report_error(
            task_id=task_id,
            base_dir=base_dir,
            exc=exc,
            fact_pack_path=fact_pack_path,
        )
        return {
            "status": "error",
            "task_id": task_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fact_pack_path": str(fact_pack_path) if fact_pack_path else "",
            "error_path": str(error_path),
        }
