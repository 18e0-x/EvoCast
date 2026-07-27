"""Model card renderer for evocast.

Renders the model_card.md template with actual values from the
final model pack export.
"""

import os
from typing import Optional

from evocast.domain.metric_semantics import DEFAULT_OBJECTIVE_METRIC


def render_model_card(
    variant_name: str,
    dataset: str,
    horizon,
    objective_metric: str = DEFAULT_OBJECTIVE_METRIC,
    metric_direction: str = "lower_is_better",
    baseline_model: str = "None",
    baseline_value: Optional[float] = None,
    variant_path: str = "",
    fit_points: Optional[list] = None,
    final_value: Optional[float] = None,
    num_seeds: int = 0,
    target_reached: bool = False,
    tournament_rank: str = "N/A",
    weaknesses: str = "Not assessed",
    export_source: str = "best_overall",
    best_overall_variant_path: str = "",
    best_overall_value: Optional[float] = None,
) -> str:
    """Render the model card markdown.

    Args:
        variant_name: Name of the best variant/model.
        dataset: Dataset name or path.
        horizon: Forecast horizon.
        objective_metric: Objective metric name.
        metric_direction: Direction of improvement.
        baseline_model: Baseline model name.
        baseline_value: Baseline objective value.
        variant_path: Path to variant module.
        fit_points: Fit points modified.
        final_value: Final objective value.
        num_seeds: Number of seed evaluations.
        target_reached: Whether user target was reached.
        tournament_rank: Rank in tournament.
        weaknesses: Known weaknesses description.

    Returns:
        Rendered markdown string.
    """
    # Compute improvement
    improvement_pct = "N/A"
    if baseline_value is not None and final_value is not None:
        if baseline_value != 0:
            pct = ((baseline_value - final_value) / abs(baseline_value)) * 100
            direction_word = "lower" if metric_direction == "lower_is_better" else "higher"
            improvement_pct = f"{pct:+.2f}% ({direction_word} is better)"

    fit_points_str = ", ".join(fit_points) if fit_points else "none"

    # Load template
    template_path = os.path.join(os.path.dirname(__file__), "model_card.md")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()
    else:
        template = _default_template()

    # Simple template substitution
    replacements = {
        "{{variant_name}}": str(variant_name),
        "{{dataset}}": str(dataset),
        "{{horizon}}": str(horizon),
        "{{objective_metric}}": str(objective_metric),
        "{{metric_direction}}": str(metric_direction),
        "{{baseline_model}}": str(baseline_model),
        "{{baseline_value}}": f"{baseline_value:.6f}" if baseline_value is not None else "N/A",
        "{{baseline_config}}": "See runtime_state.json baseline candidate payload",
        "{{variant_path}}": str(variant_path),
        "{{fit_points}}": fit_points_str,
        "{{final_value}}": f"{final_value:.6f}" if final_value is not None else "N/A",
        "{{improvement_pct}}": improvement_pct,
        "{{num_seeds}}": str(num_seeds),
        "{{tournament_rank}}": str(tournament_rank),
        "{{weaknesses}}": str(weaknesses),
        "{{target_reached}}": str(target_reached),
        "{{export_source}}": str(export_source),
        "{{best_overall_variant_path}}": str(best_overall_variant_path),
        "{{best_overall_value}}": f"{best_overall_value:.6f}" if best_overall_value is not None else "N/A",
    }

    for key, value in replacements.items():
        template = template.replace(key, value)

    # Handle conditional block
    if target_reached:
        template = template.replace(
            "{{#if not target_reached}}\nThe agent did not reach the user's target but this is the best verified result.\n{{/if}}",
            "",
        )
    else:
        template = template.replace(
            "{{#if not target_reached}}",
            "",
        )
        template = template.replace(
            "{{/if}}",
            "",
        )

    return template


def _default_template() -> str:
    """Fallback template if model_card.md is missing."""
    return """# Model Card: {{variant_name}}

## Task
- Dataset: {{dataset}}
- Horizon: {{horizon}}
- Objective metric: {{objective_metric}} ({{metric_direction}} is better)

## Best Baseline
- Model: {{baseline_model}}
- {{objective_metric}}: {{baseline_value}}
- Config: {{baseline_config}}

## Final Selected Model
- Variant: {{variant_name}}
- Path: {{variant_path}}
- Fit points: {{fit_points}}
- {{objective_metric}}: {{final_value}}
- Improvement over baseline: {{improvement_pct}}
- Export source: {{export_source}}

## Evidence
- Seeds: {{num_seeds}}
- Tournament rank: {{tournament_rank}}
- Comparison table: comparison_table.csv

## Best Overall Candidate
- Path: {{best_overall_variant_path}}
- {{objective_metric}}: {{best_overall_value}}

## Known Weaknesses
{{weaknesses}}

## Reproduction
See reproduction_commands.md for exact commands.

## Claims
- Target/SOTA reached: {{target_reached}}
- Strongest verified result: {{objective_metric}} = {{final_value}}
{{#if not target_reached}}
The agent did not reach the user's target but this is the best verified result.
{{/if}}
"""
