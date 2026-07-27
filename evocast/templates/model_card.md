# Model Card: {{variant_name}}

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
