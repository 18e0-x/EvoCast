from __future__ import annotations

from evocast.tools.tfb_ablation import classify_seed_eval_for_ablation


def test_seed_eval_stable_improvement_requests_promotion() -> None:
    result = classify_seed_eval_for_ablation(
        objective_metric="mse_norm",
        seed_eval_result={
            "mean": 0.95,
            "reference_mean": 1.0,
            "valid_metric_seeds": 3,
            "successful_seeds": 3,
            "significance_decision": {"decision": "accept"},
        },
    )

    assert result["module_effect"] == "confirmed_harmful_or_redundant"
    assert result["recommended_action"] == "promote_ablation_variant_to_current_best"
