from __future__ import annotations

from evocast.research.ablation.policy import classify_ablation_effect, classify_seed_eval_ablation_effect
from evocast.tools.tfb_ablation import classify_seed_eval_for_ablation


ABLATION_THRESHOLD = 0.02


def test_lower_is_better_threshold_buckets() -> None:
    improved = classify_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=0.98,
        threshold=ABLATION_THRESHOLD,
    )
    worsened = classify_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=1.02,
        threshold=ABLATION_THRESHOLD,
    )
    weak = classify_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=1.01,
        threshold=ABLATION_THRESHOLD,
    )

    assert improved["module_effect"] == "harmful_or_redundant_candidate"
    assert improved["recommended_action"] == "run_seed_eval"
    assert worsened["module_effect"] == "essential"
    assert worsened["recommended_action"] == "protect_current_mechanism"
    assert weak["module_effect"] == "weak_or_uncertain"
    assert weak["recommended_action"] == "record_as_context"


def test_seed_eval_post_classification_buckets() -> None:
    accepted = classify_seed_eval_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=0.97,
        threshold=ABLATION_THRESHOLD,
    )
    rejected = classify_seed_eval_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=1.03,
        threshold=ABLATION_THRESHOLD,
    )
    weak = classify_seed_eval_ablation_effect(
        objective_metric="mse_norm",
        metric_direction="lower_is_better",
        reference_value=1.0,
        candidate_value=0.995,
        threshold=ABLATION_THRESHOLD,
    )

    assert accepted["module_effect"] == "confirmed_harmful_or_redundant"
    assert accepted["recommended_action"] == "promote_ablation_variant_to_current_best"
    assert rejected["module_effect"] == "confirmed_essential"
    assert weak["module_effect"] == "seed_eval_weak_or_uncertain"


def test_seed_eval_accept_confirms_removal_candidate() -> None:
    result = classify_seed_eval_for_ablation(
        objective_metric="mse_norm",
        seed_eval_result={
            "mean": 0.97,
            "reference_mean": 1.0,
            "valid_metric_seeds": 3,
            "significance_decision": {"decision": "accept"},
        },
    )

    assert result["module_effect"] == "confirmed_harmful_or_redundant"
    assert result["recommended_action"] == "promote_ablation_variant_to_current_best"
