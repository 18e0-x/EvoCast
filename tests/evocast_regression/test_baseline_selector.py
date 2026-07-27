from __future__ import annotations

from evocast.research.baseline_selector import select_baseline_candidates
from evocast.research.model_registry import MODEL_FAMILIES, build_registry


def test_model_registry_exposes_only_local_researchable_baselines() -> None:
    registry = build_registry(verify=False)
    keys = {str(item.get("model_key") or "") for item in registry}

    assert all(item.get("local_code") is True for item in registry)
    assert "VAR_model" not in keys
    assert "AutoARIMA" not in keys
    assert "ARIMA" not in keys
    assert "NaiveSeasonal" not in keys
    assert "RandomForest" not in keys
    assert "LightGBMModel" not in keys


def test_manually_verified_model_families_use_others_for_unrepresented_architectures() -> None:
    expected = {
        "FiLM": "others",
        "Koopa": "others",
        "LightTS": "mlp",
        "MICN": "cnn",
        "TimeMixer": "mlp",
        "FreTS": "mlp",
        "TiDE": "mlp",
        "TSMixer": "mlp",
        "SCINet": "cnn",
        "KANAD": "others",
        "WPMixer": "mlp",
        "MSGNet": "gnn",
        "MambaSimple": "others",
        "TimeFilter": "gnn",
        "Amplifier": "others",
        "CMoS": "linear",
        "CrossLinear": "linear",
        "DTAF": "transformer",
        "DUET": "transformer",
        "FITS": "others",
        "HDMixer": "mlp",
        "ModernTCN": "cnn",
        "PDF": "transformer",
        "VAR_model": "others",
        "SparseTSF": "linear",
        "SRSNet": "others",
        "TimeBase": "linear",
        "TimeBridge": "transformer",
        "TimeKAN": "others",
        "TimePerceiver": "transformer",
        "xPatch": "others",
    }
    assert {key: MODEL_FAMILIES[key] for key in expected} == expected
    assert "others" in set(MODEL_FAMILIES.values())


def test_auto_selection_ignores_legacy_initial_seed_priority() -> None:
    config_data = {
        "data_config": {
            "feature_dict": {"if_univariate": False},
            "task_semantics": {"task_mode": "MM"},
        }
    }
    registry = [
        {
            "model_key": "CheapLinear",
            "family": "linear",
            "verified_import": True,
            "local_code": True,
            "supports_multivariate": True,
            "cost_level": 1,
            "reliability": 0,
        },
        {
            "model_key": "SeedTransformer",
            "family": "transformer",
            "verified_import": True,
            "local_code": True,
            "supports_multivariate": True,
            "cost_level": 3,
            "reliability": 2,
        },
    ]

    selected, report = select_baseline_candidates(
        registry=registry,
        config_data=config_data,
        candidate_count=1,
        registry_pool_size=1,
        initial_seeds=["SeedTransformer"],
        preferred_families=["linear", "transformer"],
    )

    assert [item["model_key"] for item in selected] == ["CheapLinear"]
    assert report["ignored_legacy_initial_seeds"] == ["SeedTransformer"]


def test_auto_selection_round_robins_families_and_sorts_within_family() -> None:
    config_data = {
        "data_config": {
            "feature_dict": {"if_univariate": False},
            "task_semantics": {"task_mode": "MM"},
        }
    }
    def spec(model_key: str, family: str, *, cost: int, reliability: int, sentinel: bool) -> dict:
        return {
            "model_key": model_key,
            "family": family,
            "verified_import": True,
            "local_code": True,
            "supports_multivariate": True,
            "cost_level": cost,
            "reliability": reliability,
            "sentinel": sentinel,
        }

    registry = [
        spec("ZLinear", "linear", cost=1, reliability=0, sentinel=True),
        spec("ALinear", "linear", cost=3, reliability=2, sentinel=False),
        spec("ZTransformer", "transformer", cost=1, reliability=0, sentinel=True),
        spec("ATransformer", "transformer", cost=3, reliability=2, sentinel=False),
        spec("AMlp", "mlp", cost=3, reliability=2, sentinel=False),
    ]

    selected, report = select_baseline_candidates(
        registry=registry,
        config_data=config_data,
        candidate_count=5,
        registry_pool_size=1,
        initial_seeds=["ZLinear"],
        preferred_families=["linear", "transformer", "mlp"],
    )

    assert [item["model_key"] for item in selected] == [
        "ALinear",
        "ATransformer",
        "AMlp",
        "ZLinear",
        "ZTransformer",
    ]
    assert report["selection_order"] == "family_order_round_robin_then_alphabetical_model_key"
    assert report["pool_size"] == 5


def test_non_researchable_external_models_are_not_auto_baseline_candidates() -> None:
    config_data = {
        "data_config": {
            "feature_dict": {"if_univariate": False},
            "task_semantics": {"task_mode": "MM"},
        }
    }
    registry = [
        {
            "model_key": "LocalFixture",
            "family": "linear",
            "verified_import": True,
            "local_code": True,
            "supports_multivariate": True,
            "cost_level": 1,
            "reliability": 0,
        },
        {
            "model_key": "AutoARIMA",
            "family": "statistical",
            "verified_import": True,
            "local_code": False,
            "supports_multivariate": True,
            "cost_level": 1,
            "reliability": 0,
        },
    ]

    selected, report = select_baseline_candidates(
        registry=registry,
        config_data=config_data,
        candidate_count=2,
        registry_pool_size=2,
        initial_seeds=[],
        preferred_families=["linear"],
    )

    assert [item["model_key"] for item in selected] == ["LocalFixture"]
    assert any(
        item["model_key"] == "AutoARIMA" and item["reason"] == "not_researchable_external_model"
        for item in report["pool_rejected"]
    )
