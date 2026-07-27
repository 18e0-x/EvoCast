from __future__ import annotations

from types import SimpleNamespace

from evocast.harness.session import AgentSession
from evocast.tools.model_structure import analyze_model_structure, propose_ablation_targets


def test_crossformer_runtime_mechanism_candidates_are_generated(monkeypatch) -> None:
    source_file = "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py"
    decoder_source = "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py"

    def fake_runtime(*_args, **_kwargs):
        names = [
            ("encoder.encode_blocks.0.encode_layers.0", "TwoStageAttentionLayer", source_file),
            ("encoder.encode_blocks.0.encode_layers.0.time_attention", "AttentionLayer", source_file),
            ("encoder.encode_blocks.0.encode_layers.0.dim_sender", "AttentionLayer", source_file),
            ("encoder.encode_blocks.0.encode_layers.0.dim_receiver", "AttentionLayer", source_file),
            ("decoder.decode_layers.0", "DecoderLayer", decoder_source),
            ("decoder.decode_layers.0.self_attention", "TwoStageAttentionLayer", source_file),
            ("decoder.decode_layers.0.cross_attention", "AttentionLayer", source_file),
            ("decoder.decode_layers.0.MLP1", "Sequential", decoder_source),
            ("decoder.decode_layers.0.linear_pred", "Linear", decoder_source),
        ]
        called_modules = [
            {
                "name": name,
                "input": {"kind": "tuple", "items": [{"kind": "tensor", "shape": [2, 4, 8, 64]}]},
                "output": {"kind": "tensor", "shape": [2, 4, 8, 64]},
            }
            for name, _module_type, _source in names
            if "." in name
        ]
        return {
            "status": "ok",
            "task_name": "long_term_forecast",
            "loader_model_hyper_params": {"seq_len": 96, "pred_len": 24, "seg_len": 6, "enc_in": 4, "dec_in": 4, "c_out": 4},
            "modules": [
                {
                    "name": name,
                    "type": module_type,
                    "class_name": module_type,
                    "class_module": "fake.crossformer",
                    "class_path": f"fake.crossformer.{module_type}",
                    "source_file": source,
                }
                for name, module_type, source in names
            ],
            "parameters": [
                {"name": "encoder.encode_blocks.0.encode_layers.0.time_attention.query_projection.weight", "shape": [64, 64], "requires_grad": True},
                {"name": "decoder.decode_layers.0.cross_attention.query_projection.weight", "shape": [64, 64], "requires_grad": True},
            ],
            "buffers": [],
            "active_path_probe": {
                "status": "ok",
                "called_module_count": len(called_modules),
                "called_modules": called_modules,
                "called_module_names": [item["name"] for item in called_modules],
                "input_shape": [2, 96, 4],
                "output_shape": [2, 24, 4],
            },
        }

    monkeypatch.setattr("evocast.tools.model_structure._runtime_introspection_via_tfb_loader", fake_runtime)

    session = AgentSession(
        task_id="test_deep_mechanism_ablation_targets",
        base_dir=r"C:\CODEX\TFB-master",
        client=SimpleNamespace(api_available=False),
    )
    session.ensure_dirs()
    analysis = analyze_model_structure(session, {"model_key": "Crossformer", "run_shape_probe": False})

    by_path = {
        str(item.get("path") or ""): item
        for item in list(analysis.get("candidate_fit_points") or [])
        if isinstance(item, dict)
    }

    expected = {
        "self.encoder.encode_blocks[*].encode_layers[*].time_attention": ("temporal_attention", "TwoStageAttentionLayer"),
        "self.encoder.encode_blocks[*].encode_layers[*].dim_sender": ("variable_attention", "TwoStageAttentionLayer"),
        "self.encoder.encode_blocks[*].encode_layers[*].dim_receiver": ("variable_attention", "TwoStageAttentionLayer"),
        "self.decoder.decode_layers[*].self_attention": ("decoder_self_attention", "DecoderLayer"),
        "self.decoder.decode_layers[*].cross_attention": ("decoder_cross_attention", "DecoderLayer"),
    }
    for path, (family, owner_class) in expected.items():
        assert path in by_path
        item = by_path[path]
        assert item["mechanism_family"] == family
        assert item["owner_class"] == owner_class
        assert item["owner_source_file"]
        assert item["local_component_path"]
        assert item["forward_reachable"] is True
        assert item["same_input_observable"] is True
        assert item["expanded_runtime_paths"]


def test_propose_ablation_targets_prioritizes_encoder_mechanisms_and_filters_noise() -> None:
    session = SimpleNamespace(task_id="proposal_order")
    analysis = {
        "model_key": "Crossformer",
        "mechanism_candidates": [
            {
                "path": "self.decoder.decode_layers[*]",
                "mechanism_family": "transformer_block",
                "granularity": "mechanism",
                "owner_class": "ModuleList",
                "owner_source_file": r"C:\Python\Lib\site-packages\torch\nn\modules\container.py",
                "local_component_path": "0",
                "source_file": "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
            {
                "path": "self.decoder.decode_layers[*].cross_attention.inner_attention",
                "mechanism_family": "attention",
                "granularity": "mechanism",
                "owner_class": "AttentionLayer",
                "owner_source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "local_component_path": "inner_attention",
                "source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
            {
                "path": "self.decoder.decode_layers[*].cross_attention",
                "mechanism_family": "decoder_cross_attention",
                "granularity": "mechanism",
                "owner_class": "DecoderLayer",
                "owner_source_file": "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
                "local_component_path": "cross_attention",
                "expanded_runtime_paths": ["self.decoder.decode_layers.0.cross_attention"],
                "source_file": "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
            {
                "path": "self.decoder.decode_layers[*].self_attention",
                "mechanism_family": "decoder_self_attention",
                "granularity": "mechanism",
                "owner_class": "DecoderLayer",
                "owner_source_file": "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
                "local_component_path": "self_attention",
                "expanded_runtime_paths": ["self.decoder.decode_layers.0.self_attention"],
                "source_file": "ts_benchmark/baselines/time_series_library/layers/Crossformer_EncDec.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
            {
                "path": "self.encoder.encode_blocks[*].encode_layers[*].time_attention",
                "mechanism_family": "temporal_attention",
                "granularity": "mechanism",
                "owner_class": "TwoStageAttentionLayer",
                "owner_source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "local_component_path": "time_attention",
                "expanded_runtime_paths": ["self.encoder.encode_blocks.0.encode_layers.0.time_attention"],
                "source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
            {
                "path": "self.encoder.encode_blocks[*].encode_layers[*].dim_sender",
                "mechanism_family": "variable_attention",
                "granularity": "mechanism",
                "owner_class": "TwoStageAttentionLayer",
                "owner_source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "local_component_path": "dim_sender",
                "expanded_runtime_paths": ["self.encoder.encode_blocks.0.encode_layers.0.dim_sender"],
                "source_file": "ts_benchmark/baselines/time_series_library/layers/SelfAttention_Family.py",
                "forward_reachable": True,
                "same_input_observable": True,
                "confidence": 0.9,
            },
        ],
        "safe_fit_points": [],
        "forward": {"called_components": []},
        "source_files": [{"path": "ts_benchmark/baselines/time_series_library/models/Crossformer.py"}],
    }

    proposal = propose_ablation_targets(session, {"analysis": analysis, "model_key": "Crossformer", "max_targets": 6})
    paths = [item["component_path"] for item in proposal["targets"]]

    assert "self.decoder.decode_layers[*]" not in paths
    assert "self.decoder.decode_layers[*].cross_attention.inner_attention" not in paths
    assert "self.encoder.encode_blocks[*].encode_layers[*].time_attention" in paths[:4]
    assert "self.encoder.encode_blocks[*].encode_layers[*].dim_sender" in paths[:4]


def test_runtime_mechanism_candidates_canonicalize_inner_model_paths(monkeypatch) -> None:
    def fake_runtime(*_args, **_kwargs):
        names = [
            ("cluster.gate.distribution_fit", "Sequential", "ts_benchmark/baselines/duet/layers/distributional_router_encoder.py"),
            ("cluster.noise.distribution_fit", "Sequential", "ts_benchmark/baselines/duet/layers/distributional_router_encoder.py"),
        ]
        called_modules = [
            {
                "name": name,
                "input": {"kind": "tuple", "items": [{"kind": "tensor", "shape": [2, 96, 8]}]},
                "output": {"kind": "tensor", "shape": [2, 4]},
            }
            for name, _module_type, _source in names
        ]
        return {
            "status": "ok",
            "task_name": "long_term_forecast",
            "modules": [
                {
                    "name": name,
                    "type": module_type,
                    "class_name": module_type,
                    "class_module": "fake.duet",
                    "class_path": f"fake.duet.{module_type}",
                    "source_file": source,
                }
                for name, module_type, source in names
            ],
            "inner_modules": [
                {
                    "name": f"model.{name}",
                    "type": module_type,
                    "class_name": module_type,
                    "class_module": "fake.duet",
                    "class_path": f"fake.duet.{module_type}",
                    "source_file": source,
                }
                for name, module_type, source in names
            ],
            "parameters": [],
            "inner_parameters": [],
            "buffers": [],
            "inner_buffers": [],
            "active_path_probe": {
                "status": "ok",
                "called_module_count": len(called_modules),
                "called_modules": called_modules,
                "called_module_names": [item["name"] for item in called_modules],
                "input_shape": [2, 96, 8],
                "output_shape": [2, 96, 8],
            },
        }

    monkeypatch.setattr("evocast.tools.model_structure._runtime_introspection_from_adapter", fake_runtime)

    session = AgentSession(
        task_id="test_duet_runtime_inner_path_canonicalization",
        base_dir=r"D:\EvoCast",
        client=SimpleNamespace(api_available=False),
    )
    session.ensure_dirs()
    analysis = analyze_model_structure(session, {"model_key": "DUET", "run_shape_probe": False, "force_refresh": True})

    mechanism_paths = {
        str(item.get("path") or ""): item
        for item in list(analysis.get("mechanism_candidates") or [])
        if isinstance(item, dict)
    }

    assert "self.model.cluster.gate.distribution_fit" in mechanism_paths
    assert "self.cluster.gate.distribution_fit" not in mechanism_paths
    assert mechanism_paths["self.model.cluster.gate.distribution_fit"]["display_component_path"] == "self.cluster.gate.distribution_fit"


def test_duet_static_component_recursion_surfaces_distribution_fit() -> None:
    session = AgentSession(
        task_id="test_duet_static_distribution_fit",
        base_dir=r"D:\EvoCast",
        client=SimpleNamespace(api_available=False),
    )
    session.ensure_dirs()
    analysis = analyze_model_structure(session, {"model_key": "DUET", "run_shape_probe": False, "force_refresh": True})

    component_names = {str(item.get("name") or "") for item in list(analysis.get("components") or []) if isinstance(item, dict)}

    assert "model.cluster.gate.distribution_fit" in component_names
    assert "model.cluster.noise.distribution_fit" in component_names
