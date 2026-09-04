"""Tests for retrieval ablation configuration generation."""

from __future__ import annotations

from evaluate_ablation import make_ablation_configs


def test_make_ablation_configs_progressively_enables_pipeline_stages():
    base = {
        "retrieval": {
            "hybrid_enabled": True,
            "query_rewrite": True,
            "cross_encoder": {"enabled": True, "model_name": "ce"},
        }
    }

    configs = make_ablation_configs(base)

    assert [name for name, _ in configs] == [
        "dense_only",
        "hybrid",
        "hybrid_rewrite",
        "full",
    ]
    assert configs[0][1]["retrieval"]["hybrid_enabled"] is False
    assert configs[0][1]["retrieval"]["query_rewrite"] is False
    assert configs[0][1]["retrieval"]["cross_encoder"]["enabled"] is False
    assert configs[1][1]["retrieval"]["hybrid_enabled"] is True
    assert configs[1][1]["retrieval"]["query_rewrite"] is False
    assert configs[1][1]["retrieval"]["cross_encoder"]["enabled"] is False
    assert configs[2][1]["retrieval"]["query_rewrite"] is True
    assert configs[2][1]["retrieval"]["cross_encoder"]["enabled"] is False
    assert configs[3][1]["retrieval"]["cross_encoder"]["enabled"] is True
    assert base["retrieval"]["cross_encoder"]["enabled"] is True
