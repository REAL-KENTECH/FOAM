from __future__ import annotations

from tools.run_sweep import build_command


def test_sweep_overrides_are_single_line_yaml_compatible_json() -> None:
    command = build_command(
        {
            "base_config": "configs/vit/paper/foam_f20_tau075_epsmax3e-7.yaml",
            "launcher": "torchrun",
            "nproc_per_node": 4,
        },
        {
            "name": "example",
            "overrides": {
                "optimizer": "stale_shampoo",
                "epsilon": 1.0e-9,
                "use_bias_correction": True,
            },
        },
    )
    override_values = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--set"
    ]
    assert override_values
    assert all("\n" not in value and "..." not in value for value in override_values)
    assert 'optimizer="stale_shampoo"' in override_values
    assert "epsilon=1e-09" in override_values
    assert "use_bias_correction=true" in override_values
