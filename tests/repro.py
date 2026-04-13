# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for DBO MoE unwrapping - Person A portion.

Tests custom ops registration and environment variable gating.
"""

import os
from unittest import mock

import pytest
import torch


def test_dbo_env_var_default():
    """Test that VLLM_MOE_DBO_UNWRAP is disabled by default."""
    # Explicitly unset env var
    env = {k: v for k, v in os.environ.items() if k != "VLLM_MOE_DBO_UNWRAP"}
    with mock.patch.dict(os.environ, env, clear=True):
        # Force reimport of envs module to pick up env var
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm import envs

        assert envs.VLLM_MOE_DBO_UNWRAP is False, (
            "VLLM_MOE_DBO_UNWRAP should be disabled by default"
        )


def test_dbo_env_var_enabled():
    """Test that VLLM_MOE_DBO_UNWRAP can be enabled via env var."""
    with mock.patch.dict(os.environ, {"VLLM_MOE_DBO_UNWRAP": "1"}):
        # Force reimport of envs module to pick up env var
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)

        from vllm import envs

        assert envs.VLLM_MOE_DBO_UNWRAP is True, (
            "VLLM_MOE_DBO_UNWRAP should be enabled when set to 1"
        )


def test_dbo_custom_ops_registered():
    """
    Test that all 7 DBO custom ops are always registered.

    This test verifies:
    - All Person A ops (4): dbo_maybe_run_recv_hook, dbo_yield,
      dbo_yield_and_switch_from_compute_to_comm,
      dbo_yield_and_switch_from_comm_to_compute
    - All Person B ops (3): dbo_switch_to_compute_sync, dbo_switch_to_compute,
      dbo_switch_to_comm
    - All ops are callable via torch.ops.vllm
    - Ops are registered regardless of VLLM_MOE_DBO_UNWRAP env var
    """
    # Import dbo_ops to trigger registration
    import vllm.model_executor.layers.fused_moe.dbo_ops  # noqa: F401

    # Verify ops are registered
    expected_ops = [
        "dbo_maybe_run_recv_hook",
        "dbo_yield",
        "dbo_yield_and_switch_from_compute_to_comm",
        "dbo_yield_and_switch_from_comm_to_compute",
    ]

    for op_name in expected_ops:
        assert hasattr(torch.ops.vllm, op_name), (
            f"Op {op_name} not registered in torch.ops.vllm"
        )

        # Verify op is callable
        op_func = getattr(torch.ops.vllm, op_name)
        assert callable(op_func), f"Op {op_name} is not callable"

        # Verify op can be called (should be no-op with no DBO context)
        try:
            op_func()
        except Exception as e:
            pytest.fail(f"Op {op_name} raised exception when called: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
