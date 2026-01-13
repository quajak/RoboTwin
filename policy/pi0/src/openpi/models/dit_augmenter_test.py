"""Tests to verify JAX and PyTorch DiT implementations produce identical outputs.

These tests ensure numerical equivalence between the JAX implementation in dit_augmenter.py
and the PyTorch implementation in src/models/dit.py.
"""

import sys
import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp

# Add the robustness-generation root to path to import PyTorch modules
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
sys.path.insert(0, REPO_ROOT)

import torch
from src.models.dit import DiT as PyTorchDiT, TimestepEmbedder as PyTorchTimestepEmbedder, DiTBlock as PyTorchDiTBlock
from src.utils.inference import solve_ode_flow_matching as pytorch_solve_ode
from src.utils.math_utils import modulate as pytorch_modulate

from openpi.models.dit_augmenter import (
    DiT as JAXDiT,
    TimestepEmbedder as JAXTimestepEmbedder,
    DiTBlock as JAXDiTBlock,
    timestep_embedding as jax_timestep_embedding,
    modulate as jax_modulate,
    solve_ode_flow_matching_jax,
    get_1d_sincos_pos_embed,
)
from openpi.models.dit_weight_loader import load_pytorch_dit_weights


def assert_allclose_jax_torch(jax_arr: jnp.ndarray, torch_arr: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5, msg: str = "") -> None:
    """Compare JAX array with PyTorch tensor."""
    jax_np = np.array(jax_arr)
    torch_np = torch_arr.detach().cpu().numpy()
    np.testing.assert_allclose(jax_np, torch_np, rtol=rtol, atol=atol, err_msg=msg)


def create_test_inputs(batch_size: int = 2, seq_len: int = 256, hidden_dim: int = 1152, seed: int = 42) -> tuple[jnp.ndarray, torch.Tensor]:
    """Create identical test inputs for both frameworks."""
    np.random.seed(seed)
    tokens = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
    return jnp.array(tokens), torch.from_numpy(tokens.copy())


class TestTimestepEmbedding:
    """Tests for timestep embedding functions."""

    @pytest.mark.parametrize("timestep", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_timestep_embedding_parity(self, timestep: float) -> None:
        """Verify timestep embedding produces identical outputs."""
        batch_size = 4
        dim = 256

        # Create inputs
        t_jax = jnp.full((batch_size,), timestep, dtype=jnp.float32)
        t_torch = torch.full((batch_size,), timestep, dtype=torch.float32)

        # Compute embeddings
        jax_emb = jax_timestep_embedding(t_jax, dim)
        torch_emb = PyTorchTimestepEmbedder.timestep_embedding(t_torch, dim)

        assert_allclose_jax_torch(jax_emb, torch_emb, rtol=1e-5, atol=1e-5, msg=f"Timestep embedding mismatch at t={timestep}")

    def test_timestep_embedding_random_values(self) -> None:
        """Test timestep embedding with random values."""
        np.random.seed(42)
        batch_size = 8
        dim = 256

        t_values = np.random.uniform(0, 1, batch_size).astype(np.float32)
        t_jax = jnp.array(t_values)
        t_torch = torch.from_numpy(t_values)

        jax_emb = jax_timestep_embedding(t_jax, dim)
        torch_emb = PyTorchTimestepEmbedder.timestep_embedding(t_torch, dim)

        assert_allclose_jax_torch(jax_emb, torch_emb, rtol=1e-5, atol=1e-5, msg="Random timestep embedding mismatch")


class TestModulate:
    """Tests for modulate function."""

    def test_modulate_parity(self) -> None:
        """Verify modulate function produces identical outputs."""
        np.random.seed(42)
        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        shift = np.random.randn(batch_size, hidden_dim).astype(np.float32)
        scale = np.random.randn(batch_size, hidden_dim).astype(np.float32) * 0.5  # Keep scale small to test clipping

        x_jax, x_torch = jnp.array(x), torch.from_numpy(x.copy())
        shift_jax, shift_torch = jnp.array(shift), torch.from_numpy(shift.copy())
        scale_jax, scale_torch = jnp.array(scale), torch.from_numpy(scale.copy())

        jax_out = jax_modulate(x_jax, shift_jax, scale_jax)
        torch_out = pytorch_modulate(x_torch, shift_torch, scale_torch)

        assert_allclose_jax_torch(jax_out, torch_out, rtol=1e-5, atol=1e-5, msg="Modulate function mismatch")

    def test_modulate_clipping(self) -> None:
        """Test that modulate clips scale values correctly."""
        np.random.seed(42)
        batch_size = 2
        seq_len = 16
        hidden_dim = 64

        x = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        shift = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        scale = np.ones((batch_size, hidden_dim), dtype=np.float32) * 2.0  # Large scale to test clipping

        x_jax, x_torch = jnp.array(x), torch.from_numpy(x.copy())
        shift_jax, shift_torch = jnp.array(shift), torch.from_numpy(shift.copy())
        scale_jax, scale_torch = jnp.array(scale), torch.from_numpy(scale.copy())

        jax_out = jax_modulate(x_jax, shift_jax, scale_jax)
        torch_out = pytorch_modulate(x_torch, shift_torch, scale_torch)

        assert_allclose_jax_torch(jax_out, torch_out, rtol=1e-5, atol=1e-5, msg="Modulate clipping mismatch")


class TestPositionalEmbedding:
    """Tests for positional embedding."""

    def test_pos_embed_shape(self) -> None:
        """Verify positional embedding has correct shape."""
        embed_dim = 768
        length = 256

        pos_embed = get_1d_sincos_pos_embed(embed_dim, length)

        assert pos_embed.shape == (1, length, embed_dim), f"Expected shape (1, {length}, {embed_dim}), got {pos_embed.shape}"

    def test_pos_embed_values(self) -> None:
        """Verify positional embedding values are in expected range."""
        embed_dim = 768
        length = 256

        pos_embed = get_1d_sincos_pos_embed(embed_dim, length)

        # Sin/cos values should be in [-1, 1]
        assert jnp.all(pos_embed >= -1.0) and jnp.all(pos_embed <= 1.0), "Positional embedding values out of range"


class TestDiTForward:
    """Tests for DiT forward pass."""

    @pytest.fixture
    def model_config(self) -> dict:
        """Common model configuration for tests."""
        return {
            "input_size": 64,
            "in_channels": 128,
            "hidden_size": 256,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "class_dropout_prob": 0.0,
            "num_classes": 10,
            "dropout": 0.0,
        }

    def test_dit_output_shape(self, model_config: dict) -> None:
        """Verify DiT output has correct shape."""
        batch_size = 2

        # Create JAX model
        jax_model = JAXDiT(
            input_size=model_config["input_size"],
            in_channels=model_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )

        # Create inputs
        np.random.seed(42)
        x = np.random.randn(batch_size, model_config["input_size"], model_config["in_channels"]).astype(np.float32)
        t = np.random.uniform(0, 1, batch_size).astype(np.float32)
        dt = np.zeros(batch_size, dtype=np.int32)
        y = np.zeros(batch_size, dtype=np.int32)

        # Initialize and run
        rng = jax.random.key(0)
        variables = jax_model.init(rng, jnp.array(x), jnp.array(t), jnp.array(dt), jnp.array(y))
        output = jax_model.apply(variables, jnp.array(x), jnp.array(t), jnp.array(dt), jnp.array(y))

        expected_shape = (batch_size, model_config["input_size"], model_config["in_channels"])
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


class TestODESolver:
    """Tests for ODE flow matching solver."""

    @pytest.fixture
    def model_config(self) -> dict:
        """Common model configuration for tests."""
        return {
            "input_size": 32,
            "in_channels": 64,
            "hidden_size": 128,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
            "class_dropout_prob": 0.0,
            "num_classes": 10,
            "dropout": 0.0,
        }

    def test_ode_solver_output_shape(self, model_config: dict) -> None:
        """Verify ODE solver output has correct shape."""
        batch_size = 2
        steps = 4

        # Create JAX model
        jax_model = JAXDiT(
            input_size=model_config["input_size"],
            in_channels=model_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )

        # Create inputs
        np.random.seed(42)
        x = np.random.randn(batch_size, model_config["input_size"], model_config["in_channels"]).astype(np.float32)

        # Initialize model
        rng = jax.random.key(0)
        dummy_t = jnp.zeros(batch_size)
        dummy_dt = jnp.zeros(batch_size, dtype=jnp.int32)
        dummy_y = jnp.zeros(batch_size, dtype=jnp.int32)
        variables = jax_model.init(rng, jnp.array(x), dummy_t, dummy_dt, dummy_y)

        # Create model function
        def model_fn(x, t, dt, y):
            return jax_model.apply(variables, x, t, dt, y, deterministic=True)

        # Run ODE solver
        output = solve_ode_flow_matching_jax(model_fn, jnp.array(x), steps=steps)

        expected_shape = (batch_size, model_config["input_size"], model_config["in_channels"])
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    @pytest.mark.parametrize("steps", [1, 4, 8, 16])
    def test_ode_solver_different_steps(self, model_config: dict, steps: int) -> None:
        """Test ODE solver with different step counts."""
        batch_size = 2

        # Create JAX model
        jax_model = JAXDiT(
            input_size=model_config["input_size"],
            in_channels=model_config["in_channels"],
            hidden_size=model_config["hidden_size"],
            depth=model_config["depth"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )

        # Create inputs
        np.random.seed(42)
        x = np.random.randn(batch_size, model_config["input_size"], model_config["in_channels"]).astype(np.float32)

        # Initialize model
        rng = jax.random.key(0)
        dummy_t = jnp.zeros(batch_size)
        dummy_dt = jnp.zeros(batch_size, dtype=jnp.int32)
        dummy_y = jnp.zeros(batch_size, dtype=jnp.int32)
        variables = jax_model.init(rng, jnp.array(x), dummy_t, dummy_dt, dummy_y)

        # Create model function
        def model_fn(x, t, dt, y):
            return jax_model.apply(variables, x, t, dt, y, deterministic=True)

        # Run ODE solver - should not raise
        output = solve_ode_flow_matching_jax(model_fn, jnp.array(x), steps=steps)

        assert output.shape == x.shape, f"Output shape mismatch for steps={steps}"
        assert not jnp.any(jnp.isnan(output)), f"NaN in output for steps={steps}"
        assert not jnp.any(jnp.isinf(output)), f"Inf in output for steps={steps}"


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_no_nan_with_small_inputs(self) -> None:
        """Test that small inputs don't produce NaN."""
        batch_size = 2
        dim = 256

        t_small = jnp.full((batch_size,), 1e-6, dtype=jnp.float32)
        emb = jax_timestep_embedding(t_small, dim)

        assert not jnp.any(jnp.isnan(emb)), "NaN produced with small timestep"
        assert not jnp.any(jnp.isinf(emb)), "Inf produced with small timestep"

    def test_no_nan_with_large_inputs(self) -> None:
        """Test that large inputs don't produce NaN."""
        batch_size = 2
        dim = 256

        t_large = jnp.full((batch_size,), 0.9999, dtype=jnp.float32)
        emb = jax_timestep_embedding(t_large, dim)

        assert not jnp.any(jnp.isnan(emb)), "NaN produced with large timestep"
        assert not jnp.any(jnp.isinf(emb)), "Inf produced with large timestep"


class TestWeightLoading:
    """Tests for weight loading functionality."""

    def test_weight_loader_import(self) -> None:
        """Test that weight loader can be imported."""
        from openpi.models.dit_weight_loader import load_pytorch_dit_weights, load_dit_from_pytorch_checkpoint

        assert callable(load_pytorch_dit_weights)
        assert callable(load_dit_from_pytorch_checkpoint)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
