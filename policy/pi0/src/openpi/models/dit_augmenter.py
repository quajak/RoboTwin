"""JAX/Flax implementation of DiT model for token augmentation.

This module provides a JAX port of the PyTorch DiT model from src/models/dit.py,
enabling seamless integration with the pi0 training pipeline.
"""

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at


def timestep_embedding(t: jnp.ndarray, dim: int, max_period: int = 10000) -> jnp.ndarray:
    """Create sinusoidal timestep embeddings matching PyTorch implementation exactly."""
    half = dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Apply adaptive layer norm modulation matching PyTorch implementation."""
    scale = jnp.clip(scale, -1, 1)
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> jnp.ndarray:
    """Generate 1D sinusoidal positional embeddings."""
    pos = jnp.arange(length, dtype=jnp.float32)
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb[None, :, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        x = nn.Dense(self.hidden_size, name="mlp_0")(t_freq)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, name="mlp_2")(x)
        return x


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    num_classes: int
    hidden_size: int

    @nn.compact
    def __call__(self, labels: jnp.ndarray) -> jnp.ndarray:
        embedding_table = self.param(
            "embedding_table",
            nn.initializers.normal(stddev=0.02),
            (self.num_classes + 1, self.hidden_size)
        )
        return embedding_table[labels]


class InputEmbedder(nn.Module):
    """Projects input tokens to hidden dimension."""
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.hidden_size, name="proj")(x)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    hidden_size: int
    mlp_ratio: float
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        x = nn.Dense(mlp_hidden_dim, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        x = nn.Dense(self.hidden_size, name="fc2")(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)
        return x


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # adaLN modulation
        modulation = nn.silu(c)
        modulation = nn.Dense(6 * self.hidden_size, name="adaLN_modulation_1")(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-attention branch
        x_norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="norm1")(x)
        x_modulated1 = modulate(x_norm1, shift_msa, scale_msa)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            name="attn"
        )(x_modulated1, x_modulated1)
        x = x + gate_msa[:, None, :] * attn_output

        # MLP branch
        x_norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="norm2")(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_output = MlpBlock(
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
            name="mlp"
        )(x_modulated2, deterministic=deterministic)
        x = x + gate_mlp[:, None, :] * mlp_output

        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    hidden_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        modulation = nn.silu(c)
        modulation = nn.Dense(2 * self.hidden_size, name="adaLN_modulation_1")(modulation)
        shift, scale = jnp.split(modulation, 2, axis=-1)

        x = nn.LayerNorm(epsilon=1e-6, use_bias=False, use_scale=False, name="norm_final")(x)
        x = modulate(x, shift, scale)
        x = nn.Dense(self.out_channels, name="linear")(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer for token augmentation.

    Matches architecture of PyTorch DiT in src/models/dit.py.
    """
    input_size: int
    in_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    num_classes: int
    dropout: float = 0.0
    ignore_dt: bool = False

    def setup(self):
        self.x_embedder = InputEmbedder(hidden_size=self.hidden_size, name="x_embedder")
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size, name="t_embedder")
        self.dt_embedder = TimestepEmbedder(hidden_size=self.hidden_size, name="dt_embedder")
        self.y_embedder = LabelEmbedder(num_classes=self.num_classes, hidden_size=self.hidden_size, name="y_embedder")

        self.blocks = [
            DiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                name=f"blocks_{i}"
            )
            for i in range(self.depth)
        ]
        self.final_layer = FinalLayer(hidden_size=self.hidden_size, out_channels=self.in_channels, name="final_layer")

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, dt: jnp.ndarray, y: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Forward pass matching PyTorch signature.

        Args:
            x: Input tokens (N, L, C)
            t: Timesteps (N,)
            dt: Delta timestep indices (N,)
            y: Class labels (N,)
            deterministic: Whether to use deterministic dropout
        """
        # Positional embedding (frozen parameter)
        pos_embed = self.param(
            "pos_embed",
            lambda rng, shape: get_1d_sincos_pos_embed(self.hidden_size, self.input_size),
            (1, self.input_size, self.hidden_size)
        )

        x = self.x_embedder(x) + pos_embed

        t_emb = self.t_embedder(t)
        if self.ignore_dt:
            dt_emb = 0
        else:
            dt_emb = self.dt_embedder(dt.astype(jnp.float32))
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb + dt_emb

        for block in self.blocks:
            x = block(x, c, deterministic=deterministic)

        x = self.final_layer(x, c)
        return x


@at.typecheck
def solve_ode_flow_matching_jax(
    model_fn: Any,
    x: at.Float[at.Array, "b l c"],
    steps: int = 50,
    class_label: int = 0
) -> at.Float[at.Array, "b l c"]:
    """Solve ODE dx/dt = v(x, t) from t=0 to t=1 using Euler method.

    JAX version matching PyTorch solve_ode_flow_matching exactly.

    Args:
        model_fn: Function that takes (x, t, dt, y) and returns velocity
        x: Initial state x(0) [B, L, C]
        steps: Number of integration steps
        class_label: Label to pass to the model

    Returns:
        x(1): The estimated state at t=1
    """
    batch_size = x.shape[0]
    dt_step = 1.0 / steps

    # Compute dt_idx like PyTorch version
    if steps > 0:
        dt_log2 = np.log2(steps)
        dt_idx_val = int(round(dt_log2))
    else:
        dt_idx_val = 0

    dt_idx = jnp.full((batch_size,), dt_idx_val, dtype=jnp.int32)
    y = jnp.full((batch_size,), class_label, dtype=jnp.int32)

    def step_fn(carry, i):
        cur_x = carry
        t_curr = i / steps
        t_tensor = jnp.full((batch_size,), t_curr, dtype=jnp.float32)

        v = model_fn(cur_x, t_tensor, dt_idx, y)
        new_x = cur_x + v * dt_step
        return new_x, None

    final_x, _ = jax.lax.scan(step_fn, x.astype(jnp.float32), jnp.arange(steps))
    return final_x
