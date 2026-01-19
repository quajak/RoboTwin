"""JAX/Flax NNX implementation of DiT model for token augmentation.

This module provides a JAX port of the PyTorch DiT model from src/models/dit.py,
enabling seamless integration with the pi0 training pipeline.
"""

import math
from typing import Any

import flax.nnx as nnx
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


class TimestepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, rngs: nnx.Rngs, frequency_embedding_size: int = 256):
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs)
        self.mlp_2 = nnx.Linear(hidden_size, hidden_size, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        x = self.mlp_0(t_freq)
        x = nnx.silu(x)
        x = self.mlp_2(x)
        return x


class LabelEmbedder(nnx.Module):
    """Embeds class labels into vector representations."""

    def __init__(self, num_classes: int, hidden_size: int, rngs: nnx.Rngs):
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        # Initialize embedding table with normal distribution
        init_fn = nnx.initializers.normal(stddev=0.02)
        self.embedding_table = nnx.Param(
            init_fn(rngs.params(), (num_classes + 1, hidden_size))
        )

    def __call__(self, labels: jnp.ndarray) -> jnp.ndarray:
        return self.embedding_table.value[labels]


class InputEmbedder(nnx.Module):
    """Projects input tokens to hidden dimension."""

    def __init__(self, in_channels: int, hidden_size: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_channels, hidden_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.proj(x)


class MlpBlock(nnx.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, hidden_size: int, mlp_ratio: float, rngs: nnx.Rngs, dropout: float = 0.0):
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nnx.Linear(hidden_size, mlp_hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_hidden_dim, hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class DiTBlock(nnx.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # adaLN modulation
        self.adaLN_modulation_1 = nnx.Linear(hidden_size, 6 * hidden_size, rngs=rngs)

        # Layer norms (no learnable params, just normalization)
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, rngs=rngs)

        # Attention
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout,
            rngs=rngs,
        )

        # MLP
        self.mlp = MlpBlock(hidden_size, mlp_ratio, rngs, dropout=dropout)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, *, deterministic: bool = True) -> jnp.ndarray:
        # adaLN modulation
        modulation = nnx.silu(c)
        modulation = self.adaLN_modulation_1(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        # Self-attention branch
        x_norm1 = self.norm1(x)
        x_modulated1 = modulate(x_norm1, shift_msa, scale_msa)
        attn_output = self.attn(x_modulated1, deterministic=deterministic)
        x = x + gate_msa[:, None, :] * attn_output

        # MLP branch
        x_norm2 = self.norm2(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_modulated2, deterministic=deterministic)
        x = x + gate_mlp[:, None, :] * mlp_output

        return x


class FinalLayer(nnx.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size: int, out_channels: int, rngs: nnx.Rngs):
        self.adaLN_modulation_1 = nnx.Linear(hidden_size, 2 * hidden_size, rngs=rngs)
        self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, rngs=rngs)
        self.linear = nnx.Linear(hidden_size, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        modulation = nnx.silu(c)
        modulation = self.adaLN_modulation_1(modulation)
        shift, scale = jnp.split(modulation, 2, axis=-1)

        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        x = self.linear(x)
        return x


class DiT(nnx.Module):
    """Diffusion Transformer for token augmentation.

    Matches architecture of PyTorch DiT in src/models/dit.py.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        num_classes: int,
        rngs: nnx.Rngs,
        dropout: float = 0.0,
        ignore_dt: bool = False,
    ):
        self.input_size = input_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        self.dropout = dropout
        self.ignore_dt = ignore_dt

        # Embedders
        self.x_embedder = InputEmbedder(in_channels, hidden_size, rngs)
        self.t_embedder = TimestepEmbedder(hidden_size, rngs)
        self.dt_embedder = TimestepEmbedder(hidden_size, rngs)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, rngs)

        # Positional embedding (frozen parameter)
        self.pos_embed = nnx.Param(get_1d_sincos_pos_embed(hidden_size, input_size))

        # Transformer blocks
        self.blocks = [
            DiTBlock(hidden_size, num_heads, rngs, mlp_ratio, dropout)
            for _ in range(depth)
        ]

        # Final layer
        self.final_layer = FinalLayer(hidden_size, in_channels, rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        dt: jnp.ndarray,
        y: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass matching PyTorch signature.

        Args:
            x: Input tokens (N, L, C)
            t: Timesteps (N,)
            dt: Delta timestep indices (N,)
            y: Class labels (N,)
            deterministic: Whether to use deterministic dropout
        """
        x = self.x_embedder(x) + self.pos_embed.value

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
