"""Weight loader for converting PyTorch DiT checkpoints to JAX/NNX format.

This module handles the conversion of PyTorch checkpoint weights to JAX-compatible
format for use with the NNX DiT implementation in dit_augmenter.py.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models.dit_augmenter import (
    DiT,
    DiTBlock,
    FinalLayer,
    InputEmbedder,
    LabelEmbedder,
    MlpBlock,
    TimestepEmbedder,
)


def _convert_linear_weights(pt_weight: np.ndarray, pt_bias: np.ndarray | None) -> dict[str, np.ndarray]:
    """Convert PyTorch linear layer weights to JAX format.
    
    PyTorch stores weights as (out, in), JAX expects (in, out).
    """
    result = {"kernel": pt_weight.T}
    if pt_bias is not None:
        result["bias"] = pt_bias
    return result


def _get_linear_weights(pt_state: dict[str, np.ndarray], prefix: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract linear layer weights from PyTorch state dict."""
    weight = pt_state[f"{prefix}.weight"]
    bias = pt_state.get(f"{prefix}.bias")
    return weight, bias


def load_pytorch_dit_weights(checkpoint_path: str) -> dict[str, Any]:
    """Load PyTorch DiT checkpoint and convert to JAX-compatible format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint file (.pt)

    Returns:
        Dictionary of JAX-compatible weights matching DiT module structure
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle both direct state_dict and wrapped checkpoint formats
    if "ema" in checkpoint:
        pt_state = checkpoint["ema"]
    elif "model" in checkpoint:
        pt_state = checkpoint["model"]
    else:
        pt_state = checkpoint

    # Convert all tensors to numpy
    pt_state = {k: v.numpy() for k, v in pt_state.items()}

    return pt_state


def _load_timestep_embedder(embedder: TimestepEmbedder, pt_state: dict[str, np.ndarray], prefix: str) -> None:
    """Load weights into a TimestepEmbedder."""
    # mlp_0
    w, b = _get_linear_weights(pt_state, f"{prefix}.mlp.0")
    embedder.mlp_0.kernel.value = jnp.array(w.T)
    embedder.mlp_0.bias.value = jnp.array(b)
    
    # mlp_2
    w, b = _get_linear_weights(pt_state, f"{prefix}.mlp.2")
    embedder.mlp_2.kernel.value = jnp.array(w.T)
    embedder.mlp_2.bias.value = jnp.array(b)


def _load_label_embedder(embedder: LabelEmbedder, pt_state: dict[str, np.ndarray]) -> None:
    """Load weights into a LabelEmbedder."""
    embedder.embedding_table.value = jnp.array(pt_state["y_embedder.embedding_table.weight"])


def _load_input_embedder(embedder: InputEmbedder, pt_state: dict[str, np.ndarray]) -> None:
    """Load weights into an InputEmbedder."""
    w, b = _get_linear_weights(pt_state, "x_embedder.proj")
    embedder.proj.kernel.value = jnp.array(w.T)
    embedder.proj.bias.value = jnp.array(b)


def _load_mlp_block(mlp: MlpBlock, pt_state: dict[str, np.ndarray], prefix: str) -> None:
    """Load weights into an MlpBlock."""
    # fc1
    w, b = _get_linear_weights(pt_state, f"{prefix}.fc1")
    mlp.fc1.kernel.value = jnp.array(w.T)
    mlp.fc1.bias.value = jnp.array(b)
    
    # fc2
    w, b = _get_linear_weights(pt_state, f"{prefix}.fc2")
    mlp.fc2.kernel.value = jnp.array(w.T)
    mlp.fc2.bias.value = jnp.array(b)


def _load_dit_block(block: DiTBlock, pt_state: dict[str, np.ndarray], block_idx: int) -> None:
    """Load weights into a DiTBlock."""
    prefix = f"blocks.{block_idx}"
    
    # adaLN modulation
    w, b = _get_linear_weights(pt_state, f"{prefix}.adaLN_modulation.1")
    block.adaLN_modulation_1.kernel.value = jnp.array(w.T)
    block.adaLN_modulation_1.bias.value = jnp.array(b)
    
    # Attention - PyTorch uses in_proj_weight (3*hidden, hidden) and out_proj
    in_proj_weight = pt_state[f"{prefix}.attn.in_proj_weight"]
    in_proj_bias = pt_state[f"{prefix}.attn.in_proj_bias"]
    out_proj_weight = pt_state[f"{prefix}.attn.out_proj.weight"]
    out_proj_bias = pt_state[f"{prefix}.attn.out_proj.bias"]
    
    hidden_size = out_proj_weight.shape[0]
    num_heads = block.num_heads
    head_dim = hidden_size // num_heads
    
    # Split in_proj into q, k, v
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3, axis=0)
    
    # NNX MultiHeadAttention expects (in_features, num_heads, head_dim) for q, k, v kernels
    # and (num_heads, head_dim) for biases
    block.attn.query.kernel.value = jnp.array(q_weight.T.reshape(hidden_size, num_heads, head_dim))
    block.attn.query.bias.value = jnp.array(q_bias.reshape(num_heads, head_dim))
    
    block.attn.key.kernel.value = jnp.array(k_weight.T.reshape(hidden_size, num_heads, head_dim))
    block.attn.key.bias.value = jnp.array(k_bias.reshape(num_heads, head_dim))
    
    block.attn.value.kernel.value = jnp.array(v_weight.T.reshape(hidden_size, num_heads, head_dim))
    block.attn.value.bias.value = jnp.array(v_bias.reshape(num_heads, head_dim))
    
    # Output projection: NNX expects (num_heads, head_dim, out_features)
    block.attn.out.kernel.value = jnp.array(out_proj_weight.T.reshape(num_heads, head_dim, hidden_size))
    block.attn.out.bias.value = jnp.array(out_proj_bias)
    
    # MLP
    _load_mlp_block(block.mlp, pt_state, f"{prefix}.mlp")


def _load_final_layer(final_layer: FinalLayer, pt_state: dict[str, np.ndarray]) -> None:
    """Load weights into a FinalLayer."""
    # adaLN modulation
    w, b = _get_linear_weights(pt_state, "final_layer.adaLN_modulation.1")
    final_layer.adaLN_modulation_1.kernel.value = jnp.array(w.T)
    final_layer.adaLN_modulation_1.bias.value = jnp.array(b)
    
    # linear
    w, b = _get_linear_weights(pt_state, "final_layer.linear")
    final_layer.linear.kernel.value = jnp.array(w.T)
    final_layer.linear.bias.value = jnp.array(b)


def load_weights_into_dit(model: DiT, pt_state: dict[str, np.ndarray]) -> None:
    """Load PyTorch weights into an NNX DiT model.
    
    Args:
        model: The NNX DiT model to load weights into
        pt_state: PyTorch state dict (already converted to numpy)
    """
    # Load embedders
    _load_timestep_embedder(model.t_embedder, pt_state, "t_embedder")
    _load_timestep_embedder(model.dt_embedder, pt_state, "dt_embedder")
    _load_label_embedder(model.y_embedder, pt_state)
    _load_input_embedder(model.x_embedder, pt_state)
    
    # Load positional embedding
    model.pos_embed.value = jnp.array(pt_state["pos_embed"])
    
    # Load blocks
    for i, block in enumerate(model.blocks):
        _load_dit_block(block, pt_state, i)
    
    # Load final layer
    _load_final_layer(model.final_layer, pt_state)


def load_dit_from_pytorch_checkpoint(
    checkpoint_path: str,
    input_size: int = 256,
    in_channels: int = 1152,
    hidden_size: int = 768,
    depth: int = 6,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    num_classes: int = 10,
    dropout: float = 0.0,
    ignore_dt: bool = False,
    rngs: nnx.Rngs | None = None,
) -> DiT:
    """Create a DiT model and load weights from PyTorch checkpoint.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        input_size: Number of input tokens (sequence length)
        in_channels: Input token dimension
        hidden_size: Hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        num_classes: Number of classes for conditioning
        dropout: Dropout rate
        ignore_dt: Whether to ignore dt embedding
        rngs: NNX random number generators (created if None)

    Returns:
        DiT model with loaded weights
    """
    if rngs is None:
        rngs = nnx.Rngs(jax.random.key(0))
    
    # Create model
    model = DiT(
        input_size=input_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        rngs=rngs,
        dropout=dropout,
        ignore_dt=ignore_dt,
    )
    
    # Load weights
    pt_state = load_pytorch_dit_weights(checkpoint_path)
    load_weights_into_dit(model, pt_state)
    
    return model
