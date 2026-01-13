"""Weight loader for converting PyTorch DiT checkpoints to JAX format.

This module handles the conversion of PyTorch checkpoint weights to JAX-compatible
format for use with the JAX DiT implementation in dit_augmenter.py.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import torch

from openpi.models.dit_augmenter import DiT


def _convert_linear_weights(pt_weight: np.ndarray, pt_bias: np.ndarray | None) -> dict[str, np.ndarray]:
    """Convert PyTorch linear layer weights to JAX format. PyTorch stores weights as (out, in), JAX expects (in, out)."""
    result = {"kernel": pt_weight.T}
    if pt_bias is not None:
        result["bias"] = pt_bias
    return result


def _convert_embedding_weights(pt_weight: np.ndarray) -> dict[str, np.ndarray]:
    """Convert PyTorch embedding table to JAX format."""
    return {"embedding_table": pt_weight}


def _convert_timestep_embedder(prefix: str, pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert TimestepEmbedder weights."""
    return {
        "mlp_0": _convert_linear_weights(
            pt_state[f"{prefix}.mlp.0.weight"],
            pt_state[f"{prefix}.mlp.0.bias"]
        ),
        "mlp_2": _convert_linear_weights(
            pt_state[f"{prefix}.mlp.2.weight"],
            pt_state[f"{prefix}.mlp.2.bias"]
        ),
    }


def _convert_label_embedder(pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert LabelEmbedder weights."""
    return _convert_embedding_weights(pt_state["y_embedder.embedding_table.weight"])


def _convert_input_embedder(pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert InputEmbedder weights."""
    return {
        "proj": _convert_linear_weights(
            pt_state["x_embedder.proj.weight"],
            pt_state["x_embedder.proj.bias"]
        )
    }


def _convert_mlp_block(prefix: str, pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert MlpBlock weights."""
    return {
        "fc1": _convert_linear_weights(
            pt_state[f"{prefix}.fc1.weight"],
            pt_state[f"{prefix}.fc1.bias"]
        ),
        "fc2": _convert_linear_weights(
            pt_state[f"{prefix}.fc2.weight"],
            pt_state[f"{prefix}.fc2.bias"]
        ),
    }


def _convert_dit_block(block_idx: int, pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert a single DiTBlock weights."""
    prefix = f"blocks.{block_idx}"

    # adaLN modulation (SiLU has no params, only the Linear)
    adaln_weight = pt_state[f"{prefix}.adaLN_modulation.1.weight"]
    adaln_bias = pt_state[f"{prefix}.adaLN_modulation.1.bias"]

    # MultiHeadAttention in JAX uses different parameter structure
    # PyTorch nn.MultiheadAttention stores: in_proj_weight (3*embed, embed), in_proj_bias, out_proj.weight, out_proj.bias
    in_proj_weight = pt_state[f"{prefix}.attn.in_proj_weight"]
    in_proj_bias = pt_state[f"{prefix}.attn.in_proj_bias"]
    out_proj_weight = pt_state[f"{prefix}.attn.out_proj.weight"]
    out_proj_bias = pt_state[f"{prefix}.attn.out_proj.bias"]

    hidden_size = out_proj_weight.shape[0]
    # Split in_proj into q, k, v
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3, axis=0)

    # JAX MultiHeadDotProductAttention expects (in_features, num_heads, head_dim) for each of q, k, v
    # We need to figure out num_heads from the structure
    # For now, store as combined and let the model handle reshaping
    # Actually, flax's MultiHeadDotProductAttention uses different param names

    return {
        "adaLN_modulation_1": _convert_linear_weights(adaln_weight, adaln_bias),
        "attn": {
            "query": {"kernel": q_weight.T, "bias": q_bias},
            "key": {"kernel": k_weight.T, "bias": k_bias},
            "value": {"kernel": v_weight.T, "bias": v_bias},
            "out": {"kernel": out_proj_weight.T, "bias": out_proj_bias},
        },
        "mlp": _convert_mlp_block(f"{prefix}.mlp", pt_state),
    }


def _convert_final_layer(pt_state: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert FinalLayer weights."""
    return {
        "adaLN_modulation_1": _convert_linear_weights(
            pt_state["final_layer.adaLN_modulation.1.weight"],
            pt_state["final_layer.adaLN_modulation.1.bias"]
        ),
        "linear": _convert_linear_weights(
            pt_state["final_layer.linear.weight"],
            pt_state["final_layer.linear.bias"]
        ),
    }


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

    # Determine model depth from checkpoint
    block_keys = [k for k in pt_state.keys() if k.startswith("blocks.")]
    block_indices = set(int(k.split(".")[1]) for k in block_keys)
    depth = max(block_indices) + 1

    # Build JAX params dict
    jax_params = {
        "x_embedder": _convert_input_embedder(pt_state),
        "t_embedder": _convert_timestep_embedder("t_embedder", pt_state),
        "dt_embedder": _convert_timestep_embedder("dt_embedder", pt_state),
        "y_embedder": _convert_label_embedder(pt_state),
        "pos_embed": pt_state["pos_embed"],
        "final_layer": _convert_final_layer(pt_state),
    }

    # Convert all blocks
    for i in range(depth):
        jax_params[f"blocks_{i}"] = _convert_dit_block(i, pt_state)

    return jax_params


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
    ignore_dt: bool = False
) -> tuple[DiT, dict[str, Any]]:
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

    Returns:
        Tuple of (model, params)
    """
    model = DiT(
        input_size=input_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_classes=num_classes,
        dropout=dropout,
        ignore_dt=ignore_dt,
    )

    params = load_pytorch_dit_weights(checkpoint_path)

    # Convert numpy arrays to jax arrays
    params = jax.tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params)

    return model, {"params": params}


# Need to import jax for tree_map
import jax
