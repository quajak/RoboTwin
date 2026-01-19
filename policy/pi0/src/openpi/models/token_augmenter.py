"""Token augmentation module for pi0 training.

This module provides a wrapper around the DiT model for augmenting SigLIP tokens
during pi0 finetuning, with configurable camera selection and probabilistic application.
"""

import dataclasses
from collections.abc import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
from openpi.models.dit_augmenter import DiT, solve_ode_flow_matching_jax
from openpi.models.dit_weight_loader import load_dit_from_pytorch_checkpoint


@dataclasses.dataclass(frozen=True)
class TokenAugmenterConfig:
    """Configuration for token augmentation.

    Attributes:
        checkpoint_path: Path to trained DiT checkpoint (.pt file)
        augment_cameras: List of camera names to augment (e.g., ["cam_high", "cam_left_wrist"])
        augment_probability: Probability of augmenting each image (0.0 to 1.0)
        num_steps: Number of ODE solver steps for inference
        token_scale: Normalization factor for tokens (default 128.0 matching training)
        input_size: Number of input tokens (sequence length)
        in_channels: Input token dimension (1152 for SigLIP So400m)
        hidden_size: DiT hidden dimension
        depth: Number of DiT transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        num_classes: Number of classes for conditioning
    """
    checkpoint_path: str
    augment_cameras: Sequence[str]
    augment_probability: float = 0.5
    num_steps: int = 32
    token_scale: float = 128.0
    input_size: int = 256
    in_channels: int = 1152
    hidden_size: int = 768
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 10


class TokenAugmenter(nnx.Module):
    """Augments SigLIP tokens using a trained DiT model.

    This module wraps the DiT model and provides methods for augmenting
    image tokens during pi0 training.
    """

    def __init__(self, config: TokenAugmenterConfig, rngs: nnx.Rngs):
        self.config = config
        
        # Load DiT model from checkpoint
        self.dit = load_dit_from_pytorch_checkpoint(
            checkpoint_path=config.checkpoint_path,
            input_size=config.input_size,
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_classes=config.num_classes,
            dropout=0.0,
            ignore_dt=False,
            rngs=rngs,
        )

    @at.typecheck
    def augment_tokens(
        self,
        tokens: at.Float[at.Array, "b l c"],
    ) -> at.Float[at.Array, "b l c"]:
        """Apply augmentation to tokens using flow matching ODE solver.

        Args:
            tokens: Input SigLIP tokens (batch, seq_len, dim)

        Returns:
            Augmented tokens
        """
        # Normalize tokens as done during training
        normalized_tokens = tokens / self.config.token_scale

        # Create model function for ODE solver
        def model_fn(x, t, dt, y):
            return self.dit(x, t, dt, y, deterministic=True)

        # Run ODE solver
        augmented = solve_ode_flow_matching_jax(
            model_fn=model_fn,
            x=normalized_tokens,
            steps=self.config.num_steps,
            class_label=0
        )

        # Denormalize
        return augmented * self.config.token_scale

    @at.typecheck
    def maybe_augment(
        self,
        tokens: at.Float[at.Array, "b l c"],
        camera_name: str,
        rng: at.KeyArrayLike
    ) -> at.Float[at.Array, "b l c"]:
        """Conditionally augment tokens based on camera name and probability.

        Args:
            tokens: Input SigLIP tokens
            camera_name: Name of the camera (e.g., "cam_high")
            rng: Random key for probabilistic augmentation

        Returns:
            Original or augmented tokens based on probability
        """
        # Check if this camera should be augmented
        should_consider = camera_name in self.config.augment_cameras

        if not should_consider:
            return tokens

        # Probabilistic augmentation
        should_augment = jax.random.uniform(rng) < self.config.augment_probability

        return jax.lax.cond(
            should_augment,
            lambda: self.augment_tokens(tokens),
            lambda: tokens
        )


def create_token_augmenter(config: TokenAugmenterConfig, rngs: nnx.Rngs | None = None) -> TokenAugmenter:
    """Create and initialize a TokenAugmenter with weights from checkpoint.

    Args:
        config: TokenAugmenterConfig with model parameters and checkpoint path
        rngs: NNX random number generators (created if None)

    Returns:
        TokenAugmenter NNX module with loaded weights
    """
    if rngs is None:
        rngs = nnx.Rngs(jax.random.key(0))
    
    return TokenAugmenter(config=config, rngs=rngs)


def load_token_augmenter_from_config(config: TokenAugmenterConfig) -> TokenAugmenter:
    """Convenience function to load a token augmenter from config.

    Args:
        config: TokenAugmenterConfig with all necessary parameters

    Returns:
        TokenAugmenter NNX module
    """
    return create_token_augmenter(config)
