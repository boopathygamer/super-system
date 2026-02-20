"""
Vision Projector
────────────────
MLP that maps CLIP vision features (768-dim) into Mistral's
embedding space (4096-dim), creating "visual tokens" the LLM can attend to.
"""

import logging

import torch
import torch.nn as nn

from config.settings import vision_config

logger = logging.getLogger(__name__)


class VisionProjector(nn.Module):
    """
    2-layer MLP projection: CLIP space → LLM space

    Architecture (LLaVA-style):
        Linear(768, 4096) → GELU → Linear(4096, 4096)

    Input:  [batch, n_patches, 768]
    Output: [batch, n_patches, 4096]  — "visual tokens"
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or vision_config

        self.projector = nn.Sequential(
            nn.Linear(cfg.vision_dim, cfg.projection_dim),
            nn.GELU(),
            nn.Linear(cfg.projection_dim, cfg.projection_dim),
        )

        self._init_weights()

        param_count = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Vision Projector initialized: {cfg.vision_dim} → {cfg.projection_dim}, "
            f"{param_count / 1e6:.1f}M parameters"
        )

    def _init_weights(self):
        """Initialize with small random weights for stable training/adaptation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features into LLM embedding space.

        Args:
            vision_features: [batch, n_patches, vision_dim]

        Returns:
            visual_tokens: [batch, n_patches, projection_dim]
        """
        return self.projector(vision_features)

    def save(self, path: str):
        """Save projector weights."""
        torch.save(self.state_dict(), path)
        logger.info(f"Projector saved to {path}")

    def load(self, path: str, device: str = "cpu"):
        """Load projector weights."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        logger.info(f"Projector loaded from {path}")
