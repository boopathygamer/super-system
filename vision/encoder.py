"""
Vision Encoder
──────────────
CLIP ViT-L/14-336px encoder for extracting image features.
Converts images into patch embeddings that can be projected into the LLM space.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image

from config.settings import vision_config

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """
    CLIP Vision Encoder wrapper.

    Encodes images into dense patch embeddings using CLIP ViT-L/14-336px.
    Output shape: [batch, n_patches, vision_dim] = [B, 576, 768]
    """

    def __init__(self, config=None):
        super().__init__()
        cfg = config or vision_config
        self.config = cfg
        self._model = None
        self._preprocess = None
        self._loaded = False

    def _lazy_load(self):
        """Lazy-load CLIP model on first use to avoid startup overhead."""
        if self._loaded:
            return

        import open_clip

        logger.info(
            f"Loading CLIP model: {self.config.clip_model_name} "
            f"({self.config.clip_pretrained})"
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
        )

        # We only need the visual transformer
        self._model = model.visual
        self._preprocess = preprocess
        self._model.eval()
        self._loaded = True

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info(f"CLIP vision encoder loaded: {param_count / 1e6:.1f}M parameters")

    @property
    def model(self):
        self._lazy_load()
        return self._model

    @property
    def preprocess(self):
        self._lazy_load()
        return self._preprocess

    def to(self, *args, **kwargs):
        """Override to handle lazy loading."""
        self._lazy_load()
        self._model = self._model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    @torch.inference_mode()
    def encode_image(
        self,
        image: Union[Image.Image, str],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode a single image into patch embeddings.

        Args:
            image: PIL Image or path to image file
            device: Target device

        Returns:
            Tensor of shape [1, n_patches, vision_dim] = [1, 576, 768]
        """
        self._lazy_load()

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        # Preprocess
        pixel_values = self.preprocess(image).unsqueeze(0)

        if device:
            pixel_values = pixel_values.to(device)
            self._model = self._model.to(device)

        # Get patch features (before final pooling)
        features = self._encode_patches(pixel_values)

        return features

    @torch.inference_mode()
    def encode_images(
        self,
        images: List[Union[Image.Image, str]],
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode multiple images into patch embeddings.

        Args:
            images: List of PIL Images or paths

        Returns:
            Tensor of shape [batch, n_patches, vision_dim]
        """
        self._lazy_load()

        processed = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            processed.append(self.preprocess(img))

        pixel_values = torch.stack(processed)

        if device:
            pixel_values = pixel_values.to(device)
            self._model = self._model.to(device)

        features = self._encode_patches(pixel_values)
        return features

    def _encode_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from CLIP (before final projection).

        This hooks into the ViT's intermediate output to get per-patch
        features rather than the pooled [CLS] embedding.
        """
        model = self._model

        # CLIP ViT architecture:
        # conv1 -> class_embedding + positional_embedding -> transformer -> ln_post

        # Patch embedding
        x = model.conv1(pixel_values)  # [B, hidden, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, hidden, n_patches]
        x = x.permute(0, 2, 1)  # [B, n_patches, hidden]

        # Add class token and positional embeddings
        cls_token = model.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Handle positional embedding
        if hasattr(model, 'positional_embedding'):
            x = x + model.positional_embedding.to(x.dtype)

        # Pre-transformer norm (if exists)
        if hasattr(model, 'ln_pre'):
            x = model.ln_pre(x)

        # Transformer
        if hasattr(model, 'transformer'):
            x = model.transformer(x)
        elif hasattr(model, 'resblocks'):
            for block in model.resblocks:
                x = block(x)

        # Post-transformer norm
        if hasattr(model, 'ln_post'):
            x = model.ln_post(x)

        # Remove CLS token, keep patch features only
        patch_features = x[:, 1:, :]  # [B, n_patches, hidden_dim]

        return patch_features
