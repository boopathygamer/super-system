"""Tests for the vision pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_projector():
    """Test vision projector dimensions."""
    import torch
    from vision.projector import VisionProjector

    proj = VisionProjector()

    # Simulate CLIP output: [batch=1, patches=576, dim=768]
    fake_features = torch.randn(1, 576, 768)
    output = proj(fake_features)

    assert output.shape == (1, 576, 4096)
    print(f"✅ Projector: {fake_features.shape} → {output.shape}")


def test_vision_config():
    """Test vision configuration."""
    from config.settings import vision_config

    assert vision_config.vision_dim == 768
    assert vision_config.projection_dim == 4096
    assert vision_config.num_image_tokens == 576
    print(f"✅ Vision config: {vision_config.clip_model_name}")


if __name__ == "__main__":
    test_vision_config()
    test_projector()
    print("\n✅ All vision tests passed!")
