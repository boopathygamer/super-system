"""Tests for model loading and tokenizer."""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """Test configuration loads correctly."""
    from config.settings import model_config, vision_config, brain_config

    assert model_config.dim == 4096
    assert model_config.n_layers == 32
    assert model_config.n_heads == 32
    assert model_config.n_kv_heads == 8
    assert model_config.vocab_size == 32768
    assert brain_config.confidence_threshold == 0.7
    print("✅ Config test passed")


def test_tokenizer():
    """Test tokenizer encoding/decoding."""
    from core.tokenizer import MistralTokenizer

    tokenizer = MistralTokenizer()

    # Test basic encode/decode
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert "Hello" in decoded
    print(f"✅ Tokenizer: '{text}' → {len(tokens)} tokens → '{decoded}'")

    # Test chat formatting
    messages = [
        {"role": "user", "content": "What is AI?"},
    ]
    formatted = tokenizer.format_chat(messages)
    assert "[INST]" in formatted
    assert "What is AI?" in formatted
    print(f"✅ Chat format: {formatted[:80]}...")


def test_model_architecture():
    """Test model can be instantiated (without loading weights)."""
    from core.model_loader import MistralModel
    from config.settings import ModelConfig
    import torch

    # Use a tiny config for testing
    tiny_config = ModelConfig()
    tiny_config.n_layers = 1
    tiny_config.vocab_size = 100

    model = MistralModel(tiny_config)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model architecture: {params / 1e6:.1f}M params (1-layer test)")

    # Test forward pass with random input
    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        logits, kv_caches = model(x)
    assert logits.shape == (1, 10, 100)
    assert len(kv_caches) == 1
    print(f"✅ Forward pass: input {x.shape} → logits {logits.shape}")


if __name__ == "__main__":
    test_config()
    test_tokenizer()
    test_model_architecture()
    print("\n✅ All model tests passed!")
