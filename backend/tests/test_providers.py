"""
Universal API Provider — Test Suite
───────────────────────────────────
Tests for the simplified Universal LLM provider registry.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from core.model_providers import (
    ProviderRegistry,
    UniversalProvider,
    GenerationResult,
    ProviderStatus,
    ProviderType
)
from config.settings import provider_config


# ══════════════════════════════════════════════
# Test 1: Provider Registry Creation
# ══════════════════════════════════════════════
def test_provider_registry():
    registry = ProviderRegistry()
    assert registry.active is None
    assert registry.active_name == "none"
    assert len(registry.list_providers()) == 0


# ══════════════════════════════════════════════
# Test 2: Universal Provider Types & Status
# ══════════════════════════════════════════════
def test_universal_provider_init():
    provider = UniversalProvider(api_key="test-key", base_url="http://localhost", model="gpt-4o")
    assert provider.name == "universal"
    assert provider.model == "gpt-4o"
    assert provider._api_key == "test-key"
    assert provider._base_url == "http://localhost"

    stats = provider.get_stats()
    assert stats["calls"] == 0
    assert stats["errors"] == 0


# ══════════════════════════════════════════════
# Test 3: Auto-Detection
# ══════════════════════════════════════════════
def test_auto_detect_with_keys(monkeypatch):
    # Set config values natively
    monkeypatch.setattr(provider_config, "api_key", "sk-test-123")
    monkeypatch.setattr(provider_config, "base_url", "http://test")
    monkeypatch.setattr(provider_config, "model", "test-model")

    registry = ProviderRegistry.auto_detect()
    
    # Notice it defaults to universal if configured
    assert registry.active_name == "universal"
    assert registry.active.model == "test-model"


# ══════════════════════════════════════════════
# Test 4: Generate Function Bridge
# ══════════════════════════════════════════════
def test_generate_fn_bridge():
    registry = ProviderRegistry()
    
    class MockProvider(UniversalProvider):
        def __init__(self):
            super(UniversalProvider, self).__init__("universal", "mock")
            self.name = "mock"
            self.model = "mock-model"
        def generate(self, prompt, max_tokens=2048, temperature=0.7,
                     system_prompt="", **kwargs):
            if "fail" in prompt:
                return GenerationResult(error="Simulated failure", provider="mock")
            return GenerationResult(text=f"Echo: {prompt}", provider="mock")

    registry.register(MockProvider())
    registry.set_active("mock")

    # The generate_fn wrapper should return just the string
    generate_fn = registry.generate_fn()
    
    # Success case
    ans = generate_fn("hello")
    assert ans == "Echo: hello"
    
    # Error case
    ans = generate_fn("please fail")
    assert "[Error: Simulated failure]" in ans


# ══════════════════════════════════════════════
# Test 5: Fallback Chain (Simplified)
# ══════════════════════════════════════════════
def test_fallback_chain_not_available():
    registry = ProviderRegistry()
    
    ans = registry.generate("test")
    assert ans.error == "No provider available."
    

# ══════════════════════════════════════════════
# Test 6: Display
# ══════════════════════════════════════════════
def test_status_display():
    registry = ProviderRegistry()
    
    class MockProvider(UniversalProvider):
        def __init__(self):
            super(UniversalProvider, self).__init__("universal", "mock")
            self.name = "universal"
            self.model = "gpt-4"

    registry.register(MockProvider())
    registry.set_active("universal")
    
    display = registry.status_display()
    assert "╔══ Model Providers ══╗" in display
    assert "universal" in display
    assert "gpt-4" in display
    assert "Active: universal" in display

if __name__ == "__main__":
    pytest.main(["-v", __file__])
