"""
Multi-Model Provider System — Test Suite
─────────────────────────────────────────
Tests for the provider registry, auto-detection, fallback chain,
and individual provider wrappers.
"""

import sys
import os
import types

# ── Setup namespace packages to bypass agents/__init__.py ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create lightweight namespace stubs
for pkg in ['agents', 'agents.tools', 'agents.experts']:
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *pkg.split('.'))]
        mod.__package__ = pkg
        sys.modules[pkg] = mod


# ══════════════════════════════════════════════
# Test 1: Provider Registry Creation
# ══════════════════════════════════════════════

def test_provider_registry():
    from core.model_providers import ProviderRegistry, MistralLocalProvider

    registry = ProviderRegistry()

    # Register a local provider
    local = MistralLocalProvider()
    registry.register(local)
    registry.set_active("local")

    assert registry.active is not None
    assert registry.active_name == "local"
    assert len(registry.list_providers()) == 1

    # Check provider info
    info = registry.list_providers()[0]
    assert info["name"] == "local"
    assert info["active"] is True

    print("✅ test_provider_registry PASSED — registry works correctly")


# ══════════════════════════════════════════════
# Test 2: Provider Types & Status
# ══════════════════════════════════════════════

def test_provider_types():
    from core.model_providers import (
        ProviderType, ProviderStatus, GenerationResult,
    )

    # Test ProviderType enum
    assert ProviderType.GEMINI.value == "gemini"
    assert ProviderType.CLAUDE.value == "claude"
    assert ProviderType.CHATGPT.value == "chatgpt"
    assert ProviderType.LOCAL.value == "local"
    assert ProviderType.AUTO.value == "auto"

    # Test ProviderStatus
    status = ProviderStatus(name="test", available=True, model="test-v1")
    assert status.available is True

    # Test GenerationResult
    result = GenerationResult(text="Hello", provider="test")
    assert result.ok is True

    error_result = GenerationResult(error="Something failed")
    assert error_result.ok is False

    print("✅ test_provider_types PASSED — all data models work")


# ══════════════════════════════════════════════
# Test 3: Auto-Detection (No Keys)
# ══════════════════════════════════════════════

def test_auto_detect_no_keys():
    from core.model_providers import ProviderRegistry

    # Clear any existing env vars for clean test
    old_gemini = os.environ.pop("GEMINI_API_KEY", None)
    old_claude = os.environ.pop("CLAUDE_API_KEY", None)
    old_openai = os.environ.pop("OPENAI_API_KEY", None)
    old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)

    try:
        registry = ProviderRegistry.auto_detect(
            preferred="auto",
            gemini_key="",
            claude_key="",
            openai_key="",
        )

        # Should only have local provider
        assert registry.active_name == "local"
        providers = registry.list_providers()
        assert len(providers) == 1
        assert providers[0]["name"] == "local"
    finally:
        # Restore env vars
        if old_gemini:
            os.environ["GEMINI_API_KEY"] = old_gemini
        if old_claude:
            os.environ["CLAUDE_API_KEY"] = old_claude
        if old_openai:
            os.environ["OPENAI_API_KEY"] = old_openai
        if old_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    print("✅ test_auto_detect_no_keys PASSED — falls back to local")


# ══════════════════════════════════════════════
# Test 4: Auto-Detection (With Keys)
# ══════════════════════════════════════════════

def test_auto_detect_with_keys():
    from core.model_providers import ProviderRegistry

    # Test with a fake Gemini key (won't actually connect)
    registry = ProviderRegistry.auto_detect(
        preferred="auto",
        gemini_key="fake-gemini-key-for-testing",
        claude_key="",
        openai_key="",
    )

    # Should have Gemini + local
    providers = registry.list_providers()
    names = [p["name"] for p in providers]
    assert "gemini" in names
    assert "local" in names
    assert registry.active_name == "gemini"  # Gemini is first API provider

    print("✅ test_auto_detect_with_keys PASSED — detects Gemini + local fallback")


# ══════════════════════════════════════════════
# Test 5: Preferred Provider Override
# ══════════════════════════════════════════════

def test_preferred_provider():
    from core.model_providers import ProviderRegistry

    registry = ProviderRegistry.auto_detect(
        preferred="local",
        gemini_key="fake-key",
        claude_key="",
        openai_key="",
    )

    # Even though Gemini key exists, local was preferred
    assert registry.active_name == "local"

    print("✅ test_preferred_provider PASSED — respects preferred override")


# ══════════════════════════════════════════════
# Test 6: Generate Function Bridge
# ══════════════════════════════════════════════

def test_generate_fn_bridge():
    from core.model_providers import ProviderRegistry, ModelProvider, GenerationResult

    # Create a mock provider for testing
    class MockProvider(ModelProvider):
        def generate(self, prompt, max_tokens=2048, temperature=0.7,
                     system_prompt="", **kwargs):
            return GenerationResult(
                text=f"Mock response to: {prompt}",
                provider="mock",
                model="mock-v1",
            )

    registry = ProviderRegistry()
    mock = MockProvider(name="mock", model="mock-v1")
    registry.register(mock)
    registry.set_active("mock")

    # Get the generate_fn bridge
    gen_fn = registry.generate_fn()

    # This should work like the old lambda prompt: engine.generate(prompt)
    result = gen_fn("Hello world")
    assert "Mock response to: Hello world" in result
    assert isinstance(result, str)

    print("✅ test_generate_fn_bridge PASSED — generate_fn() bridge works")


# ══════════════════════════════════════════════
# Test 7: Fallback Chain
# ══════════════════════════════════════════════

def test_fallback_chain():
    from core.model_providers import ProviderRegistry, ModelProvider, GenerationResult

    # Provider that always fails
    class FailProvider(ModelProvider):
        def generate(self, prompt, **kwargs):
            return GenerationResult(error="I always fail", provider="fail")

    # Provider that always works
    class GoodProvider(ModelProvider):
        def generate(self, prompt, **kwargs):
            return GenerationResult(
                text="Fallback succeeded!",
                provider="good",
            )

    registry = ProviderRegistry()
    registry.register(FailProvider(name="fail", model="fail-v1"))
    registry.register(GoodProvider(name="good", model="good-v1"))
    registry.set_active("fail")

    # Active provider fails → should fallback to good
    result = registry.generate("test")
    assert result.ok
    assert result.text == "Fallback succeeded!"

    print("✅ test_fallback_chain PASSED — fallback works when primary fails")


# ══════════════════════════════════════════════
# Test 8: Provider Switching
# ══════════════════════════════════════════════

def test_provider_switching():
    from core.model_providers import ProviderRegistry, ModelProvider, GenerationResult

    class ProviderA(ModelProvider):
        def generate(self, prompt, **kwargs):
            return GenerationResult(text="I am A", provider="a")

    class ProviderB(ModelProvider):
        def generate(self, prompt, **kwargs):
            return GenerationResult(text="I am B", provider="b")

    registry = ProviderRegistry()
    registry.register(ProviderA(name="a", model="a-v1"))
    registry.register(ProviderB(name="b", model="b-v1"))
    registry.set_active("a")

    # Start with A
    result = registry.generate("test")
    assert result.text == "I am A"

    # Switch to B
    registry.set_active("b")
    result = registry.generate("test")
    assert result.text == "I am B"

    # Switch back to A
    registry.set_active("a")
    result = registry.generate("test")
    assert result.text == "I am A"

    print("✅ test_provider_switching PASSED — hot-switching works")


# ══════════════════════════════════════════════
# Test 9: Provider Stats Tracking
# ══════════════════════════════════════════════

def test_provider_stats():
    from core.model_providers import ModelProvider, GenerationResult

    class TrackedProvider(ModelProvider):
        def generate(self, prompt, **kwargs):
            import time
            start = time.time()
            result = GenerationResult(text="tracked", provider="tracked")
            latency = (time.time() - start) * 1000
            self._track(latency)
            return result

    p = TrackedProvider(name="tracked", model="tracked-v1")

    # Make 3 calls
    p.generate("a")
    p.generate("b")
    p.generate("c")

    stats = p.get_stats()
    assert stats["calls"] == 3
    assert stats["errors"] == 0
    assert stats["provider"] == "tracked"

    print("✅ test_provider_stats PASSED — metrics tracking works")


# ══════════════════════════════════════════════
# Test 10: ProviderConfig
# ══════════════════════════════════════════════

def test_provider_config():
    from config.settings import ProviderConfig

    config = ProviderConfig()

    # Default provider should be "auto"
    assert config.provider == "auto" or config.provider in (
        "gemini", "claude", "chatgpt", "local"
    )

    # available_providers should always include "local"
    assert "local" in config.available_providers

    # Default models should be set
    assert config.gemini_model
    assert config.claude_model
    assert config.openai_model

    print("✅ test_provider_config PASSED — ProviderConfig works correctly")


# ══════════════════════════════════════════════
# Test 11: Status Display
# ══════════════════════════════════════════════

def test_status_display():
    from core.model_providers import ProviderRegistry, MistralLocalProvider

    registry = ProviderRegistry()
    registry.register(MistralLocalProvider())
    registry.set_active("local")

    display = registry.status_display()
    assert "local" in display
    assert "Mistral" in display
    assert "Active" in display

    print("✅ test_status_display PASSED — status display renders correctly")


# ══════════════════════════════════════════════
# Run All Tests
# ══════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_provider_registry,
        test_provider_types,
        test_auto_detect_no_keys,
        test_auto_detect_with_keys,
        test_preferred_provider,
        test_generate_fn_bridge,
        test_fallback_chain,
        test_provider_switching,
        test_provider_stats,
        test_provider_config,
        test_status_display,
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("  🧪 Multi-Model Provider Tests")
    print("=" * 60, "\n")

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_fn.__name__} FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("  🎉 ALL TESTS PASSED!")
    print(f"{'=' * 60}")
