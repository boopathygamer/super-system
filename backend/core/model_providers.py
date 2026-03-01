"""
Multi-Model Provider System — Universal LLM Backend.
─────────────────────────────────────────────────────
Supports any OpenAI-compatible API endpoint.
Features:
  - Universal configuration via LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
  - Streaming support
  - Health checks + error handling
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

class ProviderType(Enum):
    """Supported LLM provider types."""
    UNIVERSAL = "universal"
    AUTO = "auto"


@dataclass
class ProviderStatus:
    """Health/status of a provider."""
    name: str
    available: bool = False
    model: str = ""
    latency_ms: float = 0.0
    error: str = ""
    last_check: float = 0.0


@dataclass
class GenerationResult:
    """Standard result from any provider."""
    text: str = ""
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


# ──────────────────────────────────────────────
# Abstract Base Provider
# ──────────────────────────────────────────────

class ModelProvider(ABC):
    """
    Base class for all LLM providers.
    Every provider must implement generate() and optionally stream().
    """

    def __init__(self, name: str, model: str = ""):
        self.name = name
        self.model = model
        self._call_count = 0
        self._total_latency = 0.0
        self._errors = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> GenerationResult:
        """Generate a response from the model."""
        ...

    def stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream a response token by token. Default: yield full result."""
        result = self.generate(prompt, max_tokens, temperature, system_prompt, **kwargs)
        if result.ok:
            yield result.text

    def health_check(self) -> ProviderStatus:
        """Check if this provider is available."""
        start = time.time()
        try:
            result = self.generate("Say 'ok'", max_tokens=5, temperature=0.0)
            latency = (time.time() - start) * 1000
            return ProviderStatus(
                name=self.name,
                available=result.ok,
                model=self.model,
                latency_ms=latency,
                error=result.error,
                last_check=time.time(),
            )
        except Exception as e:
            return ProviderStatus(
                name=self.name,
                available=False,
                error=str(e),
                last_check=time.time(),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.name,
            "model": self.model,
            "calls": self._call_count,
            "errors": self._errors,
            "avg_latency_ms": (
                self._total_latency / self._call_count
                if self._call_count > 0 else 0
            ),
        }

    def _track(self, latency_ms: float, error: bool = False):
        """Track call metrics."""
        self._call_count += 1
        self._total_latency += latency_ms
        if error:
            self._errors += 1


# ──────────────────────────────────────────────
# 1. Universal Provider
# ──────────────────────────────────────────────

class UniversalProvider(ModelProvider):
    """Universal OpenAI-compatible API provider."""

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(name="universal", model=model)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self._api_key or "sk-no-key",  # some local providers don't need a key
                base_url=self._base_url
            )
            logger.info(f"✅ Universal provider initialized — url={self._base_url} model={self.model}")
        except ImportError:
            logger.error(
                "❌ openai not installed. Run: pip install openai"
            )
            self._client = None
        except Exception as e:
            logger.error(f"❌ Universal init failed: {e}")
            self._client = None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> GenerationResult:
        if not self._client:
            return GenerationResult(error="Universal client not initialized")

        start = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            latency = (time.time() - start) * 1000
            text = response.choices[0].message.content or ""
            token_count = response.usage.total_tokens if response.usage else 0

            self._track(latency)
            return GenerationResult(
                text=text,
                provider="universal",
                model=self.model,
                tokens_used=token_count,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track(latency, error=True)
            logger.error(f"Universal generate error: {e}")
            return GenerationResult(error=str(e), provider="universal")

    def stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        if not self._client:
            return

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Universal stream error: {e}")


# ──────────────────────────────────────────────
# Provider Registry
# ──────────────────────────────────────────────

class ProviderRegistry:
    """
    Central registry for the Universal LLM provider.
    """

    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._active: Optional[str] = None
        self._fallback_order: List[str] = []
        logger.info("ProviderRegistry initialized")

    def register(self, provider: ModelProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        if provider.name not in self._fallback_order:
            self._fallback_order.append(provider.name)
        logger.info(f"Registered provider: {provider.name} ({provider.model})")

    def set_active(self, name: str) -> bool:
        """Set the active provider by name."""
        if name in self._providers:
            self._active = name
            p = self._providers[name]
            logger.info(f"🔄 Active provider → {name} ({p.model})")
            return True
        logger.warning(f"Provider '{name}' not found")
        return False

    @property
    def active(self) -> Optional[ModelProvider]:
        """Get the currently active provider."""
        if self._active and self._active in self._providers:
            return self._providers[self._active]
        return None

    @property
    def active_name(self) -> str:
        """Get the name of the active provider."""
        return self._active or "none"

    def get(self, name: str) -> Optional[ModelProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers with status."""
        return [
            {
                "name": p.name,
                "model": p.model,
                "active": p.name == self._active,
                "calls": p._call_count,
                "errors": p._errors,
            }
            for p in self._providers.values()
        ]

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using active provider.
        """
        if self.active:
            result = self.active.generate(
                prompt, max_tokens, temperature, system_prompt, **kwargs
            )
            if result.ok:
                return result
            logger.warning(
                f"Active provider '{self._active}' failed: {result.error}."
            )
            return result

        return GenerationResult(
            error="No provider available.",
        )

    def generate_fn(self) -> Callable[[str], str]:
        """
        Return a simple generate function compatible with the existing system.
        """
        def _generate(prompt: str, **kwargs) -> str:
            result = self.generate(prompt, **kwargs)
            return result.text if result.ok else f"[Error: {result.error}]"
        return _generate

    @classmethod
    def auto_detect(
        cls,
        preferred: str = "auto",
        # Kept for backwards compatibility but ignored
        gemini_key: str = None,
        claude_key: str = None,
        openai_key: str = None,
    ) -> "ProviderRegistry":
        """
        Initialize the registry from the UniversalAPIConfig.
        """
        registry = cls()

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        from config.settings import provider_config
        
        # Determine credentials
        api_key = provider_config.api_key
        base_url = provider_config.base_url
        model = provider_config.model
        
        if api_key or base_url:
            registry.register(UniversalProvider(api_key=api_key, base_url=base_url, model=model))
            registry.set_active("universal")
        
        logger.info(
            f"🔍 Auto-detected providers: {['universal'] if registry.active else []}. "
            f"Active: {registry.active_name}"
        )

        return registry

    def status_display(self) -> str:
        """Formatted status for display in CLI/logs."""
        lines = ["╔══ Model Providers ══╗"]
        for p in self._providers.values():
            icon = "🟢" if p.name == self._active else "⚪"
            lines.append(f"  {icon} {p.name:<10} → {p.model}")
        lines.append(f"  Active: {self.active_name}")
        lines.append("╚═════════════════════╝")
        return "\n".join(lines)
