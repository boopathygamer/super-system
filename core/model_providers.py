"""
Multi-Model Provider System — Universal LLM Backend.
─────────────────────────────────────────────────────
Supports 4 providers:
  1. Gemini (Google)   — google-generativeai SDK
  2. Claude (Anthropic) — anthropic SDK
  3. ChatGPT (OpenAI)  — openai SDK
  4. Local Mistral 7B  — built-in InferenceEngine

Features:
  - Unified generate() / stream() interface
  - Auto-detection from environment API keys
  - Fallback chain: primary → secondary → local
  - Health checks + error handling
  - Provider registry with hot-switching
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

class ProviderType(Enum):
    """Supported LLM provider types."""
    GEMINI = "gemini"
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    LOCAL = "local"
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
# 1. Gemini Provider (Google)
# ──────────────────────────────────────────────

class GeminiProvider(ModelProvider):
    """Google Gemini API provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        super().__init__(name="gemini", model=model)
        self._api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self.model)
            logger.info(f"✅ Gemini provider initialized — model={self.model}")
        except ImportError:
            logger.error(
                "❌ google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )
            self._client = None
        except Exception as e:
            logger.error(f"❌ Gemini init failed: {e}")
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
            return GenerationResult(error="Gemini client not initialized")

        start = time.time()
        try:
            import google.generativeai as genai

            config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self._client.generate_content(
                full_prompt,
                generation_config=config,
            )

            latency = (time.time() - start) * 1000
            text = response.text if response.text else ""
            tokens = getattr(response, "usage_metadata", None)
            token_count = 0
            if tokens:
                token_count = getattr(tokens, "total_token_count", 0)

            self._track(latency)
            return GenerationResult(
                text=text,
                provider="gemini",
                model=self.model,
                tokens_used=token_count,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track(latency, error=True)
            logger.error(f"Gemini generate error: {e}")
            return GenerationResult(error=str(e), provider="gemini")

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
            import google.generativeai as genai

            config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )

            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self._client.generate_content(
                full_prompt,
                generation_config=config,
                stream=True,
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini stream error: {e}")


# ──────────────────────────────────────────────
# 2. Claude Provider (Anthropic)
# ──────────────────────────────────────────────

class ClaudeProvider(ModelProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        super().__init__(name="claude", model=model)
        self._api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info(f"✅ Claude provider initialized — model={self.model}")
        except ImportError:
            logger.error(
                "❌ anthropic not installed. Run: pip install anthropic"
            )
            self._client = None
        except Exception as e:
            logger.error(f"❌ Claude init failed: {e}")
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
            return GenerationResult(error="Claude client not initialized")

        start = time.time()
        try:
            messages = [{"role": "user", "content": prompt}]

            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )

            latency = (time.time() - start) * 1000
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text

            token_count = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            )

            self._track(latency)
            return GenerationResult(
                text=text,
                provider="claude",
                model=self.model,
                tokens_used=token_count,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track(latency, error=True)
            logger.error(f"Claude generate error: {e}")
            return GenerationResult(error=str(e), provider="claude")

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
            with self._client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Claude stream error: {e}")


# ──────────────────────────────────────────────
# 3. ChatGPT Provider (OpenAI)
# ──────────────────────────────────────────────

class ChatGPTProvider(ModelProvider):
    """OpenAI ChatGPT API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(name="chatgpt", model=model)
        self._api_key = api_key
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import openai
            self._client = openai.OpenAI(api_key=self._api_key)
            logger.info(f"✅ ChatGPT provider initialized — model={self.model}")
        except ImportError:
            logger.error(
                "❌ openai not installed. Run: pip install openai"
            )
            self._client = None
        except Exception as e:
            logger.error(f"❌ ChatGPT init failed: {e}")
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
            return GenerationResult(error="ChatGPT client not initialized")

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
                provider="chatgpt",
                model=self.model,
                tokens_used=token_count,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track(latency, error=True)
            logger.error(f"ChatGPT generate error: {e}")
            return GenerationResult(error=str(e), provider="chatgpt")

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
            logger.error(f"ChatGPT stream error: {e}")


# ──────────────────────────────────────────────
# 4. Local Mistral Provider (Built-in)
# ──────────────────────────────────────────────

class MistralLocalProvider(ModelProvider):
    """
    Local Mistral 7B provider — wraps the existing InferenceEngine.
    This is the fallback provider that always works offline.
    """

    def __init__(self, engine=None):
        super().__init__(name="local", model="Mistral-7B-Instruct-v0.3")
        self._engine = engine
        self._loaded = engine is not None

    def load_model(self):
        """Lazy-load the local Mistral model."""
        if self._loaded:
            return

        try:
            from core.model_loader import load_model
            from core.tokenizer import MistralTokenizer
            from core.inference import InferenceEngine

            logger.info("🧠 Loading local Mistral 7B model...")
            model = load_model()
            tokenizer = MistralTokenizer()
            self._engine = InferenceEngine(model, tokenizer)
            self._loaded = True
            logger.info("✅ Local Mistral provider initialized")
        except Exception as e:
            logger.error(f"❌ Local model loading failed: {e}")
            self._loaded = False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> GenerationResult:
        if not self._loaded:
            self.load_model()

        if not self._engine:
            return GenerationResult(error="Local model not loaded")

        start = time.time()
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"

            text = self._engine.generate(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

            latency = (time.time() - start) * 1000
            self._track(latency)
            return GenerationResult(
                text=text,
                provider="local",
                model=self.model,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._track(latency, error=True)
            logger.error(f"Local generate error: {e}")
            return GenerationResult(error=str(e), provider="local")

    def stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        if not self._loaded:
            self.load_model()

        if not self._engine:
            return

        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"[INST] {system_prompt}\n\n{prompt} [/INST]"

            for chunk in self._engine.stream_generate(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Local stream error: {e}")


# ──────────────────────────────────────────────
# Provider Registry — Auto-Detection + Fallback
# ──────────────────────────────────────────────

class ProviderRegistry:
    """
    Central registry for all model providers.

    Features:
      - Auto-detect available providers from env vars / API keys
      - Fallback chain: primary → next available → local Mistral
      - Hot-switch between providers at runtime
      - Provider health monitoring
    """

    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._active: Optional[str] = None
        self._fallback_order: List[str] = []
        logger.info("ProviderRegistry initialized")

    # ── Registration ──

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

    # ── Access ──

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

    # ── Generation with Fallback ──

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = "",
        **kwargs,
    ) -> GenerationResult:
        """
        Generate using active provider with automatic fallback.

        Tries: active → fallback chain → local Mistral
        """
        # Try active provider first
        if self.active:
            result = self.active.generate(
                prompt, max_tokens, temperature, system_prompt, **kwargs
            )
            if result.ok:
                return result
            logger.warning(
                f"Active provider '{self._active}' failed: {result.error}. "
                f"Trying fallback..."
            )

        # Fallback chain
        for name in self._fallback_order:
            if name == self._active:
                continue  # Already tried
            provider = self._providers.get(name)
            if provider:
                result = provider.generate(
                    prompt, max_tokens, temperature, system_prompt, **kwargs
                )
                if result.ok:
                    logger.info(f"Fallback to '{name}' succeeded")
                    return result

        return GenerationResult(
            error="All providers failed. No model available.",
        )

    def generate_fn(self) -> Callable[[str], str]:
        """
        Return a simple generate function compatible with the existing system.

        This is the bridge between the new provider system and the existing
        `generate_fn = lambda prompt: engine.generate(prompt)` pattern.
        """
        def _generate(prompt: str, **kwargs) -> str:
            result = self.generate(prompt, **kwargs)
            return result.text if result.ok else f"[Error: {result.error}]"
        return _generate

    # ── Auto-Detection ──

    @classmethod
    def auto_detect(
        cls,
        preferred: str = "auto",
        gemini_key: str = None,
        claude_key: str = None,
        openai_key: str = None,
    ) -> "ProviderRegistry":
        """
        Auto-detect available providers from API keys and environment.

        Priority:
          1. If preferred is specified (not "auto"), use that
          2. Otherwise, first available API key wins
          3. Local Mistral is always registered as fallback

        Args:
            preferred: "auto", "gemini", "claude", "chatgpt", or "local"
            gemini_key: Gemini API key (or from GEMINI_API_KEY env)
            claude_key: Claude API key (or from CLAUDE_API_KEY env)
            openai_key: OpenAI API key (or from OPENAI_API_KEY env)

        Returns:
            Configured ProviderRegistry
        """
        registry = cls()

        # Resolve API keys from args or environment
        gkey = gemini_key or os.getenv("GEMINI_API_KEY", "")
        ckey = claude_key or os.getenv("CLAUDE_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
        okey = openai_key or os.getenv("OPENAI_API_KEY", "")

        # Try loading .env file if available
        try:
            from dotenv import load_dotenv
            load_dotenv()
            gkey = gkey or os.getenv("GEMINI_API_KEY", "")
            ckey = ckey or os.getenv("CLAUDE_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
            okey = okey or os.getenv("OPENAI_API_KEY", "")
        except ImportError:
            pass

        # Determine preferred provider from env if auto
        if preferred == "auto":
            preferred = os.getenv("LLM_PROVIDER", "auto")

        detected = []

        # Register API providers
        if gkey:
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            registry.register(GeminiProvider(api_key=gkey, model=gemini_model))
            detected.append("gemini")

        if ckey:
            claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
            registry.register(ClaudeProvider(api_key=ckey, model=claude_model))
            detected.append("claude")

        if okey:
            openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
            registry.register(ChatGPTProvider(api_key=okey, model=openai_model))
            detected.append("chatgpt")

        # Always register local Mistral as fallback
        registry.register(MistralLocalProvider())
        detected.append("local")

        # Set active provider
        if preferred != "auto" and preferred in registry._providers:
            registry.set_active(preferred)
        elif detected:
            # First API provider if available, otherwise local
            api_providers = [d for d in detected if d != "local"]
            registry.set_active(api_providers[0] if api_providers else "local")

        logger.info(
            f"🔍 Auto-detected providers: {detected}. "
            f"Active: {registry.active_name}"
        )

        return registry

    # ── Display ──

    def status_display(self) -> str:
        """Formatted status for display in CLI/logs."""
        lines = ["╔══ Model Providers ══╗"]
        for p in self._providers.values():
            icon = "🟢" if p.name == self._active else "⚪"
            lines.append(f"  {icon} {p.name:<10} → {p.model}")
        lines.append(f"  Active: {self.active_name}")
        lines.append("╚═════════════════════╝")
        return "\n".join(lines)
