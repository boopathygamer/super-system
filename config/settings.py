"""
Global configuration for the Custom LLM System.
All paths, hyperparameters, and thresholds live here.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Base Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(os.getenv("LLM_BASE_DIR", r"C:\llm"))
MODEL_DIR = BASE_DIR / "mistral-7B-Instruct-v0.3"
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory_store"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure data directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Mistral 7B model configuration."""
    model_path: Path = MODEL_DIR
    safetensors_file: str = "consolidated.safetensors"
    tokenizer_file: str = "tokenizer.model.v3"

    # Architecture (from params.json)
    dim: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    norm_eps: float = 1e-5
    vocab_size: int = 32768
    rope_theta: float = 1000000.0

    # Quantization
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


@dataclass
class GenerationConfig:
    """Text generation hyperparameters."""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: list = field(default_factory=lambda: ["</s>", "[/INST]"])


@dataclass
class VisionConfig:
    """Vision encoder configuration."""
    clip_model_name: str = "ViT-L-14-336"
    clip_pretrained: str = "openai"
    image_size: int = 336
    vision_dim: int = 768
    projection_dim: int = 4096  # Must match model dim
    num_image_tokens: int = 576  # 336/14 = 24, 24*24 = 576 patches


@dataclass
class BrainConfig:
    """Self-thinking brain configuration."""
    # Memory / Bug Diary
    memory_collection_name: str = "failure_memory"
    memory_persist_dir: str = str(MEMORY_DIR)
    max_memory_retrieval: int = 5
    decay_factor: float = 0.9  # γ for exponential decay

    # Multi-Hypothesis
    max_hypotheses: int = 5
    hypothesis_temperature: float = 1.0  # β for weight updates
    min_hypothesis_weight: float = 0.01

    # Verifier
    confidence_threshold: float = 0.7  # τ — minimum confidence to execute
    risk_threshold: float = 0.3  # κ — maximum risk to execute
    sandbox_risk_threshold: float = 0.6  # κ' — max risk for sandbox
    sandbox_confidence_threshold: float = 0.4  # τ' — min confidence for sandbox

    # Tri-Shield weights (λ₁, λ₂, λ₃, λ₄)
    lambda_robust: float = 1.0
    lambda_detect: float = 0.8
    lambda_contain: float = 0.5
    lambda_complexity: float = 0.2

    # Thinking loop
    max_iterations: int = 5
    improvement_threshold: float = 0.05  # Minimum confidence gain to continue


@dataclass
class AgentConfig:
    """Agent framework configuration."""
    max_tool_calls: int = 10
    sandbox_timeout: int = 30  # seconds
    max_retries: int = 3

    # Tool Policy Engine
    tool_profile: str = "assistant"  # minimal | coding | assistant | full
    tool_global_deny: list = field(default_factory=list)

    # Loop Detection Guardrails
    loop_detection_enabled: bool = True
    loop_warning_threshold: int = 5
    loop_critical_threshold: int = 10
    loop_circuit_breaker_threshold: int = 20

    # Session Manager
    sessions_dir: str = str(DATA_DIR / "sessions")
    session_compaction_threshold: int = 100  # messages before compaction

    # Process Manager
    max_background_processes: int = 20
    process_default_timeout: int = 300
    process_yield_ms: int = 10000  # auto-background timeout

    # Workspace Injection
    workspace_dir: str = str(DATA_DIR / "workspace")

    # Skills Registry
    skills_bundled_dir: str = str(DATA_DIR / "skills" / "bundled")
    skills_managed_dir: str = str(DATA_DIR / "skills" / "managed")

    # Streaming
    stream_chunk_size: int = 50
    stream_coalesce_ms: int = 100
    stream_break_on: str = "sentence"  # token | sentence | paragraph

    # Model Failover
    model_failover_enabled: bool = True
    model_circuit_breaker_threshold: int = 5
    model_circuit_breaker_reset_seconds: int = 60


@dataclass
class ProviderConfig:
    """Multi-model provider configuration."""
    # Active provider: auto | gemini | claude | chatgpt | local
    provider: str = os.getenv("LLM_PROVIDER", "auto")

    # API Keys (from environment variables)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    claude_api_key: str = os.getenv("CLAUDE_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Model names (customizable per provider)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    @property
    def has_any_api_key(self) -> bool:
        """Check if any API key is configured."""
        return bool(self.gemini_api_key or self.claude_api_key or self.openai_api_key)

    @property
    def available_providers(self) -> list:
        """List providers with configured API keys."""
        providers = []
        if self.gemini_api_key:
            providers.append("gemini")
        if self.claude_api_key:
            providers.append("claude")
        if self.openai_api_key:
            providers.append("chatgpt")
        providers.append("local")  # Always available
        return providers


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = os.getenv("LLM_API_HOST", "127.0.0.1")
    port: int = int(os.getenv("LLM_API_PORT", "8000"))
    reload: bool = False
    workers: int = 1


# ──────────────────────────────────────────────
# Global Singleton Configs
# ──────────────────────────────────────────────
model_config = ModelConfig()
generation_config = GenerationConfig()
vision_config = VisionConfig()
brain_config = BrainConfig()
agent_config = AgentConfig()
provider_config = ProviderConfig()
api_config = APIConfig()

