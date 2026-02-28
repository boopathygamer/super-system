"""
Global configuration for the Custom LLM System.
All paths, hyperparameters, and thresholds live here.
Loads .env file automatically for secure secret management.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ── Load .env file if present (before any os.getenv calls) ──
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key = _key.strip()
                _val = _val.strip().strip("'\"")
                if _key and _key not in os.environ:  # Don't override real env vars
                    os.environ[_key] = _val


# ──────────────────────────────────────────────
# Base Paths
# ──────────────────────────────────────────────
_DEFAULT_BASE = str(Path(__file__).parent.parent / "data")
BASE_DIR = Path(os.getenv("LLM_BASE_DIR", _DEFAULT_BASE))
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory_store"
UPLOADS_DIR = DATA_DIR / "uploads"

# Ensure data directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)



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
    # Active provider: auto | gemini | claude | chatgpt
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
        return providers


@dataclass
class ThreatScanConfig:
    """Threat Scanner configuration."""
    quarantine_dir: str = str(DATA_DIR / "threat_quarantine")
    max_file_size_mb: int = 100
    entropy_threshold: float = 7.2
    auto_scan_on_file_ops: bool = True


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = os.getenv("LLM_API_HOST", "127.0.0.1")
    port: int = int(os.getenv("LLM_API_PORT", "8000"))
    reload: bool = False
    workers: int = 1


@dataclass
class SSLConfig:
    """HTTPS / TLS configuration."""
    enabled: bool = bool(os.getenv("SSL_ENABLED", ""))
    certfile: str = os.getenv("SSL_CERTFILE", "certs/server.crt")
    keyfile: str = os.getenv("SSL_KEYFILE", "certs/server.key")

    @property
    def is_ready(self) -> bool:
        return self.enabled and Path(self.certfile).exists() and Path(self.keyfile).exists()


@dataclass
class TokenBudgetConfig:
    """Token budget management."""
    daily_limit: int = int(os.getenv("TOKEN_DAILY_LIMIT", "1000000"))
    monthly_limit: int = int(os.getenv("TOKEN_MONTHLY_LIMIT", "30000000"))
    cost_per_1k_premium: float = 0.03   # GPT-4o / Claude 3.5
    cost_per_1k_budget: float = 0.001   # Llama / Mistral
    auto_downgrade: bool = True


# ──────────────────────────────────────────────
# Global Singleton Configs
# ──────────────────────────────────────────────

agent_config = AgentConfig()
provider_config = ProviderConfig()
api_config = APIConfig()
brain_config = BrainConfig()
threat_config = ThreatScanConfig()
ssl_config = SSLConfig()
token_budget_config = TokenBudgetConfig()


