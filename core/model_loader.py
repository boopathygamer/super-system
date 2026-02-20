"""
Mistral 7B Model Loader
───────────────────────
Loads the Mistral 7B Instruct v0.3 model from safetensors with:
- Custom architecture matching params.json (GQA, RoPE, SwiGLU)
- Optional 4-bit quantization via bitsandbytes
- KV-cache for efficient autoregressive generation
"""

import json
import math
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from config.settings import model_config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with configurable theta."""

    def __init__(self, dim: int, max_seq_len: int = 32768, theta: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)
        cos = self.cos_cached[offset: offset + seq_len]
        sin = self.sin_cached[offset: offset + seq_len]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQAAttention(nn.Module):
    """Grouped Query Attention (GQA) — Mistral uses 32 Q heads, 8 KV heads."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # 4 repetitions per KV head

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match Q heads for GQA."""
        if self.n_rep == 1:
            return x
        bs, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv, self.n_rep, seq_len, head_dim)
            .reshape(bs, self.n_heads, seq_len, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(2, 3)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn, v)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output), new_kv_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Single Mistral transformer block."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 hidden_dim: int, norm_eps: float):
        super().__init__()
        self.attention = GQAAttention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward = SwiGLUFFN(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention
        h, new_kv = self.attention(
            self.attention_norm(x), cos, sin, mask, kv_cache
        )
        x = x + h

        # Pre-norm FFN
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv


# ──────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────

class MistralModel(nn.Module):
    """Mistral 7B Instruct v0.3 — Full transformer model."""

    def __init__(self, config=None):
        super().__init__()
        cfg = config or model_config
        self.config = cfg

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=cfg.dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                head_dim=cfg.head_dim,
                hidden_dim=cfg.hidden_dim,
                norm_eps=cfg.norm_eps,
            )
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=cfg.head_dim,
            theta=cfg.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            kv_caches: List of (k, v) tuples per layer
            input_embeds: Optional pre-computed embeddings (for vision tokens)

        Returns:
            logits: [batch, seq_len, vocab_size]
            new_kv_caches: Updated KV caches
        """
        if input_embeds is not None:
            h = input_embeds
        else:
            h = self.tok_embeddings(input_ids)

        bsz, seq_len, _ = h.shape

        # Calculate offset from existing KV cache
        offset = 0
        if kv_caches and kv_caches[0] is not None:
            offset = kv_caches[0][0].shape[2]

        cos, sin = self.rotary_emb(seq_len, offset)
        cos = cos.to(h.device, dtype=h.dtype)
        sin = sin.to(h.device, dtype=h.dtype)

        # Causal mask
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=h.device, dtype=h.dtype)
            mask = torch.triu(mask, diagonal=1)
            if offset > 0:
                # Prepend zeros for cached positions
                mask = torch.cat([
                    torch.zeros(seq_len, offset, device=h.device, dtype=h.dtype),
                    mask,
                ], dim=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total]

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            h, new_kv = layer(h, cos, sin, mask, layer_kv)
            new_kv_caches.append(new_kv)

        h = self.norm(h)
        logits = self.output(h)
        return logits, new_kv_caches


# ──────────────────────────────────────────────
# Weight Loading
# ──────────────────────────────────────────────

# Mapping from safetensors keys to our model keys
WEIGHT_MAP = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.weight",
    "output.weight": "output.weight",
}

# Layer-level mappings
LAYER_WEIGHT_MAP = {
    "attention.wq.weight": "attention.wq.weight",
    "attention.wk.weight": "attention.wk.weight",
    "attention.wv.weight": "attention.wv.weight",
    "attention.wo.weight": "attention.wo.weight",
    "feed_forward.w1.weight": "feed_forward.w1.weight",
    "feed_forward.w2.weight": "feed_forward.w2.weight",
    "feed_forward.w3.weight": "feed_forward.w3.weight",
    "attention_norm.weight": "attention_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
}


def _map_weights(state_dict: dict) -> dict:
    """Map safetensors weight names to our model's parameter names."""
    mapped = {}
    for key, value in state_dict.items():
        # Global weights
        if key in WEIGHT_MAP:
            mapped[WEIGHT_MAP[key]] = value
            continue

        # Layer weights: "layers.{i}.{subkey}"
        parts = key.split(".")
        if parts[0] == "layers" and len(parts) >= 3:
            layer_idx = parts[1]
            sub_key = ".".join(parts[2:])
            if sub_key in LAYER_WEIGHT_MAP:
                new_key = f"layers.{layer_idx}.{LAYER_WEIGHT_MAP[sub_key]}"
                mapped[new_key] = value
            else:
                mapped[key] = value
        else:
            mapped[key] = value

    return mapped


def load_model(
    config=None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    quantize: bool = True,
) -> MistralModel:
    """
    Load Mistral 7B from safetensors.

    Args:
        config: ModelConfig override
        device: Target device ('cuda', 'cpu')
        dtype: Data type (float16, float32)
        quantize: Whether to apply 4-bit quantization

    Returns:
        Loaded MistralModel ready for inference
    """
    cfg = config or model_config
    device = device or cfg.device
    dtype = dtype or cfg.dtype

    logger.info(f"Loading Mistral 7B from {cfg.model_path}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Quantize: {quantize}")

    # Step 1: Build model architecture
    model = MistralModel(cfg)

    # Step 2: Load safetensors weights
    weights_path = cfg.model_path / cfg.safetensors_file
    logger.info(f"Loading weights from {weights_path} ...")
    state_dict = load_file(str(weights_path))

    # Step 3: Map weight names
    mapped_dict = _map_weights(state_dict)

    # Step 4: Load into model
    missing, unexpected = model.load_state_dict(mapped_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Step 5: Quantize if requested and CUDA available
    if quantize and device == "cuda":
        try:
            import bitsandbytes as bnb
            logger.info("Applying 4-bit quantization...")
            model = _quantize_model(model, cfg)
        except ImportError:
            logger.warning("bitsandbytes not available, skipping quantization")

    # Step 6: Move to device
    model = model.to(device=device, dtype=dtype)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {param_count / 1e9:.2f}B parameters")

    return model


def _quantize_model(model: MistralModel, cfg) -> MistralModel:
    """Apply 4-bit quantization to linear layers using bitsandbytes."""
    import bitsandbytes as bnb

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "output" not in name:
            # Replace with 4-bit linear
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None

            new_layer = bnb.nn.Linear4bit(
                in_features,
                out_features,
                bias=has_bias,
                compute_dtype=compute_dtype,
                quant_type=cfg.bnb_4bit_quant_type,
            )

            # Copy weights
            new_layer.weight = bnb.nn.Params4bit(
                module.weight.data,
                requires_grad=False,
                quant_type=cfg.bnb_4bit_quant_type,
            )
            if has_bias:
                new_layer.bias = module.bias

            # Set the layer
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, new_layer)
            else:
                setattr(model, child_name, new_layer)

    return model
