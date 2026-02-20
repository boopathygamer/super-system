"""
Inference Engine
────────────────
Text generation with temperature sampling, top-p/top-k, KV-cache,
and streaming support on top of MistralModel.
"""

import logging
from typing import Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from config.settings import model_config, generation_config
from core.model_loader import MistralModel
from core.tokenizer import MistralTokenizer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    High-performance inference engine for Mistral 7B.

    Features:
    - KV-cache for efficient autoregressive generation
    - Temperature sampling with top-p and top-k
    - Repetition penalty
    - Streaming token generation
    - Batch inference support
    """

    def __init__(
        self,
        model: MistralModel,
        tokenizer: MistralTokenizer,
        gen_config=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config or generation_config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Returns:
            Generated text (without the prompt).
        """
        tokens = []
        for token_text in self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
        ):
            tokens.append(token_text)
        return "".join(tokens)

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Generator[str, None, None]:
        """
        Stream-generate text token by token.

        Yields:
            Generated text chunks (usually one token each).
        """
        cfg = self.gen_config
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature
        top_p = top_p or cfg.top_p
        top_k = top_k or cfg.top_k
        repetition_penalty = repetition_penalty or cfg.repetition_penalty
        stop_sequences = stop_sequences or cfg.stop_sequences

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # Prefill: process entire prompt at once with KV-cache
        if input_embeds is not None:
            logits, kv_caches = self.model(
                input_ids=input_tensor,
                input_embeds=input_embeds,
            )
        else:
            logits, kv_caches = self.model(input_ids=input_tensor)

        # Track generated tokens for repetition penalty
        generated_ids = list(input_ids)
        generated_text = ""

        for step in range(max_new_tokens):
            # Get logits for last position
            next_logits = logits[:, -1, :].float()

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, repetition_penalty
                )

            # Sample next token
            next_token = self._sample(next_logits, temperature, top_p, top_k)
            next_token_id = next_token.item()

            # Check for EOS
            if next_token_id == self.tokenizer.eos_id:
                break

            generated_ids.append(next_token_id)

            # Decode token
            token_text = self.tokenizer.decode([next_token_id])
            generated_text += token_text

            # Check stop sequences
            should_stop = False
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    # Trim text at stop sequence
                    idx = generated_text.find(stop_seq)
                    trimmed = generated_text[:idx]
                    if trimmed:
                        yield trimmed
                    should_stop = True
                    break

            if should_stop:
                break

            yield token_text

            # Decode step: process only the new token with KV-cache
            next_input = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )
            logits, kv_caches = self.model(
                input_ids=next_input,
                kv_caches=kv_caches,
            )

    @torch.inference_mode()
    def generate_with_embeddings(
        self,
        embeddings: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, List[int]]:
        """
        Generate from pre-computed embeddings (for vision pipeline).

        Args:
            embeddings: [1, seq_len, dim] tensor of input embeddings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            (generated_text, token_ids)
        """
        cfg = self.gen_config
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature

        # Prefill with embeddings
        logits, kv_caches = self.model(
            input_ids=torch.zeros(1, 0, dtype=torch.long, device=self.device),
            input_embeds=embeddings,
        )

        generated_ids = []
        generated_text = ""

        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()

            if cfg.repetition_penalty != 1.0 and generated_ids:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_ids, cfg.repetition_penalty
                )

            next_token = self._sample(next_logits, temperature, cfg.top_p, cfg.top_k)
            next_token_id = next_token.item()

            if next_token_id == self.tokenizer.eos_id:
                break

            generated_ids.append(next_token_id)
            token_text = self.tokenizer.decode([next_token_id])
            generated_text += token_text

            # Check stop sequences
            should_stop = False
            for stop_seq in cfg.stop_sequences:
                if stop_seq in generated_text:
                    idx = generated_text.find(stop_seq)
                    generated_text = generated_text[:idx]
                    should_stop = True
                    break
            if should_stop:
                break

            # Decode step
            next_input = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )
            logits, kv_caches = self.model(
                input_ids=next_input,
                kv_caches=kv_caches,
            )

        return generated_text, generated_ids

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """Sample a token from logits with temperature, top-p, and top-k."""
        if temperature <= 0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")

            # Scatter back
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        token_ids: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """Penalize tokens that have already appeared."""
        unique_ids = list(set(token_ids))
        if not unique_ids:
            return logits

        unique_tensor = torch.tensor(unique_ids, dtype=torch.long, device=logits.device)
        selected = logits[0, unique_tensor]

        # Apply penalty: divide positive logits, multiply negative
        selected = torch.where(
            selected > 0, selected / penalty, selected * penalty
        )
        logits[0, unique_tensor] = selected

        return logits
