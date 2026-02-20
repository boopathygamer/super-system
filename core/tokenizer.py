"""
Mistral Tokenizer Wrapper
─────────────────────────
Wraps SentencePiece tokenizer with Mistral Instruct chat template.
"""

import logging
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

from config.settings import model_config

logger = logging.getLogger(__name__)

# Special tokens
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
INST_START = "[INST]"
INST_END = "[/INST]"
IMG_TOKEN = "<image>"


class MistralTokenizer:
    """
    Tokenizer for Mistral 7B Instruct v0.3.

    Handles:
    - SentencePiece encoding/decoding
    - Mistral Instruct chat template formatting
    - Special token management for vision (image placeholders)
    """

    def __init__(self, model_path: Optional[str] = None):
        tokenizer_path = model_path or str(
            model_config.model_path / model_config.tokenizer_file
        )
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.LoadFromFile(tokenizer_path)

        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else self.eos_id
        self.vocab_size = self.sp.GetPieceSize()

        logger.info(
            f"Tokenizer loaded: vocab_size={self.vocab_size}, "
            f"bos={self.bos_id}, eos={self.eos_id}"
        )

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.sp.Encode(text)
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.Decode(tokens)

    def format_chat(
        self,
        messages: List[dict],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format messages into Mistral Instruct chat template.

        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system instruction to prepend

        Returns:
            Formatted prompt string

        Example:
            <s>[INST] System: ... \n\n User message [/INST] Assistant response </s>
            [INST] Follow up [/INST]
        """
        parts = []

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                if i == 0 and system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                parts.append(f"{INST_START} {content} {INST_END}")

            elif role == "assistant":
                parts.append(f" {content}{EOS_TOKEN}")

        # Join — first message gets BOS
        prompt = BOS_TOKEN + "".join(parts)

        # If last message was user, the model should generate next
        if messages[-1]["role"] == "user":
            pass  # prompt ends after [/INST], model generates
        # If last was assistant, strip trailing EOS for continuation
        elif messages[-1]["role"] == "assistant" and prompt.endswith(EOS_TOKEN):
            prompt = prompt[: -len(EOS_TOKEN)]

        return prompt

    def format_vision_chat(
        self,
        question: str,
        n_image_tokens: int = 576,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a vision query with image token placeholders.

        Args:
            question: User's question about the image
            n_image_tokens: Number of visual tokens to insert
            system_prompt: Optional system instruction

        Returns:
            Formatted prompt with <image> placeholders
        """
        image_placeholder = IMG_TOKEN * n_image_tokens
        content = f"{image_placeholder}\n{question}"

        if system_prompt:
            content = f"{system_prompt}\n\n{content}"

        return f"{BOS_TOKEN}{INST_START} {content} {INST_END}"

    def get_token_count(self, text: str) -> int:
        """Count tokens in text without special tokens."""
        return len(self.sp.Encode(text))
