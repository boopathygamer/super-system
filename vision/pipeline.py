"""
Vision Pipeline
───────────────
End-to-end image analysis: Image → CLIP → Project → Fuse with LLM → Generate.
This is the main entry point for all image understanding tasks.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from config.settings import model_config, vision_config, generation_config
from core.model_loader import MistralModel
from core.tokenizer import MistralTokenizer
from core.inference import InferenceEngine
from vision.encoder import VisionEncoder
from vision.projector import VisionProjector

logger = logging.getLogger(__name__)

# Expert-level system prompts for different analysis modes
ANALYSIS_PROMPTS = {
    "general": (
        "You are an expert image analyst with deep knowledge across all visual domains. "
        "Analyze the image thoroughly, noting composition, subjects, colors, context, "
        "and any notable details. Provide a comprehensive, structured analysis."
    ),
    "technical": (
        "You are a technical image analysis expert. Examine this image with precision: "
        "identify all technical elements, measurements, patterns, anomalies, and "
        "provide detailed technical observations with confidence levels."
    ),
    "medical": (
        "You are a medical imaging specialist. Analyze this medical image carefully, "
        "identifying anatomical structures, potential abnormalities, and relevant "
        "clinical observations. Note: This is for educational purposes only."
    ),
    "document": (
        "You are a document analysis expert. Extract all text, analyze layout, "
        "identify document type, key information, tables, and structure. "
        "Provide organized extraction of all content."
    ),
    "creative": (
        "You are an art and design critic with expertise in visual aesthetics. "
        "Analyze the artistic elements: composition, color theory, style, technique, "
        "emotional impact, and cultural references."
    ),
    "code": (
        "You are a software engineering expert. Analyze this code/diagram/architecture "
        "image. Extract all visible code, identify patterns, explain the architecture, "
        "and suggest improvements."
    ),
}


class VisionPipeline:
    """
    End-to-end multimodal vision pipeline.

    Flow:
        1. Load image → preprocess
        2. CLIP encode → patch embeddings [B, 576, 768]
        3. Project → visual tokens [B, 576, 4096]
        4. Fuse with text embeddings (prepend visual tokens)
        5. Generate response via Mistral

    Usage:
        pipeline = VisionPipeline(model, tokenizer, engine)
        result = pipeline.analyze("photo.jpg", "What's in this image?")
    """

    def __init__(
        self,
        model: MistralModel,
        tokenizer: MistralTokenizer,
        engine: InferenceEngine,
        vision_encoder: Optional[VisionEncoder] = None,
        projector: Optional[VisionProjector] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = engine
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Vision components
        self.encoder = vision_encoder or VisionEncoder()
        self.projector = projector or VisionProjector()

        # Move to device
        self.encoder.to(self.device)
        self.projector.to(device=self.device, dtype=self.dtype)

        logger.info("Vision pipeline initialized")

    @torch.inference_mode()
    def analyze(
        self,
        image: Union[str, Image.Image],
        question: str = "Describe this image in detail.",
        mode: str = "general",
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        chain_of_thought: bool = True,
    ) -> str:
        """
        Analyze an image with expert-level detail.

        Args:
            image: Path to image or PIL Image
            question: Question about the image
            mode: Analysis mode (general, technical, medical, document, creative, code)
            max_new_tokens: Max tokens for response
            temperature: Sampling temperature (lower = more focused)
            chain_of_thought: Whether to use chain-of-thought prompting

        Returns:
            Detailed analysis text
        """
        # Step 1: Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        logger.info(f"Analyzing image ({image.size}) with mode='{mode}'")

        # Step 2: Encode image with CLIP
        vision_features = self.encoder.encode_image(image, device=self.device)
        # vision_features: [1, 576, 768]

        # Step 3: Project to LLM space
        visual_tokens = self.projector(vision_features.to(self.dtype))
        # visual_tokens: [1, 576, 4096]

        # Step 4: Build text prompt
        system_prompt = ANALYSIS_PROMPTS.get(mode, ANALYSIS_PROMPTS["general"])

        if chain_of_thought:
            question = (
                f"{question}\n\n"
                "Think step by step:\n"
                "1. First, describe what you observe in the image\n"
                "2. Then, analyze the key elements and their relationships\n"
                "3. Finally, provide your detailed assessment"
            )

        # Format the text part
        text_prompt = self.tokenizer.format_vision_chat(
            question=question,
            n_image_tokens=visual_tokens.shape[1],
            system_prompt=system_prompt,
        )

        # Step 5: Build combined embeddings
        combined_embeds = self._fuse_vision_text(visual_tokens, text_prompt)

        # Step 6: Generate response
        response, _ = self.engine.generate_with_embeddings(
            embeddings=combined_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return response.strip()

    @torch.inference_mode()
    def analyze_multiple(
        self,
        images: List[Union[str, Image.Image]],
        question: str,
        mode: str = "general",
    ) -> str:
        """Analyze multiple images together."""
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        # Encode all images
        all_visual_tokens = []
        for img in pil_images:
            features = self.encoder.encode_image(img, device=self.device)
            tokens = self.projector(features.to(self.dtype))
            all_visual_tokens.append(tokens)

        # Concatenate visual tokens
        combined_visual = torch.cat(all_visual_tokens, dim=1)

        # Build prompt
        system_prompt = ANALYSIS_PROMPTS.get(mode, ANALYSIS_PROMPTS["general"])
        multi_question = (
            f"You are shown {len(images)} images. {question}\n"
            "Analyze each image and provide a comprehensive comparison."
        )

        text_prompt = self.tokenizer.format_vision_chat(
            question=multi_question,
            n_image_tokens=combined_visual.shape[1],
            system_prompt=system_prompt,
        )

        combined_embeds = self._fuse_vision_text(combined_visual, text_prompt)

        response, _ = self.engine.generate_with_embeddings(
            embeddings=combined_embeds,
            max_new_tokens=2048,
            temperature=0.3,
        )

        return response.strip()

    def _fuse_vision_text(
        self,
        visual_tokens: torch.Tensor,
        text_prompt: str,
    ) -> torch.Tensor:
        """
        Fuse visual tokens with text embeddings.

        Strategy: Replace <image> placeholder tokens with visual tokens,
        or prepend visual tokens before text tokens.

        Args:
            visual_tokens: [1, n_visual, dim]
            text_prompt: Formatted text prompt

        Returns:
            Combined embeddings [1, n_visual + n_text, dim]
        """
        # Tokenize text
        text_ids = self.tokenizer.encode(text_prompt, add_bos=False)
        text_tensor = torch.tensor([text_ids], dtype=torch.long, device=self.device)

        # Get text embeddings from the model's embedding layer
        text_embeds = self.model.tok_embeddings(text_tensor)
        # text_embeds: [1, n_text, 4096]

        # Find image token positions and replace, or simply prepend
        # For simplicity and robustness, we prepend visual tokens
        combined = torch.cat([visual_tokens, text_embeds], dim=1)

        return combined

    def get_supported_modes(self) -> List[str]:
        """Return list of supported analysis modes."""
        return list(ANALYSIS_PROMPTS.keys())
