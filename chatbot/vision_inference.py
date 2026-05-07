"""
vision_inference.py
===================
Multimodal image + text inference using Qwen2.5-VL-7B-Instruct.

No fine-tuning needed — Qwen2.5-VL understands images out of the box.
We inject the build persona via the system prompt so each build still
responds in character when analysing a photo.

Usage:
  from chatbot.vision_inference import get_vision_model

  model = get_vision_model()
  reply = model.generate(
      build="TITAN",
      user_message="Is my squat depth good?",
      image_input=pil_image,   # PIL.Image or base64 string or file path
  )
"""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS

# ── Vision-specific system prompt additions ────────────────────────────────────
# Appended to each build's normal system prompt when an image is provided

VISION_SUFFIX: dict[str, str] = {
    "TITAN": (
        " You can analyse workout photos and form videos. When given an image, "
        "assess form, posture, muscle engagement, or training setup with the same "
        "intensity you bring to all coaching. Be direct about what's right and wrong."
    ),
    "ORACLE": (
        " You can analyse images, diagrams, notes, code screenshots, and study materials. "
        "When given an image, break it down systematically and extract every insight "
        "that will sharpen the user's understanding."
    ),
    "PHANTOM": (
        " You can analyse movement photos and technique videos. When given an image, "
        "assess body positioning, alignment, landing mechanics, and movement quality "
        "with the precision of a movement expert."
    ),
    "SAGE": (
        " You can analyse yoga poses, meditation setups, and wellness-related images. "
        "When given an image, offer calm, grounded observations about alignment, "
        "environment, and how the scene supports or hinders wellbeing."
    ),
    "MUSE": (
        " You can analyse artwork, sketches, designs, music sheets, and creative works. "
        "When given an image, engage with it creatively — notice composition, technique, "
        "emotion, and potential. Be encouraging and specific."
    ),
    "EMPIRE": (
        " You can analyse charts, business plans, whiteboards, and strategy documents. "
        "When given an image, extract the key insights and translate them into "
        "actionable strategy with the precision of a seasoned entrepreneur."
    ),
    "GG": (
        " You can analyse screenshots, setups, and gameplay. When given an image, "
        "break down what you see with full gamer energy — settings, crosshair placement, "
        "build choices, or anything else that matters for performance. No cap."
    ),
}

# ── Default model ──────────────────────────────────────────────────────────────
DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"


# ── Helper: normalise image input to PIL ──────────────────────────────────────

def _to_pil(image_input: Union[str, bytes, "Image.Image"]) -> "Image.Image":  # noqa: F821
    """
    Accept a PIL Image, base64 string, raw bytes, or a file path.
    Always returns a PIL.Image.Image in RGB mode.
    """
    from PIL import Image

    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, str):
        # Could be a file path or a base64 data URI
        if image_input.startswith("data:"):
            # data:image/jpeg;base64,<data>
            _, encoded = image_input.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        # Assume raw base64
        return Image.open(io.BytesIO(base64.b64decode(image_input))).convert("RGB")

    raise TypeError(f"Unsupported image_input type: {type(image_input)}")


# ── VisionChat class ───────────────────────────────────────────────────────────

class VisionChat:
    """
    Wraps Qwen2.5-VL-7B-Instruct for multimodal (image + text) inference.

    Loads in 4-bit quantisation so it fits in ~8 GB VRAM alongside the
    existing 3B text model (or alone on systems with limited VRAM).

    Example:
        vc = VisionChat()
        reply = vc.generate(
            build="TITAN",
            user_message="Is my squat form good?",
            image_input=pil_image,
        )
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VLM,
        load_in_4bit: bool = True,
        device: str = "auto",
    ):
        self.model_name  = model_name
        self.load_in_4bit = load_in_4bit
        self.device      = device
        self.model       = None
        self.processor   = None
        self._loaded     = False

    # ── Loader ────────────────────────────────────────────────────────────────

    def load(self):
        """Lazy-load Qwen2.5-VL (call once at server startup)."""
        if self._loaded:
            return

        print(f"\n[VisionChat] Loading {self.model_name} ...")

        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        # 4-bit config
        bnb_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                print("  [VisionChat] BitsAndBytes not available — loading in bfloat16")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._loaded = True
        print("[VisionChat] Vision model ready.\n")

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(
        self,
        build:        str,
        user_message: str,
        image_input:  Optional[Union[str, bytes, "Image.Image"]] = None,  # noqa: F821
        history:      Optional[list[dict]] = None,
        max_new_tokens: int   = 512,
        temperature:    float = 0.7,
        top_p:          float = 0.9,
    ) -> str:
        """
        Generate a reply, optionally conditioned on an image.

        Args:
            build         : "TITAN" | "ORACLE" | "PHANTOM" | "SAGE" | "MUSE" | "EMPIRE" | "GG"
            user_message  : the user's text question
            image_input   : PIL Image, base64 string, raw bytes, or file path (optional)
            history       : prior turns [{"role": "user"|"assistant", "content": "..."}]
            max_new_tokens: max tokens to generate
            temperature   : sampling temperature
            top_p         : nucleus sampling

        Returns:
            The assistant reply as a plain string.
        """
        if not self._loaded:
            self.load()

        import torch

        build = build.upper()
        base_sys = FINETUNE_SYSTEM_PROMPTS.get(build, FINETUNE_SYSTEM_PROMPTS["ORACLE"])
        vision_add = VISION_SUFFIX.get(build, "") if image_input is not None else ""
        system = base_sys + vision_add

        # ── Build messages list ────────────────────────────────────────────────
        messages: list[dict] = [{"role": "system", "content": system}]

        # Prior history (text only — no images in history for simplicity)
        if history:
            for turn in history[-10:]:
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Current user turn (with optional image)
        if image_input is not None:
            pil_img = _to_pil(image_input)
            user_content = [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": user_message},
            ]
        else:
            user_content = user_message

        messages.append({"role": "user", "content": user_content})

        # ── Tokenise using Qwen's chat template ───────────────────────────────
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # process_vision_info handles PIL Images embedded in messages
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # ── Generate ──────────────────────────────────────────────────────────
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode only newly generated tokens
        input_len   = inputs["input_ids"].shape[1]
        new_tokens  = output_ids[0][input_len:]
        reply       = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return reply

    # ── Convenience ───────────────────────────────────────────────────────────

    def unload(self):
        """Free VRAM (call at shutdown)."""
        import gc, torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        print("[VisionChat] Vision model unloaded.")


# ── Singleton for FastAPI ──────────────────────────────────────────────────────

_vision_instance: Optional[VisionChat] = None


def get_vision_model(
    model_name:  str  = DEFAULT_VLM,
    load_in_4bit: bool = True,
) -> VisionChat:
    """
    Return the global VisionChat singleton.
    Creates and loads it on first call.
    Use in the FastAPI startup event.
    """
    global _vision_instance
    if _vision_instance is None:
        _vision_instance = VisionChat(model_name=model_name, load_in_4bit=load_in_4bit)
        _vision_instance.load()
    return _vision_instance
