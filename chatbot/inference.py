"""
inference.py
============
Load a fine-tuned LevelUp LoRA adapter and run chat inference.

Supports two modes:
  1. Interactive CLI chat (for testing / demo)
  2. Single-response function (used by the FastAPI backend)

Usage:
  # Interactive CLI
  python -m chatbot.inference --adapter models/final/levelup-qlora-all --build TITAN

  # One-shot test
  python -m chatbot.inference --adapter models/final/levelup-qlora-all --build ORACLE \
      --message "Explain gradient descent in simple terms"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.prompt_template import (
    format_inference_prompt,
    format_multi_turn_prompt,
    FINETUNE_SYSTEM_PROMPTS,
    detect_build,
)

# ── Default model paths ────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_ADAPTER_DIR = str(ROOT / "models" / "final" / "levelup-qlora-all")


# ── Model loader ───────────────────────────────────────────────────────────────

class LevelUpChat:
    """
    Wraps a fine-tuned LevelUp LoRA model for easy inference.

    Example:
        chat = LevelUpChat(adapter_path="models/final/levelup-qlora-all")
        reply = chat.generate(build="TITAN", user_message="How do I fix my deadlift?")
    """

    def __init__(
        self,
        adapter_path: str = DEFAULT_ADAPTER_DIR,
        base_model:   str = DEFAULT_BASE_MODEL,
        device:       str = "auto",
        load_in_4bit: bool = True,
    ):
        self.adapter_path = adapter_path
        self.base_model   = base_model
        self.device       = device
        self.model        = None
        self.tokenizer    = None
        self._loaded      = False

    def load(self):
        """Lazy-load the model (call once at server startup)."""
        if self._loaded:
            return

        print(f"Loading LevelUp model from {self.adapter_path} ...")

        # Quantisation config (4-bit for inference efficiency)
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
                print("  BitsAndBytes not available — loading in full precision")

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            attn_implementation="eager",
        )

        # Attach LoRA adapter
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        print("  Model loaded and ready.")

    # ── Core generation ────────────────────────────────────────────────────────

    def generate(
        self,
        build:        str,
        user_message: str,
        history:      Optional[list[dict]] = None,
        system_override: Optional[str] = None,
        max_new_tokens:  int   = 512,
        temperature:     float = 0.7,
        top_p:           float = 0.9,
        repetition_penalty: float = 1.1,
        stream:          bool  = False,
    ) -> str:
        """
        Generate a reply from the fine-tuned model.

        Args:
            build            : build name ("TITAN", "ORACLE", etc.)
            user_message     : the user's latest message
            history          : list of prior turns [{"role": "user"|"assistant", "content": "..."}]
            system_override  : pass a custom system prompt (e.g. from system_prompt.py at runtime)
            max_new_tokens   : max tokens to generate
            temperature      : sampling temperature (lower = more deterministic)
            top_p            : nucleus sampling probability
            repetition_penalty: penalise token repetition
            stream           : if True, stream tokens to stdout

        Returns:
            The assistant's reply as a plain string.
        """
        if not self._loaded:
            self.load()

        build = build.upper()
        system = system_override or FINETUNE_SYSTEM_PROMPTS.get(build, FINETUNE_SYSTEM_PROMPTS["ORACLE"])

        # Format prompt
        if history:
            prompt = format_multi_turn_prompt(system, history, user_message)
        else:
            prompt = format_inference_prompt(system, user_message)

        # Tokenise
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,        # leave headroom for 512 new tokens
        ).to(self.model.device)

        # Stop at <|eot_id|>
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        stop_ids = [eot_id] if eot_id is not None else []

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=stop_ids or self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                streamer=streamer,
            )

        # Decode only the newly generated tokens
        input_len   = inputs["input_ids"].shape[1]
        new_tokens  = output[0][input_len:]
        reply       = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return reply

    # ── Convenience ───────────────────────────────────────────────────────────

    def interactive(self, build: str):
        """Start an interactive CLI chat session."""
        if not self._loaded:
            self.load()

        print(f"\n{'='*55}")
        print(f"  LevelUp AI — {build} build  (type 'quit' to exit)")
        print(f"{'='*55}\n")

        history: list[dict] = []

        while True:
            try:
                user_msg = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("quit", "exit", "q"):
                print("GG! Logging off.")
                break

            reply = self.generate(
                build=build,
                user_message=user_msg,
                history=history,
                stream=True,
            )

            # Append to history
            history.append({"role": "user",      "content": user_msg})
            history.append({"role": "assistant",  "content": reply})

            # Keep last 6 turns (3 exchanges) in history to avoid context overflow
            if len(history) > 12:
                history = history[-12:]

            print()  # newline after streamed reply


# ── Singleton for FastAPI ──────────────────────────────────────────────────────

_chat_instance: Optional[LevelUpChat] = None

def get_chat_model(
    adapter_path: str = DEFAULT_ADAPTER_DIR,
    base_model:   str = DEFAULT_BASE_MODEL,
) -> LevelUpChat:
    """
    Return the global LevelUpChat singleton.
    Creates and loads it on first call (lazy initialisation).
    Use this in the FastAPI startup event.
    """
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = LevelUpChat(adapter_path=adapter_path, base_model=base_model)
        _chat_instance.load()
    return _chat_instance


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LevelUp AI inference")
    p.add_argument("--adapter",  default=DEFAULT_ADAPTER_DIR,
                   help="Path to LoRA adapter directory")
    p.add_argument("--base",     default=DEFAULT_BASE_MODEL,
                   help="Base model HuggingFace ID or local path")
    p.add_argument("--build",    default="ORACLE",
                   choices=["TITAN", "ORACLE", "PHANTOM", "SAGE", "MUSE", "EMPIRE", "GG"],
                   help="Which build persona to use")
    p.add_argument("--message",  default=None,
                   help="Single message (non-interactive mode)")
    p.add_argument("--max_tokens", type=int, default=512,
                   help="Max new tokens to generate")
    p.add_argument("--temp",     type=float, default=0.7,
                   help="Sampling temperature")
    p.add_argument("--no_4bit",  action="store_true",
                   help="Disable 4-bit quantisation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    chat = LevelUpChat(
        adapter_path=args.adapter,
        base_model=args.base,
        load_in_4bit=not args.no_4bit,
    )
    chat.load()

    if args.message:
        # Non-interactive: print single reply
        reply = chat.generate(
            build=args.build,
            user_message=args.message,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            stream=True,
        )
        print()
    else:
        # Interactive chat
        chat.interactive(build=args.build)
