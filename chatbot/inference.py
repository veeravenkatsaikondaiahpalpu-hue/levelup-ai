"""
This is the brain of the chatbot — loads our fine-tuned LoRA adapter on top of
LLaMA 3.2-3B and handles all text generation.

Two ways to use it:
  1. Interactive CLI — great for testing the model during development
  2. Single-shot generate() — what the FastAPI backend calls on every chat request

Quick start:
  # Chat in the terminal with TITAN persona
  python -m chatbot.inference --adapter models/final/levelup-qlora-cloud --build TITAN

  # Get a single reply and exit
  python -m chatbot.inference --adapter models/final/levelup-qlora-cloud --build ORACLE \
      --message "Explain gradient descent in simple terms"
"""

import argparse
import sys
from pathlib import Path

# Try to import torch — it's optional so the server can still start
# even if the GPU model fails to load (Ollama fallback takes over)
try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot.prompt_template import (
    FINETUNE_SYSTEM_PROMPTS,
    format_inference_prompt,
    format_multi_turn_prompt,
)

# Where to find the models — can be overridden via CLI args or the FastAPI startup env vars
DEFAULT_BASE_MODEL  = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_ADAPTER_DIR = str(ROOT / "models" / "final" / "levelup-qlora-cloud")


class LevelUpChat:
    """
    Wraps our fine-tuned LevelUp LoRA model for inference.

    The model is lazy-loaded — it doesn't touch the GPU until you call load()
    or make the first generate() call. This keeps server startup fast.

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
        self.load_in_4bit = load_in_4bit
        self.model        = None
        self.tokenizer    = None
        self._loaded      = False

    def load(self):
        """
        Load the base model and attach our LoRA adapter.
        Call this once at server startup — it's expensive (~10-30 seconds).
        """
        if self._loaded:
            return
        if not _TORCH_AVAILABLE:
            print("  torch/transformers not installed — running in no-model mode")
            return

        print(f"Loading LevelUp model from {self.adapter_path} ...")

        # Pick the best device available — GPU if possible, otherwise CPU
        use_cuda      = _TORCH_AVAILABLE and torch.cuda.is_available()
        target_device = "cuda:0" if use_cuda else "cpu"
        dtype         = torch.float16 if use_cuda else torch.float32
        print(f"  Device: {target_device}  dtype: {dtype}")

        # Try 4-bit quantisation to save VRAM (needs bitsandbytes).
        # On Windows the CUDA kernels are unreliable, so we skip it and use fp16 directly.
        bnb_config = None
        if self.load_in_4bit and use_cuda:
            try:
                import bitsandbytes
                from transformers import BitsAndBytesConfig
                bitsandbytes.cuda_setup.CUDASetup.get_instance()
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                print("  Using 4-bit quantisation (bitsandbytes)")
            except Exception as e:
                print(f"  4-bit unavailable ({e}) — falling back to fp16 on GPU")
                bnb_config = None

        # Load the base LLaMA model
        load_kwargs = dict(
            trust_remote_code=False,
            attn_implementation="eager",  # swap to "flash_attention_2" if installed for speed
        )
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["device_map"] = target_device

        base = AutoModelForCausalLM.from_pretrained(self.base_model, **load_kwargs)

        # Attach our fine-tuned LoRA weights on top of the base model
        self.model = PeftModel.from_pretrained(base, self.adapter_path)
        self.model.eval()

        # Load the tokenizer — we saved it alongside the adapter
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            trust_remote_code=False,
        )
        # LLaMA 3 has no pad token by default — reuse the EOS token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._loaded = True
        print("  Model loaded and ready.")

    def generate(
        self,
        build:        str,
        user_message: str,
        history:      list[dict] | None = None,
        system_override: str | None = None,
        max_new_tokens:  int   = 512,
        temperature:     float = 0.7,
        top_p:           float = 0.9,
        repetition_penalty: float = 1.1,
        stream:          bool  = False,
    ) -> str:
        """
        Generate a reply from the fine-tuned model.

        Args:
            build            : which persona to use ("TITAN", "ORACLE", etc.)
            user_message     : the user's latest message
            history          : previous turns as [{"role": "user"|"assistant", "content": "..."}]
            system_override  : custom system prompt (used when we have live user stats)
            max_new_tokens   : how many tokens to generate before stopping
            temperature      : creativity dial — lower is more focused, higher is more random
            top_p            : nucleus sampling threshold
            repetition_penalty: discourages the model from repeating itself
            stream           : stream tokens to stdout in real time (nice for CLI demos)

        Returns:
            The assistant's reply as a plain string.
        """
        if not self._loaded:
            self.load()

        build  = build.upper()
        system = system_override or FINETUNE_SYSTEM_PROMPTS.get(build, FINETUNE_SYSTEM_PROMPTS["ORACLE"])

        # Build the full prompt — multi-turn if we have history, single-shot otherwise
        if history:
            prompt = format_multi_turn_prompt(system, history, user_message)
        else:
            prompt = format_inference_prompt(system, user_message)

        # Tokenise and move to GPU — leave headroom for the new tokens we're about to generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.model.device)

        # Stop generation when the model produces <|eot_id|> (end-of-turn in LLaMA 3)
        eot_id   = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        stop_ids = [eot_id] if eot_id is not None else []

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

        # A very low temperature switches to greedy (deterministic) decoding — used
        # by the demo so the before/after is reproducible on stage, not stochastic.
        do_sample = temperature is not None and temperature > 0.05
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            eos_token_id=stop_ids or self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the input tokens — we only want the model's new output
        input_len  = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        reply      = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return reply

    def interactive(self, build: str):
        """
        Start an interactive CLI chat session — handy for quick demos or debugging.
        Type 'quit' or 'exit' to end the session.
        """
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

            # Keep the last 6 turns (3 full exchanges) so context doesn't blow up
            history.append({"role": "user",      "content": user_msg})
            history.append({"role": "assistant",  "content": reply})
            if len(history) > 12:
                history = history[-12:]

            print()  # newline after the streamed reply


# Module-level singleton so FastAPI only loads the model once
_chat_instance: LevelUpChat | None = None


def get_chat_model(
    adapter_path: str = DEFAULT_ADAPTER_DIR,
    base_model:   str = DEFAULT_BASE_MODEL,
) -> LevelUpChat:
    """
    Returns the global LevelUpChat instance, creating and loading it on first call.
    Subsequent calls return the same object — no re-loading.
    Used by FastAPI's startup event.
    """
    global _chat_instance
    if _chat_instance is None:
        _chat_instance = LevelUpChat(adapter_path=adapter_path, base_model=base_model)
        _chat_instance.load()
    return _chat_instance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LevelUp AI inference")
    p.add_argument("--adapter",    default=DEFAULT_ADAPTER_DIR,
                   help="Path to the LoRA adapter folder")
    p.add_argument("--base",       default=DEFAULT_BASE_MODEL,
                   help="HuggingFace model ID or local path for the base model")
    p.add_argument("--build",      default="ORACLE",
                   choices=["TITAN", "ORACLE", "PHANTOM", "SAGE", "MUSE", "EMPIRE", "GG"],
                   help="Which persona to use")
    p.add_argument("--message",    default=None,
                   help="Single message (skips the interactive loop)")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--temp",       type=float, default=0.7,
                   help="Sampling temperature")
    p.add_argument("--no_4bit",    action="store_true",
                   help="Disable 4-bit quantisation (useful if bitsandbytes is broken)")
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
        # Non-interactive mode — print the reply and exit
        reply = chat.generate(
            build=args.build,
            user_message=args.message,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            stream=True,
        )
        print()
    else:
        # Drop into the interactive chat loop
        chat.interactive(build=args.build)
