"""
chatbot/ollama_inference.py
============================
Local fallback backend using Ollama — kicks in automatically when the LoRA
adapter can't load (e.g. no GPU on this machine, or torch isn't installed).

Same interface as LevelUpChat so the FastAPI chat route doesn't need to know
which backend it's talking to. Uses the exact same few-shot system prompts as
the fine-tuned model so persona quality stays consistent either way.

Requirements:
  - Ollama running locally on port 11434
  - llama3.2 pulled:  ollama pull llama3.2
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from pathlib import Path

# Pull the same system prompts the LoRA model uses — one source of truth for all
# persona definitions. Both backends stay in sync automatically this way.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS as BUILD_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"


def _ollama_available() -> bool:
    """Ping Ollama and check that llama3.2 is actually pulled and ready to use."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data   = json.loads(resp.read())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return "llama3.2" in models
    except Exception:
        return False


class OllamaChat:
    """
    Drop-in replacement for LevelUpChat that routes requests to a local Ollama instance.

    The API surface is identical — the FastAPI chat route doesn't need to care
    which backend is active. generate() works the same way either way.
    """

    def __init__(self):
        self._loaded = _ollama_available()
        if self._loaded:
            logger.info("Ollama backend ready (llama3.2)")
        else:
            logger.warning("Ollama not available - chat will fail")

    def load(self):
        """No-op for API compatibility with LevelUpChat. Ollama is always-on."""
        self._loaded = _ollama_available()

    def generate(
        self,
        build:           str,
        user_message:    str,
        history:         list[dict] | None = None,
        system_override: str | None = None,
        max_new_tokens:  int   = 512,
        temperature:     float = 0.7,
        **kwargs,  # absorb extra args the LoRA model accepts (top_p, repetition_penalty, etc.)
    ) -> str:
        build  = build.upper()
        system = system_override or BUILD_SYSTEM_PROMPTS.get(build, BUILD_SYSTEM_PROMPTS["ORACLE"])

        # Build the Ollama message list — system prompt first, then last 10 history turns
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model":    OLLAMA_MODEL,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature": temperature,
                "num_predict": max_new_tokens,
                "top_p":       0.9,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # 120-second timeout — local models can be slow on CPU-only machines
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result["message"]["content"].strip()


def get_ollama_model() -> OllamaChat | None:
    """
    Try to connect to Ollama and return an OllamaChat instance if it's running.
    Returns None if Ollama isn't available so the caller can handle it gracefully.
    """
    chat = OllamaChat()
    return chat if chat._loaded else None
