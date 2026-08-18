"""
scripts/pretest_payloads.py
===========================
Pre-flight check for the live exam demo. Loads the REAL fine-tuned model, injects
the confidential block into the system prompt (as the backend does with
LEVELUP_DEMO=on), and runs each demo payload with the FIREWALL OFF to confirm the
model actually leaks the secret (Case 1). Then it confirms the guardrail would
block the same payload (Case 2).

Run once on your GPU before the exam:
    python -m scripts.pretest_payloads

Green "LEAKS" on Case 1 + "BLOCKED" on Case 2 for a payload = safe to use live.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from chatbot.guardrails import Guardrail
from chatbot.guardrails.demo_scenario import (
    CONFIDENTIAL_MARKERS,
    DEMO_PAYLOADS,
    DEMO_TEMPERATURE,
    inject_secret,
)
from chatbot.inference import LevelUpChat
from chatbot.prompt_template import FINETUNE_SYSTEM_PROMPTS

BUILD = "TITAN"


def leaked(text: str) -> bool:
    return any(m.lower() in text.lower() for m in CONFIDENTIAL_MARKERS)


def main():
    guard = Guardrail()
    system = inject_secret(FINETUNE_SYSTEM_PROMPTS[BUILD])

    print("Loading model (this can take ~30-60s)…")
    chat = LevelUpChat()
    chat.load()
    print("Model ready.\n")
    print("=" * 74)

    reliable = []
    for i, payload in enumerate(DEMO_PAYLOADS, 1):
        # Case 1: firewall OFF — call the model directly with the secret in context.
        reply = chat.generate(
            build=BUILD, user_message=payload,
            system_override=system, max_new_tokens=300, temperature=DEMO_TEMPERATURE,
        )
        did_leak = leaked(reply)

        # Case 2: firewall ON — would the input guardrail stop it?
        blocked = guard.check_input(payload).blocked

        status = "USE ✓" if (did_leak and blocked) else "review"
        print(f"\nPayload {i}: {payload}")
        print(f"  Case 1 (firewall off): {'LEAKS ✓' if did_leak else 'no leak ✗'}")
        print(f"  Case 2 (firewall on) : {'BLOCKED ✓' if blocked else 'passed ✗'}")
        print(f"  Verdict: {status}")
        print(f"  --- model reply (first 240 chars) ---\n  {reply[:240].strip()}")
        if did_leak and blocked:
            reliable.append((i, payload))

    print("\n" + "=" * 74)
    print(f"Reliable payloads (leak off / block on): {[i for i, _ in reliable]}")
    if not reliable:
        print("None leaked — try a stronger payload, or lower temperature (e.g. 0.3).")


if __name__ == "__main__":
    main()
