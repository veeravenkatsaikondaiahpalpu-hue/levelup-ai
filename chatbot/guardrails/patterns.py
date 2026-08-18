"""
The rule library for the LevelUp guardrail — the "signatures" the scanners match on.

Every pattern is tagged with:
  - a category   (what kind of attack/leak it is)
  - a severity   (low | medium | high | critical)
  - an OWASP id  (which OWASP LLM Top-10 2025 risk it maps to)

Keeping the rules in one place makes the guardrail explainable: for the
presentation you can point at exactly which signature fired and why. This is a
heuristic, defence-in-depth layer — it is the independent second control wrapped
around the model, not a claim that the model itself is safe.

OWASP mapping used here:
  LLM01  Prompt Injection (direct + indirect)          -> the core threat
  LLM02  Sensitive Information Disclosure               -> system-prompt / PII leaks
  LLM06  Excessive Agency / data exfiltration channels  -> auto-loading images, links
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Severity ordering (higher = worse)
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Rule:
    """A single detection signature."""
    name:     str            # short id, e.g. "instruction_override"
    category: str            # human-readable group
    severity: str            # low | medium | high | critical
    owasp:    str            # e.g. "LLM01"
    pattern:  re.Pattern     # compiled, case-insensitive
    note:     str            # one-line explanation for reports / slides


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# INPUT rules — run on the user message, history and ingested context
# These catch an attacker trying to hijack the model. Patterns are written to be
# specific enough that normal fitness / study / gaming chat does not trip them.

INPUT_RULES: list[Rule] = [
    Rule(
        name="instruction_override",
        category="Direct prompt injection",
        severity="high",
        owasp="LLM01",
        pattern=_rx(r"\b(ignore|disregard|forget|override|bypass)\b[\s\S]{0,40}?"
                    r"\b(previous|prior|above|earlier|all|your|the)\b[\s\S]{0,20}?"
                    r"\b(instruction|instructions|prompt|prompts|rule|rules|context|directive)"),
        note="Classic 'ignore previous instructions' override attempt.",
    ),
    Rule(
        name="persona_hijack",
        category="Direct prompt injection",
        severity="high",
        owasp="LLM01",
        pattern=_rx(r"\byou are now\b|\bfrom now on you\b|\bact as (an?|the)\b|"
                    r"\bpretend to be\b|\byou must now\b|\benter .{0,20}mode\b"),
        note="Tries to replace the assigned build persona with a new one.",
    ),
    Rule(
        name="jailbreak_alias",
        category="Jailbreak",
        severity="high",
        owasp="LLM01",
        pattern=_rx(r"\b(DAN|do anything now|developer mode|jailbreak|"
                    r"unfiltered|no restrictions|without any restrictions)\b"),
        note="Named jailbreak persona / 'no restrictions' framing (DAN, Developer Mode).",
    ),
    Rule(
        name="system_prompt_extraction",
        category="System-prompt extraction",
        severity="high",
        owasp="LLM02",
        pattern=_rx(r"\b(reveal|show|print|repeat|display|output|tell me|what (is|are)|give me)\b"
                    r"[\s\S]{0,40}?\b(system prompt|system message|your (instructions|rules|prompt)|"
                    r"initial (instructions|prompt)|everything above|the text above)"),
        note="Attempts to make the model disclose its hidden system instructions.",
    ),
    Rule(
        name="role_token_injection",
        category="Delimiter / role injection",
        severity="critical",
        owasp="LLM01",
        # LLaMA-3 chat control tokens, or fake chat-role delimiters, appearing in
        # untrusted text — an attempt to forge a new system/assistant turn.
        pattern=_rx(r"<\|(start_header_id|end_header_id|eot_id|begin_of_text)\|>|"
                    r"(^|\n)\s*(system|assistant)\s*:\s|###\s*(system|instruction)"),
        note="Forged chat-role delimiters / control tokens to break out of the data channel.",
    ),
    Rule(
        name="indirect_injection_directive",
        category="Indirect prompt injection",
        severity="medium",
        owasp="LLM01",
        # Imperative instructions embedded inside ingested data (the EchoLeak shape:
        # the attack lives in the document, not in a typed prompt).
        pattern=_rx(r"\b(when|if) (you|the assistant|the ai)[\s\S]{0,40}?"
                    r"\b(reply|respond|answer|say|send|include|append|forward)\b|"
                    r"\bdo not (tell|mention|inform|alert) the user\b|"
                    r"\bwithout (telling|informing|alerting) the user\b"),
        note="Instruction hidden inside ingested content telling the model to act covertly.",
    ),
    Rule(
        name="exfiltration_request",
        category="Data exfiltration",
        severity="high",
        owasp="LLM06",
        pattern=_rx(r"\b(send|post|exfiltrate|leak|upload|transmit|forward|email)\b"
                    r"[\s\S]{0,40}?\b(to|at)\b[\s\S]{0,20}?"
                    r"(https?://|www\.|[\w.-]+@[\w.-]+|[\w-]+\.(com|net|io|org|dev))"),
        note="Asks the model to send data to an external address (exfil channel).",
    ),
]


# OUTPUT rules — run on the model's reply before it reaches the user
# These catch a successful injection by its symptoms: the model leaks its prompt,
# breaks the identity rule, or emits an exfiltration channel (the EchoLeak trick).

OUTPUT_RULES: list[Rule] = [
    Rule(
        name="confidential_data_leak",
        category="Sensitive data disclosure",
        severity="critical",
        owasp="LLM02",
        # Fingerprints of the demo's planted confidential block. Catches the leak
        # on the way out even if a payload slips the input scan (defence-in-depth).
        pattern=_rx(r"LVLUP-BETA-7731|admin override pin|\[CONFIDENTIAL|"
                    r"venkat@levelup\.internal|\bNemesis\b"),
        note="Reply contains planted confidential data — blocked before it reaches the user.",
    ),
    Rule(
        name="system_prompt_leak",
        category="System-prompt disclosure",
        severity="high",
        owasp="LLM02",
        # Fingerprints of our own prompt scaffolding leaking into the reply.
        pattern=_rx(r"\[RULES[\s\S]{0,10}never break|never break these|"
                    r"\[USER STATS|\[MODE:|\bYour catchphrase\b|"
                    r"the AI companion for LevelUp"),
        note="Reply contains fragments of the hidden system prompt / rule block.",
    ),
    Rule(
        name="identity_break",
        category="Identity / policy break",
        severity="medium",
        owasp="LLM01",
        pattern=_rx(r"\b(i am|i'm|as an?) (chatgpt|gpt-?4|gpt-?3|openai|"
                    r"large language model|ai language model|language model developed|"
                    r"claude|gemini|bard)\b"),
        note="Model dropped the LevelUp persona and disclosed a base-model identity.",
    ),
    Rule(
        name="markdown_image_exfil",
        category="Data exfiltration",
        severity="critical",
        owasp="LLM06",
        # The EchoLeak mechanism: an auto-loading markdown image whose URL carries
        # smuggled text. Rendering the reply fires a GET to the attacker's host.
        pattern=_rx(r"!\[[^\]]*\]\(\s*https?://[^)]+\)"),
        note="Auto-loading markdown image — the EchoLeak zero-click exfiltration channel.",
    ),
    Rule(
        name="external_link",
        category="Untrusted link",
        severity="low",
        owasp="LLM06",
        # Non-image markdown link to an external host — sanitised (defanged), not blocked.
        pattern=_rx(r"(?<!\!)\[[^\]]+\]\(\s*https?://[^)]+\)|https?://[^\s)]+"),
        note="Outbound link in the reply; defanged unless the host is allow-listed.",
    ),
]


# Hosts the model is allowed to link to without the link being defanged.
DEFAULT_ALLOWED_DOMAINS = frozenset({
    "levelup.ai",
    "localhost",
    "127.0.0.1",
})
