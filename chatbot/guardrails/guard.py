"""
The LevelUp guardrail — a runtime "AI firewall" wrapped around the fine-tuned
LLaMA 3.2 chatbot. It inspects what goes INTO the model (user message, history,
ingested user_context) and what comes BACK OUT (the reply), and blocks or
sanitises prompt-injection, jailbreaks, system-prompt leakage and exfiltration.

Why this exists (the thesis link):
  The model flattens the system prompt, the few-shot examples, the user message
  and any retrieved context into ONE token stream and cannot reliably tell
  trusted instructions from untrusted data. This guardrail enforces that
  separation from the outside — it is an independent second control, i.e.
  defence-in-depth, not a replacement for the model's own safety training.

Design goals:
  - Explainable  : every decision lists the exact rule(s) that fired + OWASP id.
  - Offline      : pure-Python regex heuristics, no API keys, runs on the laptop.
  - Fail-safe    : an unexpected error in the guardrail never crashes the chat;
                   input checks fail OPEN (chat still answers), output leaks are
                   sanitised. This trade-off is documented and demo-visible.

Usage:
    from chatbot.guardrails import Guardrail

    guard = Guardrail()

    verdict = guard.check_input(message, history=history, user_context=ctx)
    if verdict.blocked:
        return guard.safe_refusal(build)          # don't call the model at all

    reply = model.generate(...)

    out = guard.check_output(reply)
    reply = out.safe_text                          # sanitised / replaced if needed
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .patterns import (
    DEFAULT_ALLOWED_DOMAINS,
    INPUT_RULES,
    OUTPUT_RULES,
    SEVERITY_ORDER,
    Rule,
)

# Result types

@dataclass
class Finding:
    """One rule that matched, with enough context to explain the decision."""
    rule:     str    # rule name, e.g. "instruction_override"
    category: str
    severity: str
    owasp:    str
    location: str    # where it matched: user_message | history | user_context | output
    snippet:  str    # the matched text (truncated) — evidence for the slide/report
    note:     str

    def as_dict(self) -> dict:
        return {
            "rule": self.rule, "category": self.category, "severity": self.severity,
            "owasp": self.owasp, "location": self.location,
            "snippet": self.snippet, "note": self.note,
        }


@dataclass
class Verdict:
    """The outcome of an input or output scan."""
    action:    str                    # "allow" | "sanitize" | "block"
    findings:  list[Finding] = field(default_factory=list)
    safe_text: str | None = None   # output scans: the text safe to return
    reason:    str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def max_severity(self) -> str | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=lambda s: SEVERITY_ORDER[s])

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "reason": self.reason,
            "max_severity": self.max_severity,
            "findings": [f.as_dict() for f in self.findings],
        }


# Guardrail

_SNIPPET_MAX = 120


class Guardrail:
    """Input/output firewall around the LevelUp chatbot."""

    def __init__(
        self,
        block_threshold: str = "high",
        allowed_domains=DEFAULT_ALLOWED_DOMAINS,
    ):
        """
        block_threshold : minimum severity that causes an INPUT to be blocked
                          ("medium" is stricter, "critical" is more permissive).
        allowed_domains : hosts that may appear in replies without being defanged.
        """
        self.block_level      = SEVERITY_ORDER[block_threshold]
        self.allowed_domains  = set(allowed_domains)

    # helpers

    @staticmethod
    def _snippet(match: re.Match) -> str:
        s = match.group(0).strip().replace("\n", " ")
        return s[:_SNIPPET_MAX] + ("…" if len(s) > _SNIPPET_MAX else "")

    def _scan(self, text: str, rules: list[Rule], location: str) -> list[Finding]:
        if not text:
            return []
        findings: list[Finding] = []
        for rule in rules:
            m = rule.pattern.search(text)
            if m:
                findings.append(Finding(
                    rule=rule.name, category=rule.category, severity=rule.severity,
                    owasp=rule.owasp, location=location,
                    snippet=self._snippet(m), note=rule.note,
                ))
        return findings

    # INPUT scanning

    def check_input(
        self,
        message: str,
        history: list[dict] | None = None,
        user_context: dict | None = None,
    ) -> Verdict:
        """
        Scan everything that will be concatenated into the prompt: the fresh
        user message, prior turns, and any ingested user_context strings (the
        seam where trusted and untrusted text meet).
        """
        findings: list[Finding] = []
        try:
            findings += self._scan(message, INPUT_RULES, "user_message")

            for turn in (history or []):
                # Only user-authored turns are untrusted input; skip the model's
                # own prior replies (those are covered by output scanning).
                if turn.get("role") == "user":
                    findings += self._scan(turn.get("content", ""), INPUT_RULES, "history")

            for val in self._iter_context_strings(user_context or {}):
                findings += self._scan(val, INPUT_RULES, "user_context")
        except Exception as e:  # guardrail must never crash the chat
            return Verdict(action="allow", reason=f"guardrail error (fail-open): {e}")

        if not findings:
            return Verdict(action="allow", reason="clean")

        worst = max(SEVERITY_ORDER[f.severity] for f in findings)
        if worst >= self.block_level:
            cats = ", ".join(sorted({f.category for f in findings}))
            return Verdict(action="block", findings=findings,
                           reason=f"Blocked input: {cats}")
        return Verdict(action="allow", findings=findings,
                       reason="Flagged (below block threshold) — allowed with warning")

    @staticmethod
    def _iter_context_strings(ctx: dict):
        """Yield free-text string values from user_context that could carry an
        injected instruction (e.g. journal notes, custom activity names). Numeric
        stats are ignored — only text is an injection surface."""
        for _key, val in ctx.items():
            if isinstance(val, str) and len(val) > 3:
                yield val
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        yield item
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str) and len(v) > 3:
                                yield v

    # OUTPUT scanning

    def check_output(self, text: str) -> Verdict:
        """
        Scan the model's reply. Critical exfiltration channels (auto-loading
        markdown images) and system-prompt leaks are neutralised; external links
        are defanged. The returned safe_text is always safe to send to the user.
        """
        try:
            findings = self._scan(text, OUTPUT_RULES, "output")
        except Exception as e:
            return Verdict(action="allow", safe_text=text,
                           reason=f"guardrail error (fail-open): {e}")

        if not findings:
            return Verdict(action="allow", safe_text=text, reason="clean")

        safe = text
        # Planted/sensitive data leaked into the reply — never send it onward.
        if any(f.rule == "confidential_data_leak" for f in findings):
            safe = ("I can't share internal or confidential configuration. "
                    "Let's get back to your build — what do you want to work on?")
            return Verdict(action="block", findings=findings, safe_text=safe,
                           reason="Blocked output: sensitive data disclosure")

        # Strip auto-loading images entirely (EchoLeak-style zero-click exfil).
        if any(f.rule == "markdown_image_exfil" for f in findings):
            safe = re.sub(r"!\[[^\]]*\]\(\s*https?://[^)]+\)", "[image removed by guardrail]", safe)

        # Replace a leaked system prompt with a refusal — never echo internals.
        if any(f.rule == "system_prompt_leak" for f in findings):
            safe = ("I can't share my internal instructions, but I'm here to help "
                    "with your build. What are you working on?")

        # Defang external, non-allow-listed links so nothing auto-resolves.
        safe = self._defang_links(safe)

        cats = ", ".join(sorted({f.category for f in findings}))
        return Verdict(action="sanitize", findings=findings, safe_text=safe,
                       reason=f"Sanitised output: {cats}")

    def _defang_links(self, text: str) -> str:
        def _host_allowed(url: str) -> bool:
            host = (urlparse(url).hostname or "").lower()
            return any(host == d or host.endswith("." + d) for d in self.allowed_domains)

        def _repl(m: re.Match) -> str:
            url = m.group(0)
            if _host_allowed(url):
                return url
            return url.replace("http://", "hxxp://").replace("https://", "hxxps://")

        return re.sub(r"https?://[^\s)\]]+", _repl, text)

    # Canned safe reply when input is blocked

    def safe_refusal(self, build: str = "") -> str:
        """A short, in-app refusal used when an input is blocked, so the chat
        still returns something instead of erroring."""
        return ("That message looks like it's trying to change how I work or pull "
                "out my instructions, so I didn't run it. Ask me about your build, "
                "your XP, or your next quest and I've got you.")
