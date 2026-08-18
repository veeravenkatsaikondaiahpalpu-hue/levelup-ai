"""
chatbot.guardrails
==================
A self-built runtime guardrail ("AI firewall") for the LevelUp chatbot.

Public API:
    from chatbot.guardrails import Guardrail, Verdict, Finding

    guard   = Guardrail()
    verdict = guard.check_input(message, history=..., user_context=...)
    out     = guard.check_output(reply)

See guard.py for the design rationale and patterns.py for the rule library.
"""

from .guard import Finding, Guardrail, Verdict
from .patterns import INPUT_RULES, OUTPUT_RULES, Rule

__all__ = ["Guardrail", "Verdict", "Finding", "Rule", "INPUT_RULES", "OUTPUT_RULES"]
