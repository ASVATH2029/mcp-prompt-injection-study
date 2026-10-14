"""Regex/heuristic prompt-injection detector.

This is the comparison baseline. It replicates the pattern categories used by real
MCP guard tools (MCPDome, Shrike MCP, agent-safety-mcp, houtini prompt-injection-guard,
OWASP MCP06:2025). The design goal is a *credible* rule set, not a strawman: it covers
instruction-override phrasing, fake system markup, base64 payloads (decoded and
re-scanned), persona hijacking, Unicode obfuscation, and markup-hidden injections.

The entire rule set is an inspectable module-level constant (RULES). By construction it
only matches KNOWN surface forms, which is exactly the limitation this study measures:
paraphrased variants with different wording are expected to slip past it.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

# --- Pattern categories -----------------------------------------------------------

# 1. Instruction override: the attacker tells the model to drop prior instructions.
INSTRUCTION_OVERRIDE = [
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|earlier|above)\s+(?:instructions?|directions?|prompts?|context)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|earlier|above|your)",
    r"forget\s+(?:all\s+|everything\s+|your\s+)?(?:previous|prior|the\s+above|instructions?|what)",
    r"override\s+(?:your\s+|the\s+)?(?:previous\s+)?(?:instructions?|system\s+prompt|directives?)",
    r"new\s+(?:task|directive|instruction|objective|mission)\s*:",
    r"(?:you\s+are\s+now|from\s+now\s+on\s+you\s+(?:are|will|must))",
    r"do\s+not\s+(?:follow|obey)\s+(?:the\s+)?(?:previous|prior|earlier|above)",
]

# 2. Fake system / role markup injected into tool output.
FAKE_SYSTEM_TAG = [
    r"<\s*/?\s*system\s*>",
    r"\[\s*system\s*\]",
    r"<<\s*SYS\s*>>",
    r"#{2,}\s*system",
    r"<\s*/?\s*(?:assistant|user)\s*>",
    r"\bsystem\s+prompt\s*(?:override|:)",
]

# 4. Persona / role hijacking (DAN-style jailbreaks).
PERSONA_HIJACK = [
    r"pretend\s+(?:you\s+are|to\s+be|that\s+you)",
    r"act\s+as\s+(?:a\s+|an\s+|if\s+)?",
    r"your\s+new\s+(?:persona|role|identity|character)",
    r"roleplay\s+as",
    r"\bact\s+like\s+(?:a\s+|an\s+|you)",
    r"\b(?:you\s+are|become)\s+DAN\b",
    r"enter\s+(?:developer|jailbreak|unrestricted)\s+mode",
]

# 6. Markup-hidden injection: payloads tucked into HTML comments or code fences.
MARKDOWN_ESCAPE = [
    r"<!--[^>]*?(?:ignore|system|override|disregard|instruction)[^>]*?-->",
    r"```[^`]*?(?:ignore\s+previous|system\s+prompt\s+override|new\s+directive)[^`]*?```",
    r"\[//\]:\s*#\s*\(.*?(?:ignore|override|system).*?\)",
]

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "instruction_override": INSTRUCTION_OVERRIDE,
    "fake_system_tag": FAKE_SYSTEM_TAG,
    "persona_hijack": PERSONA_HIJACK,
    "markdown_escape": MARKDOWN_ESCAPE,
}

# Compiled once at import. Case-insensitive, DOTALL so markup rules span newlines.
_COMPILED: dict[str, list[re.Pattern[str]]] = {
    category: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
    for category, patterns in CATEGORY_PATTERNS.items()
}

# 5. Unicode obfuscation: zero-width chars and bidi overrides used to smuggle text.
_SUSPICIOUS_UNICODE = {
    "‮": "rtl_override",
    "‭": "ltr_override",
    "​": "zero_width_space",
    "‌": "zero_width_non_joiner",
    "‍": "zero_width_joiner",
    "﻿": "zero_width_no_break_space",
    "⁠": "word_joiner",
}

# 3. Base64 payload detection threshold (min length of a base64-looking run).
_BASE64_MIN_LEN = 24
_BASE64_RE = re.compile(rf"[A-Za-z0-9+/]{{{_BASE64_MIN_LEN},}}={{0,2}}")


@dataclass
class DetectorResult:
    """Verdict from a detector. `label` is 1 (injection) or 0 (benign)."""

    label: int
    confidence: float
    matched_rules: list[str] = field(default_factory=list)


def _scan_patterns(text: str) -> list[str]:
    """Return names of all category rules that fire on `text`."""
    hits: list[str] = []
    for category, patterns in _COMPILED.items():
        for pat in patterns:
            if pat.search(text):
                hits.append(category)
                break  # one hit per category is enough to flag it
    return hits


def _scan_unicode(text: str) -> list[str]:
    """Flag suspicious invisible/bidi characters."""
    hits: list[str] = []
    for ch, name in _SUSPICIOUS_UNICODE.items():
        if ch in text:
            hits.append(f"unicode_trick:{name}")
    return hits


def _find_base64_payloads(text: str) -> list[str]:
    """Decode base64-looking runs and re-scan the plaintext for injection patterns.

    Real guards do this because attackers base64-encode an override phrase to dodge
    literal keyword matching. We only flag a payload if the *decoded* content itself
    trips a pattern rule, which keeps false positives on ordinary base64 (hashes,
    tokens) low.
    """
    hits: list[str] = []
    for match in _BASE64_RE.finditer(text):
        candidate = match.group(0)
        # Pad to a multiple of 4 so partial runs still decode.
        padded = candidate + "=" * (-len(candidate) % 4)
        try:
            decoded = base64.b64decode(padded, validate=True).decode("utf-8", errors="strict")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if _scan_patterns(decoded) or _scan_unicode(decoded):
            hits.append("base64_payload")
            break
    return hits


class BaselineDetector:
    """Transparent regex/heuristic injection detector.

    Usage:
        det = BaselineDetector()
        result = det.predict("ignore all previous instructions")
        # result.label == 1, result.matched_rules == ["instruction_override"]
    """

    def predict(self, text: str) -> DetectorResult:
        if not text:
            return DetectorResult(label=0, confidence=0.0)

        normalized = unicodedata.normalize("NFKC", text)

        matched: list[str] = []
        matched += _scan_patterns(text)
        # Re-scan NFKC-normalized text to catch homoglyph/compatibility obfuscation.
        if normalized != text:
            for cat in _scan_patterns(normalized):
                if cat not in matched:
                    matched.append(cat)
        matched += _scan_unicode(text)
        matched += _find_base64_payloads(text)

        # Dedupe while preserving order.
        seen: set[str] = set()
        matched = [m for m in matched if not (m in seen or seen.add(m))]

        label = 1 if matched else 0
        # Heuristic confidence: more independent categories firing => higher confidence.
        confidence = min(1.0, 0.6 + 0.2 * (len(matched) - 1)) if matched else 0.0
        return DetectorResult(label=label, confidence=confidence, matched_rules=matched)
