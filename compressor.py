"""
ClawLite Context Compressor — Reduce token cost for long conversations.
Keeps recent messages verbatim, compresses older ones to key facts.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str    # "user" | "assistant" | "system"
    content: str

    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4)


@dataclass
class CompressResult:
    messages: list[Message]
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float   # 0.0 - 1.0 (lower = more compressed)
    summary_added: bool


def _token_count(messages: list[Message]) -> int:
    return sum(m.token_estimate() for m in messages)


def _extract_key_facts(text: str, max_chars: int = 300) -> str:
    """
    Heuristic fact extraction — pulls sentences with key signals.
    No LLM call needed.
    """
    KEY_SIGNALS = [
        r'\b(decided|agreed|confirmed|kesepakatan|diputuskan|setuju)\b',
        r'\b(deadline|due|tanggal|jam|schedule)\b',
        r'\b(important|penting|critical|kritis|must|harus)\b',
        r'\b(error|bug|fixed|solved|selesai|done|✅|❌)\b',
        r'\b(result|output|hasilnya|kesimpulan|conclusion)\b',
        r'\b(use|pakai|gunakan|implement|install|deploy)\b',
        r'https?://\S+',  # URLs are important
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IPs
    ]

    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    scored = []
    for sent in sentences:
        if len(sent.strip()) < 10:
            continue
        score = sum(2 if re.search(pat, sent, re.I) else 0 for pat in KEY_SIGNALS)
        # Long sentences with no signals still worth keeping if they look factual
        if len(sent) > 60 and score == 0:
            score = 1
        scored.append((score, sent.strip()))

    scored.sort(key=lambda x: -x[0])
    result = []
    total = 0
    for _, sent in scored:
        if total + len(sent) > max_chars:
            break
        result.append(sent)
        total += len(sent)

    return " | ".join(result[:5]) if result else text[:max_chars]


class ContextCompressor:
    """
    Compress conversation context before sending to API.

    Strategy:
    - Keep last `keep_recent` messages verbatim (most relevant)
    - Compress older messages into a rolling summary
    - Always keep system message intact
    - Budget: target total <= `max_tokens`
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        keep_recent: int = 6,
        summary_max_chars: int = 800,
    ):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.summary_max_chars = summary_max_chars

    def compress(self, messages: list[dict | Message]) -> CompressResult:
        """
        Compress a list of messages dicts or Message objects.
        Input format: [{"role": "user", "content": "..."}, ...]
        """
        # Normalize to Message objects
        msgs = []
        for m in messages:
            if isinstance(m, dict):
                msgs.append(Message(role=m["role"], content=m["content"]))
            else:
                msgs.append(m)

        original_tokens = _token_count(msgs)

        # Always keep system message
        system_msgs = [m for m in msgs if m.role == "system"]
        non_system  = [m for m in msgs if m.role != "system"]

        # If already within budget, return as-is
        if original_tokens <= self.max_tokens:
            return CompressResult(
                messages=msgs,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                summary_added=False,
            )

        # Split: keep recent messages within token budget, compress the rest
        # Cap by tokens, not message count — one long message can blow the budget
        recent = []
        recent_tokens = 0
        for m in reversed(non_system):
            mt = m.token_estimate()
            if recent_tokens + mt <= self.max_tokens * 0.6:  # 60% budget for recent
                recent.insert(0, m)
                recent_tokens += mt
            elif len(recent) < 2:  # always keep at least last 2
                recent.insert(0, m)
                recent_tokens += mt
            else:
                break
        old = [m for m in non_system if m not in recent]

        # Compress old messages into summary
        summary_parts = []
        for m in old:
            role_label = "User" if m.role == "user" else "AI"
            fact = _extract_key_facts(m.content, max_chars=self.summary_max_chars // max(1, len(old)))
            summary_parts.append(f"[{role_label}]: {fact}")

        summary_text = "\n".join(summary_parts)
        if len(summary_text) > self.summary_max_chars:
            summary_text = summary_text[:self.summary_max_chars] + "..."

        summary_msg = Message(
            role="system",
            content=f"[Compressed context — {len(old)} earlier messages]\n{summary_text}"
        ) if summary_parts else None

        # Assemble final message list
        final = system_msgs[:]
        if summary_msg:
            final.append(summary_msg)
        final.extend(recent)

        compressed_tokens = _token_count(final)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return CompressResult(
            messages=final,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            summary_added=bool(summary_parts),
        )

    def to_api_format(self, result: CompressResult) -> list[dict]:
        """Convert CompressResult back to API-ready list of dicts."""
        return [{"role": m.role, "content": m.content} for m in result.messages]
