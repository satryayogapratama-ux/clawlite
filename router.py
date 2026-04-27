"""
ClawLite Router — Zero-cost query classification for smart model routing.
Classifies queries using heuristics + keyword rules (no API call needed).
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Complexity(Enum):
    SIMPLE   = "simple"    # Haiku / fast model
    MEDIUM   = "medium"    # Sonnet
    COMPLEX  = "complex"   # Opus


@dataclass
class RouteDecision:
    complexity: Complexity
    model: str
    confidence: float   # 0.0 - 1.0
    reason: str
    estimated_tokens: int


# ─── Signal Patterns ──────────────────────────────────────────────────────────

SIMPLE_PATTERNS = [
    r'\b(what time|what day|hari ini|jam berapa|tanggal|cuaca|weather)\b',
    r'\b(hello|hi|halo|hai|thanks|thank you|terima kasih|oke|ok|yes|no|ya|tidak)\b',
    r'\b(define|arti|artinya|what is|apa itu|apa arti)\b.{0,30}$',
    r'\b(how many|berapa|count|total)\b.{0,40}$',
    r'\b(translate|terjemahkan|translate to)\b',
]

COMPLEX_PATTERNS = [
    r'\b(architecture|arsitektur|design pattern|system design)\b',
    r'\b(review|audit|crosscheck|cros cek|second opinion)\b',
    r'\b(debug|fix all|refactor|rewrite|redesign)\b.{30,}',
    r'\b(analyze|analisis|evaluate|compare|pros and cons)\b.{40,}',
    r'(why|mengapa|kenapa).{60,}',
    r'\b(legal|contract|NDA|compliance|regulation|hukum)\b',
    r'\b(financial model|forecast|valuation|due diligence)\b',
    r'\b(machine learning|neural network|deep learning|training|model accuracy)\b',
]

CODE_SIGNALS = [
    r'```[\s\S]{200,}```',      # large code blocks
    r'\b(implement|build|create|buat|bangun).{0,20}(system|api|backend|frontend|database)\b',
    r'\b(class|function|def |async |await |import |require)\b',
]

OPUS_KEYWORDS = {
    'architecture', 'arsitektur', 'review', 'audit', 'crosscheck', 'cros cek',
    'legal', 'contract', 'NDA', 'compliance', 'valuation', 'machine learning',
    'second opinion', 'deep analysis', 'comprehensive', 'enterprise',
}

# ─── Router Class ─────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Route queries to the cheapest capable model.
    Supports any model mapping — default: Sonnet + Opus.
    """

    def __init__(
        self,
        simple_model:  str = "haiku",
        medium_model:  str = "sonnet",
        complex_model: str = "opus",
        force_medium:  bool = True,   # skip haiku, use sonnet as minimum
    ):
        self.models = {
            Complexity.SIMPLE:  simple_model if not force_medium else medium_model,
            Complexity.MEDIUM:  medium_model,
            Complexity.COMPLEX: complex_model,
        }
        self.force_medium = force_medium
        self.routing_history: list[RouteDecision] = []

    def classify(self, query: str, conversation_length: int = 0) -> RouteDecision:
        """Classify query complexity without any API call."""
        q = query.strip()
        q_lower = q.lower()
        word_count = len(q.split())

        score = 0  # higher = more complex
        reasons = []

        # Length signals
        if word_count < 8:
            score -= 2
            reasons.append("short query")
        elif word_count > 80:
            score += 2
            reasons.append("long query")

        # Simple patterns
        for pat in SIMPLE_PATTERNS:
            if re.search(pat, q_lower):
                score -= 2
                reasons.append("simple pattern match")
                break

        # Complex patterns
        for pat in COMPLEX_PATTERNS:
            if re.search(pat, q_lower):
                score += 3
                reasons.append("complex pattern match")
                break

        # Code signals
        code_hits = sum(1 for pat in CODE_SIGNALS if re.search(pat, q))
        if code_hits:
            score += code_hits * 1.5
            reasons.append(f"code signals ({code_hits})")

        # Opus keywords
        opus_hits = sum(1 for kw in OPUS_KEYWORDS if kw in q_lower)
        if opus_hits >= 2:
            score += 4
            reasons.append(f"opus keywords ({opus_hits})")
        elif opus_hits == 1:
            score += 2

        # Conversation depth (longer = harder to answer without full context)
        if conversation_length > 10:
            score += 1
            reasons.append("deep conversation")

        # Questions with multiple clauses
        clause_count = len(re.findall(r'[,;]|\band\b|\bdan\b|\balso\b|\bjuga\b', q_lower))
        if clause_count >= 3:
            score += 1
            reasons.append("multi-clause")

        # Decide
        if score >= 5:
            complexity = Complexity.COMPLEX
            confidence = min(0.95, 0.7 + (score - 5) * 0.05)
        elif score >= 0:
            complexity = Complexity.MEDIUM
            confidence = 0.75
        else:
            complexity = Complexity.SIMPLE
            confidence = min(0.95, 0.7 + abs(score) * 0.1)

        # Estimate tokens (rough: 4 chars ≈ 1 token)
        estimated_tokens = max(100, len(q) // 4 + 200)

        decision = RouteDecision(
            complexity=complexity,
            model=self.models[complexity],
            confidence=confidence,
            reason=" | ".join(reasons) if reasons else "default",
            estimated_tokens=estimated_tokens,
        )

        self.routing_history.append(decision)
        return decision

    def get_stats(self) -> dict:
        if not self.routing_history:
            return {}
        dist = {}
        for d in self.routing_history:
            dist[d.complexity.value] = dist.get(d.complexity.value, 0) + 1
        return {
            "total_routed": len(self.routing_history),
            "distribution": dist,
            "avg_confidence": sum(d.confidence for d in self.routing_history) / len(self.routing_history),
        }
