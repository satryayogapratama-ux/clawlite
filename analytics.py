"""
ClawLite Analytics — Real-time savings tracking with budget alerts.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


# Model pricing per 1M tokens (input + output blended estimate)
MODEL_PRICING = {
    "haiku":  0.50,   # claude-haiku-4-5 blended
    "sonnet": 3.00,   # claude-sonnet-4-6 blended
    "opus":   15.00,  # claude-opus-4-5 blended
    # Aliases
    "claude-haiku-4-5": 0.50,
    "claude-sonnet-4-6": 3.00,
    "claude-opus-4-5": 15.00,
}

DEFAULT_MODEL = "sonnet"


@dataclass
class RequestRecord:
    timestamp: float
    query_preview: str     # first 60 chars
    model: str
    tokens_in: int
    tokens_out: int
    cache_hit: bool
    cache_type: str        # "exact", "semantic", "none"
    compressed: bool
    tokens_saved_cache: int
    tokens_saved_compress: int
    cost_usd: float
    cost_saved_usd: float


class Analytics:
    """Track token usage, savings, and budget across sessions."""

    def __init__(
        self,
        state_path: str = "clawlite_analytics.json",
        daily_budget_usd: float = 0.0,   # 0 = no budget limit
    ):
        self.state_path = Path(state_path)
        self.daily_budget_usd = daily_budget_usd
        self._records: list[RequestRecord] = []
        self._load()

    def _load(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self._totals = data.get("totals", self._default_totals())
            except Exception:
                self._totals = self._default_totals()
        else:
            self._totals = self._default_totals()

    def _default_totals(self) -> dict:
        return {
            "total_requests": 0,
            "cache_hits": 0,
            "tokens_used": 0,
            "tokens_saved_cache": 0,
            "tokens_saved_compress": 0,
            "cost_usd": 0.0,
            "cost_saved_usd": 0.0,
            "session_start": time.time(),
        }

    def _save(self):
        self.state_path.write_text(json.dumps({
            "totals": self._totals,
            "last_updated": time.time(),
        }, indent=2))

    def token_cost(self, tokens: int, model: str) -> float:
        price = MODEL_PRICING.get(model.lower(), MODEL_PRICING[DEFAULT_MODEL])
        return (tokens / 1_000_000) * price

    def record(
        self,
        query: str,
        model: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_hit: bool = False,
        cache_type: str = "none",
        compressed: bool = False,
        tokens_saved_cache: int = 0,
        tokens_saved_compress: int = 0,
    ) -> RequestRecord:
        total_tokens = tokens_in + tokens_out
        cost = 0.0 if cache_hit else self.token_cost(total_tokens, model)
        cost_saved = (
            self.token_cost(tokens_saved_cache, model) +
            self.token_cost(tokens_saved_compress, model)
        )

        rec = RequestRecord(
            timestamp=time.time(),
            query_preview=query[:60],
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_hit=cache_hit,
            cache_type=cache_type,
            compressed=compressed,
            tokens_saved_cache=tokens_saved_cache,
            tokens_saved_compress=tokens_saved_compress,
            cost_usd=cost,
            cost_saved_usd=cost_saved,
        )

        self._records.append(rec)
        self._totals["total_requests"] += 1
        if cache_hit:
            self._totals["cache_hits"] += 1
        self._totals["tokens_used"] += total_tokens
        self._totals["tokens_saved_cache"] += tokens_saved_cache
        self._totals["tokens_saved_compress"] += tokens_saved_compress
        self._totals["cost_usd"] += cost
        self._totals["cost_saved_usd"] += cost_saved
        self._save()

        return rec

    def budget_check(self, model: str, estimated_tokens: int) -> dict:
        """Check if request fits within daily budget."""
        if self.daily_budget_usd <= 0:
            return {"ok": True, "reason": "no budget set"}

        today_cost = self._totals["cost_usd"]
        estimated_cost = self.token_cost(estimated_tokens, model)

        if today_cost + estimated_cost > self.daily_budget_usd:
            return {
                "ok": False,
                "reason": f"Budget exceeded: ${today_cost:.4f} used of ${self.daily_budget_usd:.2f}",
                "suggest": "haiku",  # suggest cheapest model
            }
        return {"ok": True, "remaining": self.daily_budget_usd - today_cost}

    def summary(self) -> dict:
        t = self._totals
        total_req = max(1, t["total_requests"])
        cache_rate = t["cache_hits"] / total_req * 100
        total_saved = t["tokens_saved_cache"] + t["tokens_saved_compress"]
        effective_cost = t["cost_usd"]
        would_have_cost = effective_cost + t["cost_saved_usd"]
        saving_pct = (t["cost_saved_usd"] / would_have_cost * 100) if would_have_cost > 0 else 0

        return {
            "total_requests": total_req,
            "cache_hit_rate": f"{cache_rate:.1f}%",
            "tokens_used": t["tokens_used"],
            "tokens_saved_total": total_saved,
            "cost_usd": f"${effective_cost:.4f}",
            "cost_saved_usd": f"${t['cost_saved_usd']:.4f}",
            "saving_percentage": f"{saving_pct:.1f}%",
            "daily_budget": f"${self.daily_budget_usd:.2f}" if self.daily_budget_usd > 0 else "none",
        }

    def print_summary(self):
        s = self.summary()
        print("\n" + "─" * 50)
        print("  ClawLite Analytics")
        print("─" * 50)
        for k, v in s.items():
            print(f"  {k:<25} {v}")
        print("─" * 50 + "\n")
