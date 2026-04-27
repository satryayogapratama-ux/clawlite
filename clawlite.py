#!/usr/bin/env python3
"""
ClawLite — Token-Efficient AI Routing for Claude (and any LLM)
=========================================================
3-layer optimization:
  1. Semantic Cache   — zero-token response reuse (exact + fuzzy match)
  2. Context Compressor — reduce conversation history before API call
  3. Model Router      — route to cheapest capable model automatically

Legal, transparent, provider-agnostic. Works with Anthropic, OpenAI, etc.

Usage:
    from clawlite import ClawLite

    cl = ClawLite(api_key="sk-ant-...")
    response = cl.chat("Apa itu BPJS?")
    response = cl.chat("Review arsitektur ini: ...")  # auto-routes to Opus

    cl.stats()   # show savings
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from router import ModelRouter, Complexity
from cache import SemanticCache, CacheResult
from compressor import ContextCompressor, Message
from analytics import Analytics


@dataclass
class ClawLiteResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    cache_hit: bool
    cache_type: str
    compressed: bool
    tokens_saved_cache: int
    tokens_saved_compress: int
    latency_ms: float
    cost_usd: float
    cost_saved_usd: float


class ClawLite:
    """
    Drop-in token optimizer for Claude API calls.

    Parameters
    ----------
    api_key         : Anthropic API key (or set ANTHROPIC_API_KEY env)
    medium_model    : Model for medium complexity (default: claude-sonnet-4-6)
    complex_model   : Model for complex queries (default: claude-opus-4-5)
    cache_db        : Path to cache SQLite DB
    analytics_path  : Path to analytics JSON
    max_tokens_ctx  : Max tokens to send as context (compressor budget)
    keep_recent     : How many recent messages to keep verbatim
    similarity_thr  : Cosine threshold for fuzzy cache hit (0.0-1.0)
    daily_budget    : Daily spend limit in USD (0 = unlimited)
    cache_ttl       : Default cache TTL in seconds (86400 = 24h)
    verbose         : Print routing decisions
    """

    def __init__(
        self,
        api_key: str = None,
        medium_model:  str = "claude-sonnet-4-6",
        complex_model: str = "claude-opus-4-5",
        cache_db:      str = "clawlite_cache.db",
        analytics_path: str = "clawlite_analytics.json",
        max_tokens_ctx: int = 4000,
        keep_recent:    int = 6,
        similarity_thr: float = 0.92,
        daily_budget:   float = 0.0,
        cache_ttl:      float = 86400,
        verbose:        bool = False,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.medium_model  = medium_model
        self.complex_model = complex_model
        self.cache_ttl = cache_ttl
        self.verbose = verbose

        # Components
        self.router = ModelRouter(
            medium_model=medium_model,
            complex_model=complex_model,
            force_medium=True,
        )
        self.cache = SemanticCache(
            db_path=cache_db,
            similarity_threshold=similarity_thr,
            default_ttl=cache_ttl,
        )
        self.compressor = ContextCompressor(
            max_tokens=max_tokens_ctx,
            keep_recent=keep_recent,
        )
        self.analytics = Analytics(
            state_path=analytics_path,
            daily_budget_usd=daily_budget,
        )

        self._conversation: list[Message] = []

    # ─── Public API ───────────────────────────────────────────────────────────

    def chat(
        self,
        query: str,
        system: str = None,
        force_model: str = None,
        use_cache: bool = True,
        ttl: float = None,
    ) -> ClawLiteResponse:
        """
        Send a message with full optimization pipeline.

        query       : User message
        system      : System prompt (optional)
        force_model : Override router (e.g. force_model="claude-opus-4-5")
        use_cache   : Set False to bypass cache for this request
        ttl         : Cache TTL override in seconds
        """
        t0 = time.time()

        # 1. Check cache first
        cache_result = self.cache.get(query) if use_cache else CacheResult(hit=False)

        if cache_result.hit:
            latency = (time.time() - t0) * 1000
            rec = self.analytics.record(
                query=query, model="cache",
                cache_hit=True, cache_type=cache_result.match_type,
                tokens_saved_cache=cache_result.tokens_saved,
            )
            if self.verbose:
                print(f"[ClawLite] CACHE {cache_result.match_type.upper()} hit "
                      f"(sim={cache_result.similarity:.3f}) — saved {cache_result.tokens_saved} tokens")
            return ClawLiteResponse(
                content=cache_result.response,
                model="cache",
                tokens_in=0, tokens_out=0,
                cache_hit=True, cache_type=cache_result.match_type,
                compressed=False,
                tokens_saved_cache=cache_result.tokens_saved,
                tokens_saved_compress=0,
                latency_ms=latency,
                cost_usd=0.0,
                cost_saved_usd=rec.cost_saved_usd,
            )

        # 2. Route to model
        decision = self.router.classify(query, conversation_length=len(self._conversation))
        model = force_model or decision.model

        if self.verbose:
            print(f"[ClawLite] ROUTE → {model} ({decision.complexity.value}, "
                  f"conf={decision.confidence:.2f}, reason: {decision.reason})")

        # 3. Budget check
        budget = self.analytics.budget_check(model, decision.estimated_tokens)
        if not budget["ok"]:
            if self.verbose:
                print(f"[ClawLite] BUDGET: {budget['reason']} — downgrading to {budget.get('suggest', model)}")
            model = budget.get("suggest", self.medium_model)

        # 4. Compress conversation context
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for m in self._conversation:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": query})

        compress_result = self.compressor.compress(messages)
        tokens_saved_compress = compress_result.original_tokens - compress_result.compressed_tokens

        if self.verbose and compress_result.summary_added:
            print(f"[ClawLite] COMPRESS: {compress_result.original_tokens} → "
                  f"{compress_result.compressed_tokens} tokens "
                  f"({compress_result.compression_ratio:.0%} ratio)")

        # 5. Call API
        api_messages = self.compressor.to_api_format(compress_result)
        response_text, tokens_in, tokens_out = self._call_api(api_messages, model)

        # 6. Store in cache + conversation
        if use_cache and response_text:
            self.cache.set(
                query=query,
                response=response_text,
                model=model,
                tokens_used=tokens_in + tokens_out,
                ttl=ttl or self.cache_ttl,
            )
        self._conversation.append(Message(role="user", content=query))
        self._conversation.append(Message(role="assistant", content=response_text))

        # 7. Record analytics
        latency = (time.time() - t0) * 1000
        rec = self.analytics.record(
            query=query, model=model,
            tokens_in=tokens_in, tokens_out=tokens_out,
            compressed=compress_result.summary_added,
            tokens_saved_compress=max(0, tokens_saved_compress),
        )

        return ClawLiteResponse(
            content=response_text,
            model=model,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_hit=False, cache_type="none",
            compressed=compress_result.summary_added,
            tokens_saved_cache=0,
            tokens_saved_compress=max(0, tokens_saved_compress),
            latency_ms=latency,
            cost_usd=rec.cost_usd,
            cost_saved_usd=rec.cost_saved_usd,
        )

    def reset_conversation(self):
        """Clear conversation history (start fresh)."""
        self._conversation.clear()

    def stats(self):
        """Print savings summary."""
        self.analytics.print_summary()
        cache_stats = self.cache.stats()
        print(f"  {'cache_entries':<25} {cache_stats['total_entries']}")
        print(f"  {'cache_total_hits':<25} {cache_stats['total_hits']}")
        print()

    def cache_stats(self) -> dict:
        return self.cache.stats()

    def router_stats(self) -> dict:
        return self.router.get_stats()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _call_api(self, messages: list[dict], model: str) -> tuple[str, int, int]:
        """Call Anthropic API. Returns (response_text, tokens_in, tokens_out)."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            # Separate system from messages
            system_content = None
            chat_messages = []
            for m in messages:
                if m["role"] == "system":
                    system_content = (system_content or "") + m["content"] + "\n"
                else:
                    chat_messages.append(m)

            kwargs = {
                "model": model,
                "max_tokens": 4096,
                "messages": chat_messages,
            }
            if system_content:
                kwargs["system"] = system_content.strip()

            resp = client.messages.create(**kwargs)
            text = resp.content[0].text if resp.content else ""
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        except ImportError:
            raise ImportError("pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e
