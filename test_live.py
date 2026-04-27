#!/usr/bin/env python3
"""
ClawLite Live Test — Real Anthropic API, real routing, real savings.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from clawlite import ClawLite

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def sep(n=56): print("─" * n)

print()
print("  ClawLite — LIVE TEST (Real API)")
sep()

cl = ClawLite(
    api_key=API_KEY,
    medium_model="claude-sonnet-4-6",
    complex_model="claude-opus-4-5",
    cache_db="/tmp/clawlite_live_cache.db",
    analytics_path="/tmp/clawlite_live_analytics.json",
    similarity_thr=0.82,
    daily_budget=0.10,
    verbose=True,
)

# ── Test 1: Sonnet query ──────────────────────────────────────────────────────
print("\n[1/4] Medium query → should route to Sonnet")
sep()
r = cl.chat("Apa itu Docker container? Jelaskan singkat.")
print(f"\nResponse ({r.tokens_in}in/{r.tokens_out}out | {r.latency_ms:.0f}ms | ${r.cost_usd:.4f}):")
print(r.content[:200])

# ── Test 2: Cache hit (exact) ─────────────────────────────────────────────────
print("\n\n[2/4] Exact same query → should CACHE HIT, 0 tokens, <5ms")
sep()
r2 = cl.chat("Apa itu Docker container? Jelaskan singkat.")
print(f"\nCache: {r2.cache_type.upper()} | Tokens saved: {r2.tokens_saved_cache} | {r2.latency_ms:.1f}ms")
assert r2.cache_hit, "Expected cache hit!"
assert r2.tokens_in == 0, "Expected 0 tokens!"
print("✅ Cache hit confirmed")

# ── Test 3: Semantic cache hit ────────────────────────────────────────────────
print("\n\n[3/4] Rephrased query → should SEMANTIC CACHE HIT")
sep()
r3 = cl.chat("Apa itu container Docker? Tolong jelaskan.")
print(f"\nCache: {r3.cache_type.upper()} | Sim: N/A in response | {r3.latency_ms:.1f}ms")
if r3.cache_hit:
    print(f"✅ Semantic hit! Saved {r3.tokens_saved_cache} tokens")
else:
    print("⚠️  Semantic miss (below threshold — still a valid result)")

# ── Test 4: Opus routing ──────────────────────────────────────────────────────
print("\n\n[4/4] Complex query → should route to OPUS")
sep()
r4 = cl.chat(
    "Review arsitektur sistem ERP pabrik rokok secara komprehensif. "
    "Analisis security, scalability, dan berikan rekomendasi improvement.",
    use_cache=False,  # don't cache this, too specific
)
print(f"\nModel: {r4.model} | {r4.tokens_in}in/{r4.tokens_out}out | ${r4.cost_usd:.4f} | {r4.latency_ms:.0f}ms")
print(r4.content[:200])

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n")
cl.stats()
