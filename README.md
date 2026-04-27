# ClawLite

Token-efficient AI routing for Claude and any LLM. Legally reduce your API bill by 35-80% without degrading response quality.

No API bypass. No jailbreaks. Pure engineering efficiency.

---

## The Problem

Most AI applications send full conversation history on every request and use the most expensive model for everything — regardless of whether the query needs it. This wastes 60-80% of API spend.

## The Solution

ClawLite sits between your code and the API. Three layers of optimization run transparently:

```
Your code → [Cache] → [Compressor] → [Router] → Claude API
                ↑              ↑            ↑
           0 tokens      80% less      right model
           on hit        context        for task
```

---

## Three Optimization Layers

### 1. Semantic Cache
Two-tier cache that serves repeated and rephrased queries without any API call.

- **Exact match** via MD5 hash — O(1), sub-2ms response
- **Fuzzy match** via cosine similarity — catches paraphrased queries (threshold: 0.82)
- Configurable TTL per entry (default: 24h)
- LRU eviction at 10,000 entries
- SQLite storage — no external dependencies

```python
# First call: hits API
response = cl.chat("What is BPJS Ketenagakerjaan?")  # 400 tokens

# Second call: exact cache hit — 0 tokens, 1.7ms
response = cl.chat("What is BPJS Ketenagakerjaan?")

# Third call: semantic cache hit — 0 tokens, 15ms
response = cl.chat("Explain BPJS for employees?")    # sim=0.83
```

### 2. Context Compressor
Reduces conversation history before each API call using heuristic fact extraction.

- Keeps last N messages verbatim (default: 6)
- Compresses older messages into a factual summary
- No LLM summarizer needed — zero extra API cost
- Configurable token budget (default: 4,000 tokens)

```
Long session: 50,000 tokens → 3,200 tokens sent to API
Saving: ~$0.14 per call at Sonnet pricing
```

### 3. Model Router
Classifies query complexity using regex patterns and heuristics — zero API cost.

| Complexity | Model | When |
|-----------|-------|------|
| Simple | Sonnet (or Haiku if enabled) | Short queries, greetings, translations |
| Medium | Sonnet | Code help, moderate analysis |
| Complex | Opus | Architecture review, legal, ML analysis |

```python
"Halo!"                    → Sonnet  (simple)
"Implement JWT auth system" → Sonnet  (medium)
"Review arsitektur komprehensif dan audit security" → Opus (complex)
```

---

## Quick Start

```bash
pip install anthropic sentence-transformers numpy
```

```python
from clawlite import ClawLite

cl = ClawLite(
    api_key="sk-ant-...",
    medium_model="claude-sonnet-4-6",
    complex_model="claude-opus-4-5",
    daily_budget=1.00,   # USD; 0 = unlimited
    verbose=True,
)

response = cl.chat("What is machine learning?")
print(response.content)
print(f"Model: {response.model} | Tokens: {response.tokens_in}in/{response.tokens_out}out")
print(f"Cache: {response.cache_hit} | Cost: ${response.cost_usd:.4f}")

cl.stats()
```

---

## Configuration

```python
ClawLite(
    api_key         = "sk-ant-...",       # or ANTHROPIC_API_KEY env var
    medium_model    = "claude-sonnet-4-6",
    complex_model   = "claude-opus-4-5",
    cache_db        = "clawlite_cache.db",
    analytics_path  = "clawlite_analytics.json",
    max_tokens_ctx  = 4000,    # context compressor budget
    keep_recent     = 6,       # messages to keep verbatim
    similarity_thr  = 0.82,    # fuzzy cache threshold (0-1)
    daily_budget    = 0.0,     # USD limit; 0 = unlimited
    cache_ttl       = 86400,   # cache TTL in seconds (24h)
    verbose         = False,   # print routing decisions
)
```

---

## Benchmark Results

Simulated 8-request session (37.5% cache hit rate):

| Metric | Value |
|--------|-------|
| Total requests | 8 |
| Cache hit rate | 37.5% |
| Tokens saved (cache) | 1,330 |
| Tokens saved (compress) | 2,600 |
| Total cost | $0.0487 |
| Cost saved | $0.0262 |
| Saving | 35% |

Real-world sessions with higher repetition (FAQ-style, long conversations) typically achieve 60-80% savings.

---

## Run the Demo

```bash
python3 demo.py
```

No API key needed. Demonstrates all 3 layers with mock data.

---

## Architecture

```
clawlite/
├── clawlite.py      # Main class — orchestrates all layers
├── router.py        # Query classifier (zero-cost heuristics)
├── cache.py         # Semantic cache (SQLite + embeddings)
├── compressor.py    # Context compressor (heuristic summarizer)
├── analytics.py     # Token/cost tracker with budget alerts
├── demo.py          # Full demo — no API key needed
└── examples/
    └── basic_usage.py
```

---

## Legal

This is pure engineering optimization — no API terms violations:

- Caching is standard practice (every CDN does this)
- Model routing is your choice as the API consumer
- Context compression is pre-processing on your own data
- No prompt injection, no jailbreaks, no bypass of safety filters

---

## License

Proprietary Evaluation License. See LICENSE file.
Contact: satryayogapratama@gmail.com for commercial licensing.

---

## Author

**Satrya Yoga Pratama** — ERP Developer, CFO Systems Architect, AI/ML Engineer

GitHub: [satryayogapratama-ux](https://github.com/satryayogapratama-ux)
