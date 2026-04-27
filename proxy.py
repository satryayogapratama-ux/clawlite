#!/usr/bin/env python3
"""
ClawLite Proxy — Anthropic API-compatible local proxy with caching + routing.

Runs on localhost:8383. OpenClaw points to this as a custom provider.
Every call goes through: cache → router → compressor → real Anthropic → cache store.

Start:
    python3 proxy.py

OpenClaw config (models.providers):
    "clawlite": {
        "baseUrl": "http://localhost:8383",
        "api": "anthropic-messages",
        "apiKey": "<your-real-api-key>",
        "models": [{"id": "claude-sonnet-4-6", "name": "Sonnet (ClawLite)"},
                   {"id": "claude-opus-4-5",   "name": "Opus (ClawLite)"}]
    }
"""

import json
import logging
import os
import sys
import time
import threading
from pathlib import Path

import requests
from flask import Flask, request, jsonify, Response, stream_with_context

sys.path.insert(0, str(Path(__file__).parent))
from router import ModelRouter, Complexity
from cache import SemanticCache, CacheResult
from compressor import ContextCompressor
from analytics import Analytics

# ─── Config ───────────────────────────────────────────────────────────────────

PORT          = int(os.environ.get("CLAWLITE_PORT", "8383"))
UPSTREAM      = "https://api.anthropic.com"
DB_DIR        = Path(os.environ.get("CLAWLITE_DB_DIR", str(Path(__file__).parent)))
SIMILARITY    = float(os.environ.get("CLAWLITE_SIM_THRESHOLD", "0.82"))
MAX_CTX       = int(os.environ.get("CLAWLITE_MAX_CTX_TOKENS", "6000"))
KEEP_RECENT   = int(os.environ.get("CLAWLITE_KEEP_RECENT", "6"))
DAILY_BUDGET  = float(os.environ.get("CLAWLITE_DAILY_BUDGET", "0"))
CACHE_TTL     = float(os.environ.get("CLAWLITE_CACHE_TTL", "86400"))
LOG_LEVEL     = os.environ.get("CLAWLITE_LOG_LEVEL", "INFO")
BEARER_TOKEN  = os.environ.get("CLAWLITE_BEARER_TOKEN", "")  # optional auth for proxy itself

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [ClawLite] %(message)s")
log = logging.getLogger("clawlite")

# ─── Component init ───────────────────────────────────────────────────────────

router     = ModelRouter(medium_model="claude-sonnet-4-6", complex_model="claude-opus-4-5", force_medium=True)
cache      = SemanticCache(db_path=str(DB_DIR / "clawlite_proxy_cache.db"), similarity_threshold=SIMILARITY, default_ttl=CACHE_TTL)
compressor = ContextCompressor(max_tokens=MAX_CTX, keep_recent=KEEP_RECENT)
analytics  = Analytics(state_path=str(DB_DIR / "clawlite_proxy_analytics.json"), daily_budget_usd=DAILY_BUDGET)

app = Flask(__name__)

# ─── Uptime tracking ──────────────────────────────────────────────────────────

START_TIME = time.time()
VERSION = "1.1.0"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _extract_query(messages: list[dict]) -> str:
    """Get last user message as the cache/routing query."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
    return ""


def _make_cache_response(cached_text: str, model: str, req_id: str) -> dict:
    """Build Anthropic-format response from cached text."""
    return {
        "id": f"clawlite-cache-{req_id}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": cached_text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "clawlite_cache_hit": True,
    }


def _forward_to_anthropic(payload: dict, headers: dict, max_retries: int = 3) -> requests.Response:
    """Forward request to real Anthropic API with exponential backoff retry logic."""
    # Case-insensitive header lookup
    def _h(key):
        return headers.get(key) or headers.get(key.lower()) or headers.get(key.title()) or headers.get(key.upper())

    upstream_headers = {
        "content-type": "application/json",
        "anthropic-version": _h("anthropic-version") or "2023-06-01",
    }
    # Forward API key
    api_key = _h("x-api-key")
    auth    = _h("authorization")
    if api_key:
        upstream_headers["x-api-key"] = api_key
    elif auth:
        upstream_headers["authorization"] = auth

    # Forward beta headers if present
    beta = _h("anthropic-beta")
    if beta:
        upstream_headers["anthropic-beta"] = beta

    streaming = payload.get("stream", False)

    # Exponential backoff: 1s, 2s, 4s
    backoff_times = [1, 2, 4]
    retryable_status_codes = {429, 503, 529}
    retryable_exceptions = (requests.ConnectionError, requests.Timeout)

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                f"{UPSTREAM}/v1/messages",
                json=payload,
                headers=upstream_headers,
                stream=streaming,
                timeout=(10, 300),  # connect 10s, read 300s
            )

            # Check for retryable HTTP status codes
            if resp.status_code in retryable_status_codes and attempt < max_retries:
                backoff = backoff_times[attempt]
                if resp.status_code == 529:
                    log.warning(f"Anthropic overloaded (529) — retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                else:
                    log.warning(f"Anthropic returned {resp.status_code} — retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(backoff)
                continue

            # Do NOT retry on client errors (400, 401, 403, 404, etc)
            if 400 <= resp.status_code < 500:
                return resp

            return resp

        except retryable_exceptions as e:
            if attempt < max_retries:
                backoff = backoff_times[attempt]
                log.warning(f"Upstream connection error ({type(e).__name__}) — retrying in {backoff}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(backoff)
                continue
            else:
                raise

    # Should not reach here, but just in case
    raise requests.ConnectionError("Max retries exceeded for upstream Anthropic API")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    uptime_seconds = int(time.time() - START_TIME)
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "proxy_status": "ok",
        "uptime_seconds": uptime_seconds,
        "cache_entries": cache.stats()["total_entries"],
        "analytics": analytics.summary(),
    })


@app.route("/v1/messages", methods=["POST"])
def messages():
    t0 = time.time()

    # Optional proxy auth
    if BEARER_TOKEN:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {BEARER_TOKEN}":
            return jsonify({"error": "unauthorized"}), 401

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    messages_list = body.get("messages", [])
    orig_model    = body.get("model", "claude-sonnet-4-6")
    is_streaming  = body.get("stream", False)

    # 1. Extract query for cache + routing
    query = _extract_query(messages_list)
    if not query:
        # No user message — pass through directly
        try:
            resp = _forward_to_anthropic(body, dict(request.headers))
        except requests.ConnectionError as e:
            return jsonify({"error": f"upstream connection error: {e}"}), 502
        except requests.Timeout:
            return jsonify({"error": "upstream timeout"}), 504

        return Response(resp.content, status=resp.status_code,
                       content_type=resp.headers.get("content-type", "application/json"))

    req_id = f"{int(t0 * 1000)}"

    # 2. Cache check (skip for streaming — streamed responses harder to reconstruct)
    if not is_streaming:
        cache_result = cache.get(query)
        if cache_result.hit:
            log.info(f"CACHE {cache_result.match_type.upper()} hit | sim={cache_result.similarity:.3f} | saved {cache_result.tokens_saved} tokens | {(time.time()-t0)*1000:.0f}ms")
            analytics.record(query=query, model="cache", cache_hit=True,
                           cache_type=cache_result.match_type, tokens_saved_cache=cache_result.tokens_saved)
            return jsonify(_make_cache_response(cache_result.response, orig_model, req_id))

    # 3. Route to best model
    decision = router.classify(query, conversation_length=len(messages_list))
    routed_model = decision.model
    log.info(f"ROUTE → {routed_model} ({decision.complexity.value}, conf={decision.confidence:.2f}) | {query[:50]}")

    # 4. Budget check
    budget = analytics.budget_check(routed_model, decision.estimated_tokens)
    if not budget["ok"]:
        log.warning(f"BUDGET exceeded — downgrading to sonnet: {budget['reason']}")
        routed_model = "claude-sonnet-4-6"

    # 5. Compress messages
    compress_result = compressor.compress(messages_list)
    tokens_saved_compress = max(0, compress_result.original_tokens - compress_result.compressed_tokens)
    if compress_result.summary_added:
        log.info(f"COMPRESS: {compress_result.original_tokens} → {compress_result.compressed_tokens} tokens (saved {tokens_saved_compress})")

    # Build final payload — extract system from compressed messages
    compressed_messages = []
    compressed_system_parts = []
    for m in compress_result.messages:
        if m.role == "system":
            compressed_system_parts.append(m.content)
        else:
            compressed_messages.append({"role": m.role, "content": m.content})

    final_payload = {**body, "model": routed_model, "messages": compressed_messages}
    if compressed_system_parts:
        final_payload["system"] = "\n".join(compressed_system_parts)

    # 6. Forward to Anthropic with retry logic
    try:
        upstream_resp = _forward_to_anthropic(final_payload, dict(request.headers))
    except requests.Timeout:
        return jsonify({"error": "upstream timeout"}), 504
    except requests.ConnectionError as e:
        return jsonify({"error": f"upstream connection error: {e}"}), 502

    # Handle 529 gracefully
    if upstream_resp.status_code == 529:
        log.warning(f"Anthropic overloaded (529) after retries — returning error to client")
        return jsonify({
            "error": {
                "type": "overloaded_error",
                "message": "Anthropic API is overloaded, please try again later"
            },
            "retry_after": 60
        }), 529

    # 7. For streaming, pass through with error handling
    if is_streaming:
        analytics.record(query=query, model=routed_model, compressed=compress_result.summary_added,
                        tokens_saved_compress=tokens_saved_compress)

        def generate():
            try:
                for chunk in upstream_resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
                # Ensure clean close
                yield b"data: [DONE]\n\n"
            except Exception as e:
                log.error(f"Stream generation error: {e}")
                yield b"data: [DONE]\n\n"

        return Response(stream_with_context(generate()),
                       status=upstream_resp.status_code,
                       content_type=upstream_resp.headers.get("content-type", "text/event-stream"))

    # 8. Parse, cache, return
    if upstream_resp.status_code != 200:
        return Response(upstream_resp.content, status=upstream_resp.status_code,
                       content_type=upstream_resp.headers.get("content-type", "application/json"))

    try:
        resp_json = upstream_resp.json()
    except Exception as e:
        log.error(f"Failed to parse upstream response: {e}")
        return Response(upstream_resp.content, status=200, content_type="application/json")

    # Extract text for caching
    resp_text = ""
    for block in resp_json.get("content", []):
        if block.get("type") == "text":
            resp_text += block.get("text", "")

    tokens_in  = resp_json.get("usage", {}).get("input_tokens", 0)
    tokens_out = resp_json.get("usage", {}).get("output_tokens", 0)

    if resp_text:
        cache.set(query, resp_text, model=routed_model,
                 tokens_used=tokens_in + tokens_out, ttl=CACHE_TTL)

    analytics.record(
        query=query, model=routed_model,
        tokens_in=tokens_in, tokens_out=tokens_out,
        compressed=compress_result.summary_added,
        tokens_saved_compress=tokens_saved_compress,
    )

    latency = (time.time() - t0) * 1000
    log.info(f"DONE {tokens_in}in/{tokens_out}out | {latency:.0f}ms | ${analytics.token_cost(tokens_in+tokens_out, routed_model):.4f}")

    # Inject metadata (non-breaking extension fields)
    resp_json["clawlite_model_requested"] = orig_model
    resp_json["clawlite_model_used"] = routed_model
    resp_json["clawlite_cache_hit"] = False

    return jsonify(resp_json)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"ClawLite Proxy v{VERSION} starting on port {PORT}")
    log.info(f"Cache DB: {DB_DIR}/clawlite_proxy_cache.db")
    log.info(f"Similarity threshold: {SIMILARITY} | Max context: {MAX_CTX} tokens")
    log.info(f"Streaming: supported (cache disabled for stream) | Budget: {'$'+str(DAILY_BUDGET) if DAILY_BUDGET else 'unlimited'}")
    log.info(f"Retry logic: enabled (max 3 retries, exponential backoff 1s/2s/4s)")

    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
