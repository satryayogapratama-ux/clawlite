"""Gunicorn config for ClawLite Proxy — production-grade WSGI server."""

import multiprocessing
import os

# Bind
bind = "127.0.0.1:8383"

# Workers — 2x CPU + 1, min 2 for concurrency
# gthread: blocking I/O per thread, correct for upstream HTTP proxy calls
# sync workers block the entire worker during Anthropic API call (10-60s) — wrong for this use case
workers = 2
threads = 8          # 2 workers x 8 threads = 16 concurrent upstream calls
worker_class = "gthread"

# Timeouts
timeout = 300        # upstream Anthropic can be slow (streaming)
keepalive = 5
graceful_timeout = 30

# Logging
accesslog = "-"      # stdout
errorlog  = "-"      # stderr
loglevel  = "info"

# Stability
max_requests = 500           # recycle workers to prevent memory leak
max_requests_jitter = 50     # randomize to avoid thundering herd
preload_app = True           # load app once, fork — cheaper memory

# Process name
proc_name = "clawlite-proxy"
