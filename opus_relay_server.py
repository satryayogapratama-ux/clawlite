#!/usr/bin/env python3
"""HTTP relay server for Anthropic Claude Opus code reviews with history & replay."""

import http.server
import json
import logging
import os
import secrets
import socketserver
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Configuration
MAX_REQUEST_SIZE = 100 * 1024  # 100KB
MAX_CODE_LENGTH = 50 * 1024    # 50KB
API_TIMEOUT = 120
LISTEN_PORT = int(os.environ.get("RELAY_PORT", "5679"))
LISTEN_HOST = os.environ.get("RELAY_HOST", "0.0.0.0")
DB_PATH = "/opt/opus-relay/history.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_config():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    auth_token = os.environ.get("RELAY_AUTH_TOKEN")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable required")
    if not auth_token:
        raise RuntimeError("RELAY_AUTH_TOKEN environment variable required")
    return {"api_key": api_key, "auth_token": auth_token}


CONFIG = get_config()


# ─── Database initialization ──────────────────────────────────────────────────

def init_db():
    """Initialize SQLite database for history storage."""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            prompt TEXT,
            code TEXT,
            context TEXT,
            response TEXT,
            duration_ms INTEGER,
            replay_of INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def save_review(prompt, code, context, response, duration_ms, replay_of=None):
    """Save a review to database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created_at = datetime.utcnow().isoformat() + "Z"
    c.execute("""
        INSERT INTO reviews (created_at, prompt, code, context, response, duration_ms, replay_of)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (created_at, prompt, code, context, response, duration_ms, replay_of))
    conn.commit()
    review_id = c.lastrowid
    conn.close()
    return review_id


def get_review(review_id):
    """Get a review by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, created_at, prompt, code, context, response, duration_ms
        FROM reviews WHERE id = ?
    """, (review_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "created_at": row[1],
            "prompt": row[2],
            "code": row[3],
            "context": row[4],
            "response": row[5],
            "duration_ms": row[6]
        }
    return None


def get_recent_reviews(limit=20):
    """Get recent reviews."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, created_at, prompt, response, duration_ms
        FROM reviews
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    
    reviews = []
    for row in rows:
        prompt_text = row[2] or ""
        prompt_preview = prompt_text[:100] if len(prompt_text) > 100 else prompt_text
        reviews.append({
            "id": row[0],
            "created_at": row[1],
            "prompt_preview": prompt_preview,
            "duration_ms": row[4]
        })
    return reviews


# ─── HTTP Server ──────────────────────────────────────────────────────────────

class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class RelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_json_response(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, status, message):
        self.send_json_response(status, {"error": message})

    def authenticate(self):
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header[7:]
        return secrets.compare_digest(token, CONFIG["auth_token"])

    def do_POST(self):
        client_ip = self.client_address[0]

        if not self.authenticate():
            logger.warning("Auth failure from %s", client_ip)
            self.send_error_response(401, "Unauthorized")
            return

        # Handle /replay/<id> endpoint
        if self.path.startswith("/replay/"):
            self.handle_replay(client_ip)
            return

        # Regular review endpoint
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self.send_error_response(400, "Invalid Content-Length")
            return

        if length <= 0:
            self.send_error_response(400, "Empty request body")
            return
        if length > MAX_REQUEST_SIZE:
            self.send_error_response(413, "Request too large")
            return

        try:
            raw_body = self.rfile.read(length)
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON")
            return

        custom_prompt = body.get("prompt", "")
        code = body.get("code", "")
        context = body.get("context", "")

        if custom_prompt and isinstance(custom_prompt, str) and custom_prompt.strip():
            if len(custom_prompt) > MAX_REQUEST_SIZE:
                self.send_error_response(413, "Prompt too large")
                return
            logger.info("Prompt request from %s: %d bytes", client_ip, len(custom_prompt))
            try:
                t0 = time.time()
                review = self.call_anthropic_api_raw(custom_prompt)
                duration_ms = int((time.time() - t0) * 1000)
                review_id = save_review(prompt=custom_prompt, code=None, context=None, 
                                       response=review, duration_ms=duration_ms)
                logger.info(f"Review #{review_id} saved (duration: {duration_ms}ms)")
                self.send_json_response(200, {"review": review, "review_id": review_id})
            except urllib.error.HTTPError as e:
                logger.error("Anthropic API HTTP error: %d", e.code)
                self.send_error_response(502, "Upstream API error")
            except urllib.error.URLError as e:
                logger.error("Anthropic API connection error: %s", e.reason)
                self.send_error_response(502, "Upstream connection failed")
            except Exception:
                logger.exception("Unexpected error")
                self.send_error_response(500, "Internal server error")
            return

        if not isinstance(code, str) or not isinstance(context, str):
            self.send_error_response(400, "Invalid field types")
            return
        if not code.strip():
            self.send_error_response(400, "Code field required")
            return
        if len(code) > MAX_CODE_LENGTH:
            self.send_error_response(413, "Code too large")
            return

        logger.info("Review request from %s: %d bytes", client_ip, len(code))

        try:
            t0 = time.time()
            review = self.call_anthropic_api(code, context)
            duration_ms = int((time.time() - t0) * 1000)
            review_id = save_review(prompt=None, code=code, context=context,
                                   response=review, duration_ms=duration_ms)
            logger.info(f"Review #{review_id} saved (duration: {duration_ms}ms)")
            self.send_json_response(200, {"review": review, "review_id": review_id})
        except urllib.error.HTTPError as e:
            logger.error("Anthropic API HTTP error: %d", e.code)
            self.send_error_response(502, "Upstream API error")
        except urllib.error.URLError as e:
            logger.error("Anthropic API connection error: %s", e.reason)
            self.send_error_response(502, "Upstream connection failed")
        except Exception:
            logger.exception("Unexpected error")
            self.send_error_response(500, "Internal server error")

    def handle_replay(self, client_ip):
        """Handle POST /replay/<id> to re-run a review."""
        try:
            review_id = int(self.path.split("/")[-1])
        except (ValueError, IndexError):
            self.send_error_response(400, "Invalid review ID")
            return

        original = get_review(review_id)
        if not original:
            self.send_error_response(404, "Review not found")
            return

        logger.info(f"Replaying review #{review_id} from {client_ip}")

        try:
            t0 = time.time()
            
            if original["prompt"]:
                # Raw prompt replay
                review = self.call_anthropic_api_raw(original["prompt"])
            else:
                # Code review replay
                review = self.call_anthropic_api(original["code"], original["context"])
            
            duration_ms = int((time.time() - t0) * 1000)
            
            # Save as new entry with reference to original
            new_context = original.get("context", "") or ""
            if new_context:
                new_context += f"\n[Replay of review #{review_id}]"
            else:
                new_context = f"[Replay of review #{review_id}]"
            
            new_id = save_review(
                prompt=original.get("prompt"),
                code=original.get("code"),
                context=new_context,
                response=review,
                duration_ms=duration_ms,
                replay_of=review_id
            )
            
            logger.info(f"Replay complete: new review #{new_id} (duration: {duration_ms}ms)")
            self.send_json_response(200, {
                "review": review,
                "review_id": new_id,
                "replay_of": review_id,
                "duration_ms": duration_ms
            })
        except urllib.error.HTTPError as e:
            logger.error("Anthropic API HTTP error: %d", e.code)
            self.send_error_response(502, "Upstream API error")
        except urllib.error.URLError as e:
            logger.error("Anthropic API connection error: %s", e.reason)
            self.send_error_response(502, "Upstream connection failed")
        except Exception:
            logger.exception("Unexpected error during replay")
            self.send_error_response(500, "Internal server error")

    def call_anthropic_api_raw(self, prompt):
        payload = json.dumps({
            "model": "claude-opus-4-7",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload, method="POST",
            headers={
                "x-api-key": CONFIG["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result.get("content", [])
            if content and isinstance(content[0], dict):
                return content[0].get("text", "")
            return ""

    def call_anthropic_api(self, code, context):
        prompt = (
            "You are a senior code reviewer. Review this code for bugs, "
            "security, performance, best practices.\n\n"
            f"Context: {context}\n\n"
            f"Code:\n{code}"
        )
        payload = json.dumps({
            "model": "claude-opus-4-7",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload, method="POST",
            headers={
                "x-api-key": CONFIG["api_key"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result.get("content", [])
            if content and isinstance(content[0], dict):
                return content[0].get("text", "")
            return ""

    def do_GET(self):
        if self.path == "/health":
            self.send_json_response(200, {"status": "ok", "version": "1.1.0"})
        elif self.path == "/history":
            if not self.authenticate():
                logger.warning("Auth failure from %s", self.client_address[0])
                self.send_error_response(401, "Unauthorized")
                return
            reviews = get_recent_reviews(limit=20)
            self.send_json_response(200, {"reviews": reviews})
        else:
            self.send_error_response(404, "Not found")

    def log_message(self, format, *args):
        logger.debug("%s - %s", self.client_address[0], format % args)


def main():
    init_db()
    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), RelayHandler)
    logger.info("Opus relay v1.1.0 starting on %s:%d", LISTEN_HOST, LISTEN_PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
