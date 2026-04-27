"""
ClawLite Semantic Cache — Zero-token response reuse.
Exact match (hash) + fuzzy match (cosine similarity).
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


@dataclass
class CacheEntry:
    query: str
    response: str
    model: str
    tokens_used: int
    created_at: float
    ttl: float          # seconds; 0 = never expires
    hit_count: int = 0


@dataclass
class CacheResult:
    hit: bool
    response: Optional[str] = None
    similarity: float = 0.0
    match_type: str = "none"   # "exact", "semantic", "none"
    tokens_saved: int = 0


# Module-level model cache (loaded once per process)
_EMBED_MODEL = None

def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None and HAS_EMBEDDINGS:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


class SemanticCache:
    """
    Two-layer cache:
    1. Exact match via MD5 hash — O(1), zero embedding cost
    2. Fuzzy match via cosine similarity — catches rephrased queries
    """

    def __init__(
        self,
        db_path: str = "clawlite_cache.db",
        similarity_threshold: float = 0.82,  # tuned for Indonesian mixed-language content
        default_ttl: float = 86400,  # 24 hours
        embed_model: str = "all-MiniLM-L6-v2",
        max_entries: int = 10000,
    ):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.embed_model_name = embed_model
        self.max_entries = max_entries
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    query TEXT,
                    response TEXT,
                    model TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    embedding BLOB,
                    created_at REAL,
                    expires_at REAL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON cache(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
            conn.commit()

    def _hash(self, query: str, tenant: str = "default") -> str:
        key = f"{tenant}:{query.strip().lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def _embed(self, text: str) -> Optional[bytes]:
        model = _get_model(self.embed_model_name)
        if not model:
            return None
        vec = model.encode(text, convert_to_numpy=True)
        return vec.astype("float32").tobytes()

    def _cosine(self, a: bytes, b: bytes) -> float:
        if not HAS_EMBEDDINGS:
            return 0.0
        va = np.frombuffer(a, dtype=np.float32)
        vb = np.frombuffer(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    def _is_expired(self, expires_at: float) -> bool:
        return expires_at > 0 and time.time() > expires_at

    def get(self, query: str, tenant: str = "default") -> CacheResult:
        """Look up query. Returns CacheResult with hit=True if found."""
        now = time.time()
        query_hash = self._hash(query, tenant)

        with sqlite3.connect(self.db_path) as conn:
            # Layer 1: exact match
            row = conn.execute(
                "SELECT id, response, tokens_used, expires_at FROM cache WHERE query_hash=?",
                (query_hash,)
            ).fetchone()

            if row:
                if self._is_expired(row[3]):
                    conn.execute("DELETE FROM cache WHERE id=?", (row[0],))
                else:
                    conn.execute("UPDATE cache SET hit_count=hit_count+1 WHERE id=?", (row[0],))
                    return CacheResult(hit=True, response=row[1], similarity=1.0,
                                      match_type="exact", tokens_saved=row[2])

            # Layer 2: semantic match
            if not HAS_EMBEDDINGS:
                return CacheResult(hit=False)

            query_emb = self._embed(query)
            if not query_emb:
                return CacheResult(hit=False)

            rows = conn.execute(
                "SELECT id, response, tokens_used, embedding, expires_at FROM cache WHERE embedding IS NOT NULL"
            ).fetchall()

            best_sim, best_row = 0.0, None
            for r in rows:
                if self._is_expired(r[4]):
                    continue
                if r[3]:
                    sim = self._cosine(query_emb, r[3])
                    if sim > best_sim:
                        best_sim, best_row = sim, r

            if best_row and best_sim >= self.similarity_threshold:
                conn.execute("UPDATE cache SET hit_count=hit_count+1 WHERE id=?", (best_row[0],))
                return CacheResult(hit=True, response=best_row[1], similarity=best_sim,
                                  match_type="semantic", tokens_saved=best_row[2])

        return CacheResult(hit=False)

    def set(self, query: str, response: str, model: str = "",
            tokens_used: int = 0, ttl: float = None, tenant: str = "default") -> bool:
        """Store a query-response pair."""
        if ttl is None:
            ttl = self.default_ttl

        query_hash = self._hash(query, tenant)
        expires_at = time.time() + ttl if ttl > 0 else 0
        embedding = self._embed(query)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache
                    (query_hash, query, response, model, tokens_used, embedding, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (query_hash, query, response, model, tokens_used,
                      embedding, time.time(), expires_at))
                conn.commit()
            self._evict_if_needed()
            return True
        except Exception as e:
            return False

    def _evict_if_needed(self):
        """LRU eviction when cache exceeds max_entries."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            if count > self.max_entries:
                # Delete oldest 10% by created_at
                evict_n = max(1, count // 10)
                conn.execute("""
                    DELETE FROM cache WHERE id IN (
                        SELECT id FROM cache ORDER BY created_at ASC LIMIT ?
                    )
                """, (evict_n,))
                conn.commit()

    def invalidate(self, query: str) -> bool:
        """Remove a specific entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache WHERE query_hash=?", (self._hash(query),))
            conn.commit()
        return True

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM cache WHERE expires_at > 0 AND expires_at < ?", (time.time(),))
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            total   = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            hits    = conn.execute("SELECT SUM(hit_count) FROM cache").fetchone()[0] or 0
            saved   = conn.execute("SELECT SUM(tokens_used * hit_count) FROM cache").fetchone()[0] or 0
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE expires_at > 0 AND expires_at < ?", (time.time(),)
            ).fetchone()[0]
        return {
            "total_entries": total,
            "total_hits": hits,
            "total_tokens_saved": saved,
            "expired_entries": expired,
        }
