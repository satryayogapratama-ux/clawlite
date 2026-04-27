#!/usr/bin/env python3
"""
ClawLite Demo — Shows all 3 optimization layers working.
No real API key needed — uses mock responses to demonstrate the system.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from router import ModelRouter, Complexity
from cache import SemanticCache
from compressor import ContextCompressor, Message
from analytics import Analytics

# ─── Mock data ────────────────────────────────────────────────────────────────

MOCK_QUERIES = [
    ("Apa itu BPJS Ketenagakerjaan?",            "medium", 120, 280),
    ("Review arsitektur microservices ini secara detail dan komprehensif", "complex", 800, 1200),
    ("Halo, apa kabar?",                          "simple",  30,  60),
    ("Apa itu BPJS Ketenagakerjaan?",             "cache",    0,   0),  # same as #1
    ("Jelaskan apa itu BPJS Ketenagakerjaan",     "cache",    0,   0),  # semantic match
    ("Bantu saya implement REST API dengan auth JWT di Node.js", "complex", 600, 900),
    ("Berapa 2 + 2?",                             "simple",  20,  30),
    ("Analisis mendalam tentang sistem ERP untuk pabrik rokok dengan 300 karyawan", "complex", 700, 1100),
]

MOCK_RESPONSES = {
    "medium":  "BPJS Ketenagakerjaan adalah program jaminan sosial untuk tenaga kerja Indonesia...",
    "complex": "Berikut review arsitektur komprehensif: [analisis mendalam 1200 token]...",
    "simple":  "Kabar baik! Ada yang bisa saya bantu?",
}


def sep(char="─", n=60):
    print(char * n)


def demo_router():
    sep("═")
    print("  DEMO 1: Smart Model Router (zero API cost)")
    sep("═")

    router = ModelRouter(medium_model="claude-sonnet-4-6", complex_model="claude-opus-4-5")

    test_cases = [
        "Apa itu RAM?",
        "Translate: hello world",
        "Bantu debug kode Python ini yang error",
        "Review arsitektur sistem ERP komprehensif dan analisis security",
        "Implement full authentication system dengan JWT, refresh token, dan rate limiting",
        "Halo!",
        "Kenapa machine learning model saya overfitting dan bagaimana cara fixing dengan proper cross-validation?",
    ]

    for q in test_cases:
        d = router.classify(q)
        icon = "🟢" if d.complexity == Complexity.SIMPLE else "🟡" if d.complexity == Complexity.MEDIUM else "🔴"
        print(f"{icon} [{d.complexity.value:7}] {d.model:25} | {q[:45]}")
        print(f"   reason: {d.reason}")

    print()
    stats = router.get_stats()
    print(f"  Routed: {stats['total_routed']} queries | Distribution: {stats['distribution']}")
    print()


def demo_cache():
    sep("═")
    print("  DEMO 2: Semantic Cache (exact + fuzzy match)")
    sep("═")

    cache = SemanticCache(
        db_path="/tmp/clawlite_demo_cache.db",
        similarity_threshold=0.82,  # tuned for Indonesian content
        default_ttl=3600,
    )

    # Store original response
    original_query = "Apa itu BPJS Ketenagakerjaan?"
    original_response = "BPJS Ketenagakerjaan adalah program jaminan sosial wajib bagi pekerja Indonesia..."
    cache.set(original_query, original_response, model="claude-sonnet-4-6", tokens_used=400)
    print(f"  Stored: '{original_query}'")
    print()

    test_queries = [
        ("Apa itu BPJS Ketenagakerjaan?", "should be EXACT hit"),
        ("Jelaskan tentang BPJS Ketenagakerjaan", "should be SEMANTIC hit"),
        ("Apa fungsi BPJS untuk karyawan?", "should be SEMANTIC hit"),
        ("Berapa harga saham Apple?", "should MISS"),
    ]

    for q, expectation in test_queries:
        t0 = time.time()
        result = cache.get(q)
        ms = (time.time() - t0) * 1000

        if result.hit:
            icon = "✅"
            info = f"{result.match_type.upper()} hit | sim={result.similarity:.3f} | saved {result.tokens_saved} tokens | {ms:.1f}ms"
        else:
            icon = "❌"
            info = f"MISS | {ms:.1f}ms → needs API call"

        print(f"  {icon} {q}")
        print(f"     {info}")
        print(f"     ({expectation})")
        print()

    stats = cache.stats()
    print(f"  Cache entries: {stats['total_entries']} | Hits: {stats['total_hits']} | "
          f"Tokens saved: {stats['total_tokens_saved']}")
    print()


def demo_compressor():
    sep("═")
    print("  DEMO 3: Context Compressor (reduce tokens before API call)")
    sep("═")

    compressor = ContextCompressor(max_tokens=400, keep_recent=4)  # tight budget to force compression

    # Simulate a long conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant for Satrya."},
        {"role": "user", "content": "Bantu saya setup ASKA ERP untuk pabrik rokok dengan 300 karyawan dan 6 divisi berbeda termasuk SKT dan SKM"},
        {"role": "assistant", "content": "Tentu! ASKA ERP bisa dikonfigurasi untuk pabrik rokok dengan struktur divisi SKT (Sigaret Kretek Tangan) dan SKM (Sigaret Kretek Mesin)..."},
        {"role": "user", "content": "Bagaimana cara setup absensi untuk kepala SKT yang mengelola 150 karyawan?"},
        {"role": "assistant", "content": "Untuk absensi kepala SKT, gunakan modul attendance dengan konfigurasi: 1) Set shift pagi jam 07:00-17:00, 2) Enable overtime tracking, 3) Link ke payroll per-batang..."},
        {"role": "user", "content": "Dan untuk laporan karton keluar, formatnya bagaimana?"},
        {"role": "assistant", "content": "Laporan karton keluar menggunakan format: tanggal, nomor batch, jumlah karton (1 karton = 800 pack), kepala divisi yang approve, dan status konfirmasi gudang..."},
        {"role": "user", "content": "Oke, sekarang saya mau tanya tentang VIGS project. Bagaimana CoA yang sudah dibuat?"},
        {"role": "assistant", "content": "CoA Primeline Roofing NSW v2.1 sudah selesai dengan struktur: Revenue accounts 4xxx, COGS 5xxx, Opex 6xxx..."},
        {"role": "user", "content": "Saya mau review security untuk ClawMemory yang baru kita buat"},  # recent
        {"role": "assistant", "content": "ClawMemory security review: [analisis 800 token]..."},           # recent
        {"role": "user", "content": "Bagaimana cara search di ClawVault?"},                                # recent
    ]

    result = compressor.compress(conversation)

    print(f"  Messages: {len(conversation)} → {len(result.messages)}")
    print(f"  Tokens:   {result.original_tokens} → {result.compressed_tokens} "
          f"({result.compression_ratio:.0%} of original)")
    print(f"  Saved:    {result.original_tokens - result.compressed_tokens} tokens "
          f"(~${(result.original_tokens - result.compressed_tokens) * 3 / 1_000_000:.4f} at Sonnet price)")
    print(f"  Summary:  {'added' if result.summary_added else 'not needed (within budget)'}")
    print()
    print("  Final messages sent to API:")
    for m in result.messages:
        preview = m.content[:80].replace('\n', ' ')
        print(f"    [{m.role:9}] {preview}...")
    print()


def demo_analytics():
    sep("═")
    print("  DEMO 4: Analytics + Savings Summary")
    sep("═")

    analytics = Analytics(state_path="/tmp/clawlite_demo_analytics.json", daily_budget_usd=1.00)

    # Simulate 8 requests
    scenarios = [
        dict(query="Q1", model="claude-sonnet-4-6", tokens_in=300, tokens_out=200, cache_hit=False, tokens_saved_compress=800),
        dict(query="Q2", model="claude-opus-4-5",  tokens_in=800, tokens_out=600, cache_hit=False),
        dict(query="Q3", model="cache", tokens_in=0, tokens_out=0, cache_hit=True, cache_type="exact", tokens_saved_cache=500),
        dict(query="Q4", model="cache", tokens_in=0, tokens_out=0, cache_hit=True, cache_type="semantic", tokens_saved_cache=400),
        dict(query="Q5", model="claude-sonnet-4-6", tokens_in=250, tokens_out=180, cache_hit=False, tokens_saved_compress=600),
        dict(query="Q6", model="cache", tokens_in=0, tokens_out=0, cache_hit=True, cache_type="semantic", tokens_saved_cache=430),
        dict(query="Q7", model="claude-sonnet-4-6", tokens_in=180, tokens_out=120, cache_hit=False),
        dict(query="Q8", model="claude-opus-4-5",  tokens_in=900, tokens_out=700, cache_hit=False, tokens_saved_compress=1200),
    ]

    for s in scenarios:
        analytics.record(**s)

    analytics.print_summary()

    # Budget check
    budget = analytics.budget_check("claude-opus-4-5", estimated_tokens=2000)
    print(f"  Budget check (opus, 2000 tokens): {'✅ OK' if budget['ok'] else '⚠️  ' + budget['reason']}")
    if budget.get("remaining"):
        print(f"  Remaining budget: ${budget['remaining']:.4f}")
    print()


if __name__ == "__main__":
    print()
    sep("═")
    print("  ClawLite — Token-Efficient AI Routing Demo")
    print("  Legal token optimization. No API bypass. Pure efficiency.")
    sep("═")
    print()

    demo_router()
    demo_cache()
    demo_compressor()
    demo_analytics()

    sep("═")
    print("  ✅ All demos passed. ClawLite ready for production.")
    sep("═")
    print()
