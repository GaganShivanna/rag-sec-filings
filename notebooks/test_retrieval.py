from dotenv import load_dotenv
load_dotenv()

from src.retrieval.query_engine import retrieve

# Test 1 — broad query
print("=" * 60)
print("Query: Apple risk factors 2025")
print("=" * 60)
results = retrieve("What were Apple's biggest risk factors in 2025?", top_k=5)
for i, r in enumerate(results):
    print(f"\nChunk {i+1} | {r['ticker']} | Score: {r['score']:.4f}")
    print(r["content"][:300])

# Test 2 — company filtered query
print("\n" + "=" * 60)
print("Query: NVDA revenue filtered to NVDA only")
print("=" * 60)
results = retrieve(
    "What was NVIDIA's data center revenue growth?",
    top_k=5,
    ticker_filter="NVDA"
)
for i, r in enumerate(results):
    print(f"\nChunk {i+1} | {r['ticker']} | Score: {r['score']:.4f}")
    print(r["content"][:300])