from dotenv import load_dotenv
load_dotenv()

from src.hallucination.nli_scorer import load_nli_model, score_answer
from src.generation.generator import query_pipeline

# Load NLI model once — takes 30-60 seconds first time
print("Loading NLI model...")
nli = load_nli_model()
print("Model loaded\n")

# Test 1 — grounded answer (should PASS)
print("=" * 60)
print("TEST 1 — Grounded query (expect PASS)")
print("=" * 60)
result = query_pipeline("What was NVIDIA's data center revenue in 2025?")
answer = result["answer"]
chunks = result["chunks_used"]

print(f"Answer: {answer}\n")
scoring = score_answer(answer, chunks, nli)
print(f"Verdict: {scoring['verdict']}")
print(f"Support rate: {scoring['support_rate']:.0%}")
print(f"Should refuse: {scoring['should_refuse']}")
print(f"Reason: {scoring['reason']}")

# Test 2 — hallucinated answer (should FAIL or WARN)
print("\n" + "=" * 60)
print("TEST 2 — Hallucinated claim (expect FAIL/WARN)")
print("=" * 60)
fake_answer = "NVIDIA's data center revenue was $500 billion in 2025, making it the largest company in history."
print(f"Fake answer: {fake_answer}\n")
scoring2 = score_answer(fake_answer, chunks, nli)
print(f"Verdict: {scoring2['verdict']}")
print(f"Support rate: {scoring2['support_rate']:.0%}")
print(f"Should refuse: {scoring2['should_refuse']}")
print(f"Reason: {scoring2['reason']}")
for r in scoring2["claim_results"]:
    print(f"  [{r['verdict']}] {r['claim'][:100]}")