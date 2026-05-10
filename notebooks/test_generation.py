from dotenv import load_dotenv
load_dotenv()

from src.generation.generator import query_pipeline

queries = [
    "What were Apple's biggest risk factors in fiscal year 2025?",
    "What was NVIDIA's data center revenue growth in 2025?",
    "How did Microsoft describe its AI strategy in its most recent 10-K?",
]

for query in queries:
    print("\n" + "=" * 70)
    print(f"Q: {query}")
    print("=" * 70)
    result = query_pipeline(query)
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Tokens used: {result.get('usage', {})}")