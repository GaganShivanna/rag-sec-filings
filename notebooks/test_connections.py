from dotenv import load_dotenv
import os
load_dotenv()

# Test 1 — OpenAI
from openai import OpenAI
client = OpenAI()
emb = client.embeddings.create(input="test SEC filing", model="text-embedding-3-large")
print(f"✓ OpenAI — embedding dim: {len(emb.data[0].embedding)}")  # should print 3072

# Test 2 — Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
print(f"✓ Pinecone — index stats: {index.describe_index_stats()}")

print("\n All connections working. Ready to build.")