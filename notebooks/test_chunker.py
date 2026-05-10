from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import load_all_filings
from src.ingestion.chunker import chunk_documents

docs = load_all_filings("data/raw")

# Test fixed chunking
fixed_chunks = chunk_documents(docs, strategy="fixed", chunk_size=1000, overlap=200)
print(f"\nFixed chunking: {len(fixed_chunks)} total chunks")
print(f"Sample chunk:\n{fixed_chunks[0].content[:300]}")
print(f"Chunk ID: {fixed_chunks[0].chunk_id}")

# Test semantic chunking
semantic_chunks = chunk_documents(docs, strategy="semantic", chunk_size=1000)
print(f"\nSemantic chunking: {len(semantic_chunks)} total chunks")
print(f"Sample chunk:\n{semantic_chunks[0].content[:300]}")

# Compare
print(f"\nFixed: {len(fixed_chunks)} chunks")
print(f"Semantic: {len(semantic_chunks)} chunks")
print(f"Avg fixed chunk size: {sum(c.char_count for c in fixed_chunks) // len(fixed_chunks)} chars")
print(f"Avg semantic chunk size: {sum(c.char_count for c in semantic_chunks) // len(semantic_chunks)} chars")