import os
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from src.ingestion.chunker import Chunk

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 50  # embed 50 chunks at a time


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME"))


def embed_chunks(chunks: list[Chunk], client: OpenAI) -> list[list[float]]:
    """
    Embed a batch of chunks using OpenAI text-embedding-3-large.
    Processes in batches to avoid rate limits.
    """
    all_embeddings = []
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        try:
            response = client.embeddings.create(
                input=[c.content for c in batch],
                model=EMBEDDING_MODEL
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

            logger.info(f"Embedded batch {batch_num}/{total_batches} ({len(batch)} chunks)")

            # Respect rate limits — pause between batches
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Embedding failed for batch {batch_num}: {e}")
            # Add zero vectors as placeholders so indexing doesn't break
            all_embeddings.extend([[0.0] * EMBEDDING_DIM] * len(batch))

    return all_embeddings


def upsert_to_pinecone(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    index
) -> int:
    """
    Upsert chunk embeddings to Pinecone with metadata.
    Returns number of vectors successfully upserted.
    """
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk.chunk_id,
            "values": embedding,
            "metadata": {
                "ticker": chunk.ticker,
                "filing_type": chunk.filing_type,
                "content": chunk.content[:1000],  # Pinecone metadata limit
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "char_count": chunk.char_count
            }
        })

    # Upsert in batches of 100 (Pinecone limit)
    upsert_batch_size = 100
    total_upserted = 0

    for i in range(0, len(vectors), upsert_batch_size):
        batch = vectors[i:i + upsert_batch_size]
        try:
            index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.info(f"Upserted {total_upserted}/{len(vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Upsert failed for batch starting at {i}: {e}")

    return total_upserted


def run_ingestion_pipeline(data_dir: str = "data/raw", strategy: str = "semantic"):
    """
    Full ingestion pipeline:
    Load filings → chunk → embed → upsert to Pinecone
    """
    from src.ingestion.loader import load_all_filings
    from src.ingestion.chunker import chunk_documents

    logger.info("Starting ingestion pipeline...")

    # Step 1 — Load
    documents = load_all_filings(data_dir)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # Step 2 — Chunk
    chunks = chunk_documents(documents, strategy=strategy, chunk_size=1000, overlap=200)
    logger.info(f"Total chunks to embed: {len(chunks)}")

    # Step 3 — Embed
    client = get_openai_client()
    index = get_pinecone_index()

    # Check existing vectors to avoid re-embedding
    stats = index.describe_index_stats()
    existing = stats.get("total_vector_count", 0)
    if existing > 0:
        logger.warning(f"Pinecone already has {existing} vectors. Proceeding will add duplicates.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            logger.info("Aborted.")
            return

    embeddings = embed_chunks(chunks, client)

    # Step 4 — Upsert
    total = upsert_to_pinecone(chunks, embeddings, index)
    logger.info(f"Ingestion complete. {total} vectors in Pinecone.")