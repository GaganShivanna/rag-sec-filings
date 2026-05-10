import logging
from dataclasses import dataclass
from typing import Generator
from src.ingestion.loader import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of text ready for embedding."""
    chunk_id: str        # unique id: AAPL_10-K_0_42
    content: str         # the actual text
    ticker: str
    filing_type: str
    chunk_index: int     # position within the document
    total_chunks: int    # total chunks in this document
    char_count: int


def fixed_chunker(
    doc: Document,
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[Chunk]:
    """
    Split document into fixed-size chunks with overlap.

    Why overlap? Without it, a sentence split across two chunks
    loses context on both sides. Overlap ensures continuity.

    chunk_size=1000 chars ~ 250 tokens (well within embedding limits)
    overlap=200 chars ~ 50 tokens of shared context between chunks
    """
    content = doc.content
    chunks = []
    start = 0
    index = 0

    while start < len(content):
        end = start + chunk_size

        # Don't cut mid-word — extend to next space
        if end < len(content):
            next_space = content.find(" ", end)
            if next_space != -1 and next_space - end < 100:
                end = next_space

        chunk_text = content[start:end].strip()

        if len(chunk_text) > 100:  # skip tiny leftover chunks
            chunks.append(Chunk(
                chunk_id=f"{doc.ticker}_{doc.filing_type}_{hash(doc.file_path) % 10000}_{index}",
                content=chunk_text,
                ticker=doc.ticker,
                filing_type=doc.filing_type,
                chunk_index=index,
                total_chunks=0,  # filled in below
                char_count=len(chunk_text)
            ))
            index += 1

        start = end - overlap  # step back by overlap for next chunk

    # Now we know total — fill it in
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def semantic_chunker(
    doc: Document,
    max_chunk_size: int = 1200,
    min_chunk_size: int = 200
) -> list[Chunk]:
    """
    Split document on natural sentence boundaries.

    Why this matters: Fixed chunking can cut mid-sentence.
    Semantic chunking respects sentence structure, producing
    cleaner, more coherent chunks for the LLM to reason over.

    Strategy: accumulate sentences until we hit max_chunk_size,
    then start a new chunk at the next sentence boundary.
    """
    import re
    
    # Split on sentence-ending punctuation followed by whitespace
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(doc.content)

    chunks = []
    current = []
    current_len = 0
    index = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current_len + len(sentence) > max_chunk_size and current_len >= min_chunk_size:
            # Flush current chunk
            chunk_text = " ".join(current)
            chunks.append(Chunk(
                chunk_id=f"{doc.ticker}_{doc.filing_type}_{hash(doc.file_path) % 10000}_s{index}",
                content=chunk_text,
                ticker=doc.ticker,
                filing_type=doc.filing_type,
                chunk_index=index,
                total_chunks=0,
                char_count=len(chunk_text)
            ))
            index += 1
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)

    # Flush remaining
    if current:
        chunk_text = " ".join(current)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(Chunk(
                chunk_id=f"{doc.ticker}_{doc.filing_type}_{hash(doc.file_path) % 10000}_s{index}",
                content=chunk_text,
                ticker=doc.ticker,
                filing_type=doc.filing_type,
                chunk_index=index,
                total_chunks=0,
                char_count=len(chunk_text)
            ))

    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    return chunks


def chunk_documents(
    documents: list[Document],
    strategy: str = "fixed",
    chunk_size: int = 1000,
    overlap: int = 200
) -> list[Chunk]:
    """
    Chunk all documents using the specified strategy.
    strategy: "fixed" or "semantic"
    """
    all_chunks = []

    for doc in documents:
        if strategy == "fixed":
            chunks = fixed_chunker(doc, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = semantic_chunker(doc, max_chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed' or 'semantic'")

        all_chunks.extend(chunks)
        logger.info(f"{doc.ticker} {doc.filing_type} → {len(chunks)} chunks ({strategy})")

    logger.info(f"Total chunks: {len(all_chunks)} from {len(documents)} documents")
    return all_chunks