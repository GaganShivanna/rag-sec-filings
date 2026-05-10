import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from src.retrieval.query_engine import retrieve

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-4o-mini"  # fast and cheap for dev, swap to gpt-4o for prod


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the prompt for GPT-4.
    
    Key design decisions:
    1. System prompt instructs model to ONLY use provided context
    2. Each chunk is labeled with its source (ticker + chunk index)
       so the model can cite sources in its answer
    3. Explicit instruction to say "I don't have enough information"
       rather than hallucinate — this is the first line of defense
       before our NLI hallucination detector kicks in
    """
    # Format retrieved chunks as numbered context blocks
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1} | {chunk['ticker']} 10-K | Relevance: {chunk['score']:.3f}]\n"
            f"{chunk['content']}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = """You are a financial analyst assistant specializing in SEC filings.

Answer the user's question using ONLY the provided context from SEC 10-K filings.

Rules:
- Base every claim strictly on the provided context
- Cite sources by referencing [Source N] for each claim you make
- If the context does not contain enough information to answer, say exactly:
  "I don't have sufficient information in the provided filings to answer this."
- Never speculate or use knowledge outside the provided context
- Be precise with numbers — quote exact figures from the filings"""

    user_prompt = f"""Context from SEC Filings:
{context}

Question: {query}

Answer based strictly on the context above, citing sources:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def generate_answer(
    query: str,
    chunks: list[dict],
    client: OpenAI
) -> dict:
    """
    Generate a grounded answer from retrieved chunks.
    Returns the answer text and the chunks used as context.
    """
    if not chunks:
        return {
            "answer": "I don't have sufficient information in the provided filings to answer this.",
            "chunks_used": [],
            "model": LLM_MODEL
        }

    messages = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,  # low temp = more factual, less creative
        max_tokens=1000
    )

    answer = response.choices[0].message.content

    logger.info(f"Generated answer ({len(answer)} chars) using {len(chunks)} chunks")

    return {
        "answer": answer,
        "chunks_used": chunks,
        "model": LLM_MODEL,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
    }


def query_pipeline(
    query: str,
    top_k: int = 8,
    ticker_filter: str = None
) -> dict:
    """
    Full query pipeline — retrieval + generation.
    This is the main function the API will call.
    
    Flow:
    query → embed → Pinecone search → MMR rerank → GPT-4 → answer
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1 — retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k, ticker_filter=ticker_filter)

    if not chunks:
        return {
            "answer": "No relevant information found in the SEC filings.",
            "chunks_used": [],
            "sources": []
        }

    # Step 2 — generate grounded answer
    result = generate_answer(query, chunks, client)

    # Step 3 — format sources for response
    sources = list(set([
        f"{c['ticker']} 10-K" for c in chunks
    ]))

    result["sources"] = sources
    result["query"] = query

    return result