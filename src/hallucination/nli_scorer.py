import os
import logging
from dotenv import load_dotenv
from transformers import pipeline
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NLI Labels from DeBERTa model
ENTAILMENT = "entailment"       # context SUPPORTS the claim
CONTRADICTION = "contradiction" # context CONTRADICTS the claim
NEUTRAL = "neutral"             # context neither supports nor contradicts

# Confidence threshold — below this we refuse to answer
CONFIDENCE_THRESHOLD = 0.65


def load_nli_model():
    """
    Load DeBERTa NLI model locally.
    First run downloads ~400MB — subsequent runs load from cache.
    
    Why DeBERTa over BERT?
    DeBERTa uses disentangled attention — separately models content
    and position of tokens. This gives significantly better performance
    on NLI tasks, especially for longer financial sentences.
    """
    logger.info("Loading DeBERTa NLI model (first run downloads ~400MB)...")
    nli_pipeline = pipeline(
        "text-classification",
        model="cross-encoder/nli-deberta-v3-base",
        device=-1  # CPU — use 0 for GPU if available
    )
    logger.info("NLI model loaded successfully")
    return nli_pipeline


def score_claim_against_context(
    claim: str,
    context: str,
    nli_pipeline
) -> dict:
    """
    Score a single claim against a context chunk using NLI.
    
    NLI (Natural Language Inference) answers:
    Given this context, does this claim follow? (entailment)
    Given this context, is this claim false? (contradiction)
    Given this context, can we tell either way? (neutral)
    
    Example:
    Context: "NVIDIA revenue was $130.5 billion in fiscal 2025"
    Claim: "NVIDIA revenue exceeded $100 billion"
    → ENTAILMENT (supported)
    
    Context: "NVIDIA revenue was $130.5 billion in fiscal 2025"  
    Claim: "NVIDIA revenue was $200 billion"
    → CONTRADICTION
    
    Context: "NVIDIA has strong data center demand"
    Claim: "NVIDIA's CEO is Jensen Huang"
    → NEUTRAL (not verifiable from context)
    """
    try:
        # DeBERTa NLI takes premise + hypothesis
        result = nli_pipeline(
            {"text": context, "text_pair": claim},
            truncation=True,
            max_length=512
        )

        label = result[0]["label"].lower()
        score = result[0]["score"]

        # Map to our standard labels
        if label == "entailment":
            verdict = "SUPPORTED"
        elif label == "contradiction":
            verdict = "CONTRADICTED"
        else:
            verdict = "UNVERIFIABLE"

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": score,
            "raw_label": label
        }

    except Exception as e:
        logger.error(f"NLI scoring failed: {e}")
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "raw_label": "error"
        }


def score_claim_against_all_chunks(
    claim: str,
    chunks: list[dict],
    nli_pipeline
) -> dict:
    """
    Score a claim against ALL retrieved chunks and return
    the best (most confident) verdict.
    
    Why score against all chunks?
    A claim might be supported by chunk 3 but not chunk 1.
    We want the most charitable interpretation — if ANY chunk
    supports the claim, it should be marked SUPPORTED.
    
    Priority order: SUPPORTED > CONTRADICTED > UNVERIFIABLE
    """
    best_supported = None
    best_contradiction = None
    best_neutral = None

    for chunk in chunks:
        result = score_claim_against_context(
            claim,
            chunk["content"][:500],  # truncate for speed
            nli_pipeline
        )

        if result["verdict"] == "SUPPORTED":
            if best_supported is None or result["confidence"] > best_supported["confidence"]:
                best_supported = result
                best_supported["supporting_chunk"] = chunk["ticker"]

        elif result["verdict"] == "CONTRADICTED":
            if best_contradiction is None or result["confidence"] > best_contradiction["confidence"]:
                best_contradiction = result

        else:
            if best_neutral is None or result["confidence"] > best_neutral["confidence"]:
                best_neutral = result

    # Return best result in priority order
    if best_supported:
        return best_supported
    elif best_contradiction:
        return best_contradiction
    else:
        return best_neutral or {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "raw_label": "neutral"
        }


def split_into_claims(answer: str) -> list[str]:
    """
    Split a generated answer into individual verifiable claims.
    
    Why split?
    GPT-4 generates multi-sentence answers. Each sentence makes
    a different claim. We need to verify each one independently
    because a single hallucinated sentence can corrupt an otherwise
    grounded answer.
    
    Simple approach: split on sentence boundaries.
    Production approach: use an LLM to extract atomic claims.
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    # Filter out very short fragments and source citations
    claims = []
    for s in sentences:
        s = s.strip()
        # Skip very short sentences and pure citation lines
        if len(s) < 30:
            continue
        if s.startswith("Sources:") or s.startswith("[Source"):
            continue
        claims.append(s)

    return claims


def score_answer(
    answer: str,
    chunks: list[dict],
    nli_pipeline
) -> dict:
    """
    Score an entire generated answer for hallucination.
    
    Returns:
    - per-claim verdicts (supported/contradicted/unverifiable)
    - aggregate confidence score
    - overall verdict (PASS / WARN / FAIL)
    - whether the answer should be shown or refused
    
    Thresholds:
    PASS  — >80% claims supported, no contradictions
    WARN  — mixed support, show with warning
    FAIL  — contradictions found or <50% supported → refuse
    """
    claims = split_into_claims(answer)

    if not claims:
        return {
            "verdict": "PASS",
            "confidence": 1.0,
            "claim_results": [],
            "should_refuse": False,
            "reason": "No verifiable claims found"
        }

    logger.info(f"Scoring {len(claims)} claims against {len(chunks)} chunks...")

    claim_results = []
    for claim in claims:
        result = score_claim_against_all_chunks(claim, chunks, nli_pipeline)
        claim_results.append(result)
        logger.info(f"  [{result['verdict']}] ({result['confidence']:.2f}) {claim[:80]}...")

    # Compute aggregate stats
    total = len(claim_results)
    supported = sum(1 for r in claim_results if r["verdict"] == "SUPPORTED")
    contradicted = sum(1 for r in claim_results if r["verdict"] == "CONTRADICTED")
    unverifiable = sum(1 for r in claim_results if r["verdict"] == "UNVERIFIABLE")

    support_rate = supported / total if total > 0 else 0
    avg_confidence = sum(r["confidence"] for r in claim_results) / total

    # Determine overall verdict
    if contradicted > 0:
        verdict = "FAIL"
        should_refuse = True
        reason = f"Found {contradicted} contradicted claim(s)"
    elif support_rate >= 0.8:
        verdict = "PASS"
        should_refuse = False
        reason = f"{supported}/{total} claims supported"
    elif support_rate >= 0.5:
        verdict = "WARN"
        should_refuse = False
        reason = f"Only {supported}/{total} claims verified"
    else:
        verdict = "FAIL"
        should_refuse = True
        reason = f"Insufficient support: {supported}/{total} claims verified"

    return {
        "verdict": verdict,
        "confidence": avg_confidence,
        "support_rate": support_rate,
        "claim_results": claim_results,
        "should_refuse": should_refuse,
        "reason": reason,
        "stats": {
            "total_claims": total,
            "supported": supported,
            "contradicted": contradicted,
            "unverifiable": unverifiable
        }
    }