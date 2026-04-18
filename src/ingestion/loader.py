import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a single loaded SEC filing document."""
    content: str
    ticker: str
    filing_type: str
    file_path: str
    char_count: int


def clean_html(raw_html: str) -> str:
    """Strip HTML tags from SEC filings — they come as raw HTML."""
    soup = BeautifulSoup(raw_html, "lxml")
    
    # Remove script and style tags entirely
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    
    text = soup.get_text(separator=" ")
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def load_filing(file_path: str, ticker: str, filing_type: str = "10-K") -> Optional[Document]:
    """
    Load and clean a single SEC filing from disk.
    Returns None if file is unreadable or too short to be useful.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        raw = path.read_text(encoding="utf-8", errors="ignore")

        # SEC filings come as HTML — clean them
        if "<html" in raw.lower() or "<body" in raw.lower():
            content = clean_html(raw)
        else:
            content = raw.strip()

        # Skip files that are too short to be meaningful
        if len(content) < 500:
            logger.warning(f"Skipping {file_path} — too short ({len(content)} chars)")
            return None

        logger.info(f"Loaded {ticker} {filing_type} — {len(content):,} chars")

        return Document(
            content=content,
            ticker=ticker,
            filing_type=filing_type,
            file_path=str(path),
            char_count=len(content)
        )

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def load_all_filings(data_dir: str = "data/raw") -> list[Document]:
    """
    Walk the data/raw directory and load all SEC filings.
    SEC Edgar downloader saves files as:
    data/raw/sec-edgar-filings/{TICKER}/{FILING_TYPE}/{accession}/primary-document.html
    """
    documents = []
    base = Path(data_dir) / "sec-edgar-filings"

    if not base.exists():
        logger.error(f"Data directory not found: {base}. Run the downloader first.")
        return []

    for ticker_dir in sorted(base.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        for filing_type_dir in ticker_dir.iterdir():
            if not filing_type_dir.is_dir():
                continue
            filing_type = filing_type_dir.name

            for accession_dir in filing_type_dir.iterdir():
                if not accession_dir.is_dir():
                    continue

                # Find the main document
                for candidate in ["primary-document.html", "primary-document.htm", "filing-details.html"]:
                    candidate_path = accession_dir / candidate
                    if candidate_path.exists():
                        doc = load_filing(str(candidate_path), ticker, filing_type)
                        if doc:
                            documents.append(doc)
                        break

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents