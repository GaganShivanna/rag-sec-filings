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
    """Strip HTML tags and XBRL data from SEC filings."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove all non-narrative tags
    for tag in soup(["script", "style", "head",
                     "ix:header", "ix:hidden",
                     "xbrli:context", "xbrli:unit",
                     "xbrli:xbrl", "link:roletype"]):
        tag.decompose()

    # Unwrap inline XBRL tags but keep their text content
    for tag in soup.find_all(["ix:nonfraction", "ix:nonnumeric", "ix:continuation"]):
        tag.unwrap()

    text = soup.get_text(separator=" ")

    # Filter out lines that are pure XBRL artifacts
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are mostly XBRL namespace references
        if "xbrli:" in line or "iso4217:" in line or "us-gaap:" in line:
            continue
        if len(line) < 20:
            continue
        lines.append(line)

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

        # Find the 10-K narrative section
        doc_start = raw.upper().find("TYPE>10-K")
        if doc_start != -1:
            raw = raw[doc_start:]

        # Cut before exhibits start
        next_exhibit = raw.upper().find("TYPE>EX-")
        if next_exhibit != -1:
            raw = raw[:next_exhibit]

        # Find HTML start
        html_start = raw.lower().find("<html")
        if html_start != -1:
            raw = raw[html_start:]

        # Clean and extract text
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
    data/raw/sec-edgar-filings/{TICKER}/{FILING_TYPE}/{accession}/full-submission.txt
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

                # Handle both .txt and .html formats
                for candidate in ["full-submission.txt", "primary-document.html", "primary-document.htm"]:
                    candidate_path = accession_dir / candidate
                    if candidate_path.exists():
                        doc = load_filing(str(candidate_path), ticker, filing_type)
                        if doc:
                            documents.append(doc)
                        break

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents