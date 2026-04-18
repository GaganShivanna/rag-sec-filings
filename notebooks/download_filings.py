from dotenv import load_dotenv
load_dotenv()

from sec_edgar_downloader import Downloader

# Initialize — put your real name and email here
dl = Downloader("Gagan Shivanna", "gshivann@usc.edu", "data/raw")

# 5 companies, 3 years each = 15 filings total
# Enough to build and test the full pipeline
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

for ticker in tickers:
    print(f"Downloading {ticker} 10-K filings...")
    dl.get("10-K", ticker, limit=3)
    print(f"✓ {ticker} done")

print("\nAll filings downloaded. Check data/raw/sec-edgar-filings/")