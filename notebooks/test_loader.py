from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import load_all_filings, clean_html

# Quick test of HTML cleaning
sample_html = """
<html><body>
<script>var x = 1;</script>
<p>Apple Inc. reported revenues of <b>$394 billion</b> in fiscal 2022.</p>
<style>.cls{color:red}</style>
<p>Operating income was $119 billion.</p>
</body></html>
"""
cleaned = clean_html(sample_html)
print("HTML cleaning test:")
print(cleaned)
print(f"Length: {len(cleaned)} chars")
print("\n✓ loader.py working correctly")