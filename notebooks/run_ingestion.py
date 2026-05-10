from dotenv import load_dotenv
load_dotenv()

from src.ingestion.embedder import run_ingestion_pipeline

# This will embed all 4975 chunks and push to Pinecone
# Cost: ~$0.06 at text-embedding-3-large pricing
# Time: ~5-8 minutes
run_ingestion_pipeline(data_dir="data/raw", strategy="semantic")