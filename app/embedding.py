# app/embedding.py

import vertexai
from vertexai.language_models import TextEmbeddingModel
import json
import os
import time
from typing import List, Dict, Any

# --- Configuration ---
# Your project and location are now read from environment variables
PROJECT_ID = os.environ.get("PROJECT_ID", "vertex-ai-co-pilot")
LOCATION = os.environ.get("REGION", "europe-west4")

# The name of the embedding model to use.
EMBEDDING_MODEL_NAME = "text-embedding-004"

# Names of the input and output files.
INGESTED_DATA_FILE = "ingested_data.jsonl"
EMBEDDED_DATA_FILE = "embedded_data.jsonl"

# The batch size for API requests to avoid rate limits.
# A small batch size is good for testing and avoiding API quotas.
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 5))

def get_embeddings_with_retry(model: TextEmbeddingModel, texts: List[str]) -> List[Any]:
    """
    Generates embeddings with exponential backoff for rate limit handling.
    This function is crucial for robust API calls.
    """
    delay = 1  # Initial delay in seconds
    for _ in range(5):  # Retry up to 5 times
        try:
            return model.get_embeddings(texts)
        except Exception as e:
            print(f"Embedding request failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Double the delay for the next retry
    raise Exception("Failed to get embeddings after multiple retries.")

def generate_embeddings() -> None:
    """
    Reads chunks from the ingested data file, generates embeddings for them
    in batches, and saves the results to a new JSONL file.
    This approach is memory-efficient for large datasets.
    """
    # Initialize the Vertex AI SDK.
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Load the pre-trained embedding model.
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    
    print(f"Starting embedding generation using model: {EMBEDDING_MODEL_NAME}")
    
    with open(INGESTED_DATA_FILE, 'r') as infile, open(EMBEDDED_DATA_FILE, 'w') as outfile:
        batch = []
        for line in infile:
            item = json.loads(line)
            batch.append(item)
            
            # When the batch is full, process it.
            if len(batch) >= EMBEDDING_BATCH_SIZE:
                texts = [item["text"] for item in batch]
                try:
                    embeddings = get_embeddings_with_retry(embedding_model, texts)
                    
                    # Add the embedding to each item and write to the output file.
                    for j, item in enumerate(batch):
                        item["embedding"] = embeddings[j].values
                        outfile.write(json.dumps(item) + '\n')
                
                except Exception as e:
                    print(f"Error processing a batch: {e}. Skipping this batch.")
                
                batch = []  # Reset batch after processing

        # Process any remaining items in the last batch.
        if batch:
            texts = [item["text"] for item in batch]
            try:
                embeddings = get_embeddings_with_retry(embedding_model, texts)
                for j, item in enumerate(batch):
                    item["embedding"] = embeddings[j].values
                    outfile.write(json.dumps(item) + '\n')
            except Exception as e:
                print(f"Error processing the final batch: {e}. Skipping this batch.")
            
    print(f"Embedding complete. '{EMBEDDED_DATA_FILE}' created.")

if __name__ == "__main__":
    if not os.path.exists(INGESTED_DATA_FILE):
        print(f"Error: The file '{INGESTED_DATA_FILE}' was not found. Please run data_ingestion.py first.")
    else:
        generate_embeddings()