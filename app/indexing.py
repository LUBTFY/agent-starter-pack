# app/indexing.py

import vertexai
from google.cloud import aiplatform, storage
import os
import time

# --- Configuration ---
PROJECT_ID = os.environ.get("PROJECT_ID", "vertex-ai-co-pilot")
LOCATION = os.environ.get("REGION", "europe-west4")

EMBEDDED_DATA_FILE = os.environ.get("EMBEDDED_DATA_FILE", "embedded_data.jsonl")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-vector-search-data")
GCS_UPLOAD_FOLDER = os.environ.get("GCS_UPLOAD_FOLDER", "vector_search/embedded_chunks")

INDEX_DISPLAY_NAME = os.environ.get("INDEX_DISPLAY_NAME", "my_rag_index")
ENDPOINT_DISPLAY_NAME = os.environ.get("ENDPOINT_DISPLAY_NAME", "my_rag_endpoint")

EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))

def upload_data_to_gcs(source_file: str, bucket_name: str, destination_folder: str) -> str:
    """
    Uploads a local file to GCS and returns the GCS URI.
    It also creates the bucket if it doesn't already exist.
    """
    storage_client = storage.Client(project=PROJECT_ID)
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"Bucket '{bucket_name}' not found. Creating it...")
        bucket = storage_client.create_bucket(bucket_name, location=LOCATION)

    destination_blob_name = os.path.join(destination_folder, os.path.basename(source_file))
    blob = bucket.blob(destination_blob_name)
    
    if blob.exists():
        print(f"File {destination_blob_name} already exists in GCS. Skipping upload.")
    else:
        print(f"Uploading {source_file} to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_filename(source_file)
        print("Upload complete.")
    
    return f"gs://{bucket_name}/{destination_blob_name}"

def create_and_deploy_index(gcs_uri: str) -> aiplatform.matching_engine.MatchingEngineIndexEndpoint:
    """
    Creates and deploys a Vertex AI Vector Search index, reusing existing resources.
    """
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # 1. Check for an existing Index
    my_index = None
    for index in aiplatform.matching_engine.MatchingEngineIndex.list():
        if index.display_name == INDEX_DISPLAY_NAME:
            print(f"Found existing index: '{INDEX_DISPLAY_NAME}'.")
            my_index = index
            break
            
    # 2. If Index exists, check if it's already deployed
    if my_index and my_index.deployed_indexes:
        print(f"Index '{INDEX_DISPLAY_NAME}' is already deployed. Returning existing endpoint.")
        endpoint_resource_name = my_index.deployed_indexes[0].index_endpoint
        return aiplatform.matching_engine.MatchingEngineIndexEndpoint(endpoint_resource_name)

    # 3. If Index does not exist, create it
    if not my_index:
        print(f"Creating a new index with display name: '{INDEX_DISPLAY_NAME}'...")
        
        # This is the corrected line: use create_tree_ah_index() with direct parameters.
        my_index = aiplatform.matching_engine.MatchingEngineIndex.create_tree_ah_index(
            display_name=INDEX_DISPLAY_NAME,
            contents_delta_uri=gcs_uri,
            dimensions=EMBEDDING_DIMENSIONS,
            approximate_neighbors_count=150,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
        )
        print("Index creation job sent. Waiting for completion...")
        my_index.wait()
        print(f"Index '{INDEX_DISPLAY_NAME}' created successfully.")

    # 4. Check for an existing Endpoint
    my_endpoint = None
    for endpoint in aiplatform.matching_engine.MatchingEngineIndexEndpoint.list():
        if endpoint.display_name == ENDPOINT_DISPLAY_NAME:
            print(f"Found existing endpoint '{ENDPOINT_DISPLAY_NAME}'. Re-using it.")
            my_endpoint = endpoint
            break
            
    # 5. If Endpoint does not exist, create it
    if not my_endpoint:
        print(f"Creating a new endpoint with display name: '{ENDPOINT_DISPLAY_NAME}'...")
        my_endpoint = aiplatform.matching_engine.MatchingEngineIndexEndpoint.create(
            display_name=ENDPOINT_DISPLAY_NAME,
            public_endpoint_enabled=True
        )
    
    # 6. Deploy the Index to the Endpoint
    print(f"Deploying index to endpoint '{ENDPOINT_DISPLAY_NAME}'...")
    deployed_index_id = f"deployed_{INDEX_DISPLAY_NAME.replace('_', '')}_{int(time.time())}"
    
    my_endpoint.deploy_index(
        index=my_index,
        deployed_index_id=deployed_index_id,
        machine_type="n1-standard-16",
        min_replica_count=1,
        max_replica_count=2,
    )
    print("Index deployment complete.")
    
    return my_endpoint

if __name__ == "__main__":
    if not os.path.exists(EMBEDDED_DATA_FILE):
        print(f"❌ Error: The file '{EMBEDDED_DATA_FILE}' was not found. Please run embedding.py first.")
    else:
        gcs_data_path = upload_data_to_gcs(
            source_file=EMBEDDED_DATA_FILE,
            bucket_name=GCS_BUCKET_NAME,
            destination_folder=GCS_UPLOAD_FOLDER
        )

        index_endpoint = create_and_deploy_index(gcs_data_path)
        print(f"✅ Index endpoint ready: {index_endpoint.resource_name}")
        if index_endpoint.public_endpoint_domain_name:
            print(f"🌍 Public endpoint domain: {index_endpoint.public_endpoint_domain_name}")