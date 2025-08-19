# app/data_ingestion.py

import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
import json
import os
import uuid

# --- Configuration ---
URLS_TO_INGEST = [
    "https://github.com/GoogleCloudPlatform/agent-starter-pack",
    "https://googlecloudplatform.github.io/agent-starter-pack/",
    "https://github.com/Intelligent-Internet/ii-agent",
    "https://google.github.io/adk-docs/",
    "https://cloud.google.com/products/agent-builder",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview",
    "https://agent2agent.ren/",
    "https://codelabs.developers.google.com/devsite/codelabs/building-ai-agents-vertexai",
    "https://medium.com/google-cloud/what-are-ai-agents-and-how-can-you-build-them-on-google-cloud-2f9d81e68f70",
    "https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf",
    "https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf",
    "https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf",
    "https://cdn.openai.com/business-guides-and-resources/identifying-and-scaling-ai-use-cases.pdf",
    "https://www.anthropic.com/engineering/claude-code-best-practices",
    "https://assets.anthropic.com/m/66daaa23018ab0fd/original/Anthropic-enterprise-ebook-digital.pdf",
    "https://www.anthropic.com/engineering/building-effective-agents",
    "https://www.kaggle.com/whitepaper-agent-companion?",
    "https://www.kaggle.com/whitepaper-prompt-engineering",
    "https://cloud.google.com/blog/topics/training-and-development/rag-best-practices",
    "https://medium.com/@garg-shelvi/vertex-ai-vector-search-rag-pipelines-85b75118aae2",
    "https://cloud.google.com/vertex-ai/docs/start/introduction-mlops",
    "https://medium.com/@vinaydevarasetty/a-practical-guide-to-mlops-for-generative-ai-using-google-cloud-vertex-ai-a1f81ff29add",
    "https://www.cloudsufi.com/building-scalable-ai-pipelines-with-vertex-ai-on-google-cloud/",
    "https://cloud.google.com/blog/products/identity-security/cloud-ciso-perspectives-how-google-secures-ai-agents",
    "https://bgiri-gcloud.medium.com/how-to-secure-your-ai-agents-fb33a8f901ba",
    "https://medium.com/@harrissolangi/the-enterprise-guide-to-scaling-ai-agents-on-google-cloud-platform-77f25a324bb5",
    "https://google.github.io/adk-docs/get-started/quickstart/",
    "https://www.youtube.com/watch?v=8rlNdKywldQ" # Added this URL
]

# Chunking parameters are now read from environment variables
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
INGESTED_DATA_FILE = "ingested_data.jsonl"

def get_youtube_transcript(url: str):
    """Fetches the transcript from a YouTube video URL."""
    try:
        video_id = url.split("v=")[1]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error getting YouTube transcript from {url}: {e}")
        return None

def scrape_web_page(url: str):
    """Scrapes text content from a standard web page URL."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()

        text = " ".join(soup.stripped_strings)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def get_github_repo_content(url: str):
    """Placeholder for GitHub repo content. This is more complex."""
    print(f"Skipping GitHub repository for now. We will handle this separately.")
    return None

def ingest_urls(urls: list):
    """
    Ingests content from a list of URLs and returns chunks with metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    with open(INGESTED_DATA_FILE, "w") as outfile:
        for url in urls:
            print(f"Ingesting content from: {url}")
            domain = urlparse(url).netloc
            content = None

            if "youtube.com" in domain:
                content = get_youtube_transcript(url)
            elif "github.com" in domain:
                content = get_github_repo_content(url)
            else:
                content = scrape_web_page(url)

            if content:
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    chunk_with_metadata = {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "source": url,
                        "title": domain
                    }
                    outfile.write(json.dumps(chunk_with_metadata) + '\n')
                print(f"  - Ingested {len(chunks)} chunks.")
    
    print(f"\nSuccessfully ingested content and saved to {INGESTED_DATA_FILE}.")

if __name__ == "__main__":
    ingest_urls(URLS_TO_INGEST)