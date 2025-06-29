# embed_documents_by_namespace.py
import os
import json
import chromadb
import time
from tqdm import tqdm
from langchain_ollama import OllamaEmbeddings

from dotenv import load_dotenv
load_dotenv()

# Initialize embedder - using Ollama for full local operation
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="chromadb")

# Rate limiting settings for local Ollama (much more generous than API limits)
REQUESTS_PER_MINUTE = 120  # Local models can handle higher rates
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # 0.5 second between requests

# Load and process all JSON/JSONL files in ocr/Data_Json directory
data_dir = "ocr/Data_Json"

print(f"🔍 Scanning directory: {data_dir}")
print(f"📊 Using embedding model: mxbai-embed-large (1024 dimensions)")
print(f"⚡ Rate limit: {REQUESTS_PER_MINUTE} requests/minute ({DELAY_BETWEEN_REQUESTS:.1f}s delay)\n")

for filename in os.listdir(data_dir):
    if filename.endswith("_chunks.json") or filename.endswith("_chunks.jsonl"):
        file_path = os.path.join(data_dir, filename)
        
        # Read chunks based on file format
        chunks = []
        if filename.endswith("_chunks.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        elif filename.endswith("_chunks.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num} in {filename}: {e}")
                            continue

        # Use filename (without _chunks.json or _chunks.jsonl) as collection name
        collection_name = filename.replace("_chunks.json", "").replace("_chunks.jsonl", "")
        collection = client.get_or_create_collection(name=collection_name)

        # Only embed if collection is empty
        if collection.count() == 0:
            print(f"🚀 Starting embedding for {len(chunks)} chunks from {filename}")
            print(f"⏱️  Rate limit: {REQUESTS_PER_MINUTE} requests/minute ({DELAY_BETWEEN_REQUESTS:.1f}s delay between requests)")
            
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding chunks from {filename}")):
                content = chunk.get("content")

                # Rate limiting: wait between requests (except for the first one)
                if i > 0:
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                # Generate simple chunk_id
                chunk_id = str(i)
                
                try:
                    # Use the content directly for embedding (no LLM summary for speed)
                    content_embedding = embeddings.embed_query(content)
                except Exception as e:
                    print(f"\n❌ Error embedding chunk {chunk_id}: {e}")
                    print("⏸️  Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    try:
                        content_embedding = embeddings.embed_query(content)
                        print(f"✅ Retry successful for chunk {chunk_id}")
                    except Exception as retry_error:
                        print(f"❌ Retry failed for chunk {chunk_id}: {retry_error}")
                        continue

                # Create metadata based on the actual JSONL structure
                metadata = {
                    "insurer": chunk.get("Insurer"),
                    "document_name": chunk.get("Document_Name"),
                    "document_date": chunk.get("Document_Date"),
                    "product_type": chunk.get("Product_type"),
                    "page_no": chunk.get("Page_no"),
                    "section_title": chunk.get("Section_Title"),
                    "subheading": chunk.get("Subheading"),
                    "content_label": chunk.get("content_label"),
                    "source_file": filename,
                    "chunk_id": chunk_id
                }

                collection.add(
                    documents=[content],
                    embeddings=[content_embedding],
                    metadatas=[metadata],
                    ids=[f"{collection_name}-chunk-{chunk_id}"]
                )

            print(f"✅ Embedded and stored {len(chunks)} chunks from {filename} in collection '{collection_name}'.\n")
        else:
            print(f"✅ Collection '{collection_name}' already populated. Skipping embedding.")

print("🎉 All document chunks have been processed and embedded!")
