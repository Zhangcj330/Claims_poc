# embed_documents_by_namespace.py
import os
import json
import chromadb
import time
from tqdm import tqdm
#from langchain_openai import OpenAIEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


from dotenv import load_dotenv
load_dotenv()

# Initialize embedder
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
embeddings =OllamaEmbeddings(model="mxbai-embed-large")
# Persistent ChromaDB client
client = chromadb.PersistentClient(path="chromadb")

# Rate limiting settings for OpenAI Embedding API (more generous limits)
REQUESTS_PER_MINUTE = 120  # OpenAI typically allows much higher rates
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # 1 second between requests

# Load and process all JSON/JSONL files in ocr/Data_Json directory
data_dir = "ocr/Data_Json/summarized"
#data_dir = "Data_Json"

# Choosie - change the file name from chunks to chunks_with_summaries
for filename in os.listdir(data_dir):
    if filename.endswith("chunks_with_summaries.json") or filename.endswith("chunks_with_summaries.jsonl"):
        file_path = os.path.join(data_dir, filename)
        
        # Read chunks based on file format
        chunks = []
        if filename.endswith("chunks_with_summaries.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        elif filename.endswith("chunks_with_summaries.jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            chunks.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON on line {line_num} in {filename}: {e}")
                            continue

        # Use filename (without chunks_with_summaries.json or chunks_with_summaries.jsonl) as collection name
        collection_name = filename.replace("_chunks_with_summaries.jsonl", "").replace("_chunks_with_summaries.json", "")
        collection = client.get_or_create_collection(name=collection_name)

        # Only embed if collection is empty
        if collection.count() == 0:
            print(f"🚀 Starting embedding for {len(chunks)} chunks from {filename}")
            print(f"⏱️  Rate limit: {REQUESTS_PER_MINUTE} requests/minute ({DELAY_BETWEEN_REQUESTS:.1f}s delay between requests)")
            
            for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding chunks from {filename}")):
                summary = chunk.get("ai_summary")
                content = chunk.get("content")
                # Rate limiting: wait between requests (except for the first one)
                if i > 0:
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                # Generate simple chunk_id
                chunk_id = str(i)
                
                try:
                    # Use the content directly for embedding (no LLM summary for speed)
                    summary_embedding = embeddings.embed_query(summary)
                except Exception as e:
                    print(f"\n❌ Error embedding chunk {chunk_id}: {e}")
                    print("⏸️  Waiting 60 seconds before retrying...")
                    time.sleep(60)
                    try:
                        summary_embedding = embeddings.embed_query(content)
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
                    "summary": summary,
                    "chunk_id": chunk_id
                }

                collection.add(
                    documents=[content],
                    embeddings=[summary_embedding],
                    metadatas=[metadata],
                    ids=[f"{collection_name}-chunk-{chunk_id}"]
                )

            print(f"✅ Embedded and stored {len(chunks)} chunks from {filename} in collection '{collection_name}'.\n")
        else:
            print(f"✅ Collection '{collection_name}' already populated. Skipping embedding.")

print("🎉 All document chunks have been processed and embedded!")
