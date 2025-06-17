# embed_documents_by_namespace.py
import os
import json
import chromadb
from tqdm import tqdm
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Initialize LLM and embedder
llm = OllamaLLM(model="llama3.1:8b")
embedder = OllamaEmbeddings(model="mxbai-embed-large")

# Persistent ChromaDB client
client = chromadb.PersistentClient(path="chroma_multiple_2")

# Load and process all JSON files in Data_Json
data_dir = "Data_Json"

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Use filename (without .json) as collection name
        collection_name = os.path.splitext(filename)[0]
        collection = client.get_or_create_collection(name=collection_name)

        # Only embed if collection is empty
        if collection.count() == 0:
            for i, item in enumerate(tqdm(chunks, desc=f"Embedding chunks from {filename}")):
                content = item["content"]
                title = item.get("title", "")
                source_file = filename

                summary_prompt = (
                    f"Summarise the following, capturing as much information as you can. "
                    f"This will be used for retrieval so ensure that you capture all main points:\n\n{content}\n\nSummary:"
                )
                # summary = llm.invoke(summary_prompt)
                summary = content  # Bypass LLM for speed

                summary_embedding = embedder.embed_query(summary)

                metadata = {
                    "header": item.get("header", ""),
                    "subheader": item.get("subheader", ""),
                    "title": title,
                    "chunk_index": item.get("chunk_index", 0),
                    "summary": summary,
                    "raw_content": content,
                    "source_file": source_file
                }

                collection.add(
                    documents=[summary],
                    embeddings=[summary_embedding],
                    metadatas=[metadata],
                    ids=[f"{source_file}-chunk-{i}"]
                )

            print(f"✅ Embedded and stored {len(chunks)} chunks from {filename} in collection '{collection_name}'.\n")
        else:
            print(f"✅ Collection '{collection_name}' already populated. Skipping embedding.")
