import chromadb
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

class ChunkSimilarityAnalyzer:
    def __init__(self, chromadb_path: str = "chromadb"):
        """Initialize the analyzer with ChromaDB client and embeddings."""
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.embedder = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.collections = {col.name: col for col in self.client.list_collections()}
        print(f"Available collections: {list(self.collections.keys())}")

    def get_all_chunks(self, collection_name: str) -> Dict:
        """Get all chunks from a collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        result = collection.get()
        return result

    def search_with_scores(self, query: str, collection_name: str, n_results: int = 10) -> Dict:
        """Search for chunks and return with similarity scores."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        query_embedding = self.embedder.embed_query(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        
        return results

    def get_chunk_similarity(self, chunk_id: str, collection_name: str, target_query: str = None) -> Dict:
        """Get similarity score for a specific chunk against a query or all other chunks."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Get the specific chunk
        chunk_data = collection.get(ids=[chunk_id])
        if not chunk_data['ids']:
            raise ValueError(f"Chunk {chunk_id} not found")
        
        chunk_content = chunk_data['documents'][0]
        
        if target_query:
            # Calculate similarity with query
            query_embedding = self.embedder.embed_query(target_query)
            chunk_embedding = self.embedder.embed_documents([chunk_content])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            print(f"Chunk ID: {chunk_id}")
            print(f"Chunk Content: {chunk_content}")
            print(f"Query: {target_query}")
            print(f"Similarity Score: {similarity}")
            print(f"Metadata: {chunk_data['metadatas'][0]}")
            return {
                'chunk_id': chunk_id,
                'chunk_content': chunk_content,
                'query': target_query,
                'similarity_score': similarity,
                'metadata': chunk_data['metadatas'][0]
            }
        else:
            # Calculate similarity with all other chunks
            all_chunks = collection.get()
            similarities = []
            
            chunk_embedding = self.embedder.embed_documents([chunk_content])[0]
            
            for i, other_content in enumerate(all_chunks['documents']):
                if all_chunks['ids'][i] != chunk_id:
                    other_embedding = self.embedder.embed_documents([other_content])[0]
                    similarity = np.dot(chunk_embedding, other_embedding) / (
                        np.linalg.norm(chunk_embedding) * np.linalg.norm(other_embedding)
                    )
                    similarities.append({
                        'other_chunk_id': all_chunks['ids'][i],
                        'similarity_score': similarity,
                        'other_metadata': all_chunks['metadatas'][i]
                    })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                'chunk_id': chunk_id,
                'chunk_content': chunk_content,
                'metadata': chunk_data['metadatas'][0],
                'top_similar_chunks': similarities[:10]
            }

    def analyze_query_results(self, query: str, collection_name: str, show_all: bool = False) -> pd.DataFrame:
        """Analyze why certain chunks are/aren't selected for a query."""
        if show_all:
            # Get all chunks and their similarities
            all_chunks = self.get_all_chunks(collection_name)
            query_embedding = self.embedder.embed_query(query)
            
            similarities = []
            for i, content in enumerate(all_chunks['documents']):
                chunk_embedding = self.embedder.embed_documents([content])[0]
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                
                similarities.append({
                    'chunk_id': all_chunks['ids'][i],
                    'similarity_score': similarity,
                    'content_preview': content[:200] + "..." if len(content) > 200 else content,
                    'page_no': all_chunks['metadatas'][i].get('page_no', 'N/A'),
                    'section_title': all_chunks['metadatas'][i].get('section_title', 'N/A')
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return pd.DataFrame(similarities)
        else:
            # Get top search results
            results = self.search_with_scores(query, collection_name, n_results=20)
            
            data = []
            for i in range(len(results['ids'][0])):
                data.append({
                    'chunk_id': results['ids'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'content_preview': results['documents'][0][i][:200] + "..." if len(results['documents'][0][i]) > 200 else results['documents'][0][i],
                    'page_no': results['metadatas'][0][i].get('page_no', 'N/A'),
                    'section_title': results['metadatas'][0][i].get('section_title', 'N/A')
                })
            
            return pd.DataFrame(data)

    def compare_chunks(self, chunk_id1: str, chunk_id2: str, collection_name: str) -> Dict:
        """Compare similarity between two specific chunks."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Get both chunks
        chunk1_data = collection.get(ids=[chunk_id1])
        chunk2_data = collection.get(ids=[chunk_id2])
        
        if not chunk1_data['ids'] or not chunk2_data['ids']:
            raise ValueError("One or both chunks not found")
        
        chunk1_content = chunk1_data['documents'][0]
        chunk2_content = chunk2_data['documents'][0]
        
        # Calculate embeddings and similarity
        embeddings = self.embedder.embed_documents([chunk1_content, chunk2_content])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        return {
            'chunk1_id': chunk_id1,
            'chunk1_content': chunk1_content,
            'chunk1_metadata': chunk1_data['metadatas'][0],
            'chunk2_id': chunk_id2,
            'chunk2_content': chunk2_content,
            'chunk2_metadata': chunk2_data['metadatas'][0],
            'similarity_score': similarity
        }

    def find_least_similar_chunks(self, query: str, collection_name: str, n_results: int = 10) -> pd.DataFrame:
        """Find chunks with lowest similarity to query to understand why they weren't selected."""
        all_chunks = self.get_all_chunks(collection_name)
        query_embedding = self.embedder.embed_query(query)
        
        similarities = []
        for i, content in enumerate(all_chunks['documents']):
            chunk_embedding = self.embedder.embed_documents([content])[0]
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            
            similarities.append({
                'chunk_id': all_chunks['ids'][i],
                'similarity_score': similarity,
                'content_preview': content[:200] + "..." if len(content) > 200 else content,
                'page_no': all_chunks['metadatas'][i].get('page_no', 'N/A'),
                'section_title': all_chunks['metadatas'][i].get('section_title', 'N/A')
            })
        
        # Sort by similarity (ascending for least similar)
        similarities.sort(key=lambda x: x['similarity_score'])
        return pd.DataFrame(similarities[:n_results])

analyzer = ChunkSimilarityAnalyzer()

analyzer.get_chunk_similarity("TAL_AcceleratedProtection_2022-08-05l-chunk-110", "TAL_AcceleratedProtection_2022-08-05", "What are the critical illness conditions covered?")