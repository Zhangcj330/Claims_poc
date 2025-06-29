from retrieval import ChunkSimilarityAnalyzer

def main():
    # Initialize analyzer
    analyzer = ChunkSimilarityAnalyzer()
    collection_name = "TAL_AcceleratedProtection_2022-08-05"
    
    print("=== ChromaDB Chunk Similarity Analysis Demo ===\n")
    
    # Example 1: Analyze query results
    query = "What are the critical illness conditions covered?"
    print(f"Query: {query}")
    print("\n1. TOP SEARCH RESULTS:")
    results_df = analyzer.analyze_query_results(query, collection_name)
    print(results_df[['chunk_id', 'similarity_score', 'section_title', 'page_no']].head())
    
    # Example 2: Specific chunk similarity
    chunk_id = results_df.iloc[0]['chunk_id']  # Use the top result
    print(f"\n2. SPECIFIC CHUNK ANALYSIS ({chunk_id}):")
    chunk_analysis = analyzer.get_chunk_similarity(chunk_id, collection_name, query)
    print(f"Similarity Score: {chunk_analysis['similarity_score']:.4f}")
    print(f"Page: {chunk_analysis['metadata'].get('page_no', 'N/A')}")
    print(f"Section: {chunk_analysis['metadata'].get('section_title', 'N/A')}")
    print(f"Content: {chunk_analysis['chunk_content'][:300]}...")
    
    # Example 3: Compare two chunks
    if len(results_df) > 1:
        chunk1_id = results_df.iloc[0]['chunk_id']
        chunk2_id = results_df.iloc[1]['chunk_id']
        print(f"\n3. CHUNK COMPARISON:")
        comparison = analyzer.compare_chunks(chunk1_id, chunk2_id, collection_name)
        print(f"Similarity between chunks: {comparison['similarity_score']:.4f}")
        print(f"Chunk 1 - Page: {comparison['chunk1_metadata'].get('page_no')}, Section: {comparison['chunk1_metadata'].get('section_title')}")
        print(f"Chunk 2 - Page: {comparison['chunk2_metadata'].get('page_no')}, Section: {comparison['chunk2_metadata'].get('section_title')}")
    
    # Example 4: Why chunks weren't selected (least similar)
    print(f"\n4. LEAST SIMILAR CHUNKS (Why not selected):")
    least_similar = analyzer.find_least_similar_chunks(query, collection_name, n_results=3)
    print(least_similar[['chunk_id', 'similarity_score', 'section_title', 'page_no']])
    
    # Example 5: Get all similarities (if you want to see everything)
    print(f"\n5. ALL CHUNKS SIMILARITY DISTRIBUTION:")
    all_results = analyzer.analyze_query_results(query, collection_name, show_all=True)
    print(f"Total chunks: {len(all_results)}")
    print(f"Highest similarity: {all_results['similarity_score'].max():.4f}")
    print(f"Lowest similarity: {all_results['similarity_score'].min():.4f}")
    print(f"Average similarity: {all_results['similarity_score'].mean():.4f}")
    
    # Show distribution
    print("\nSimilarity score ranges:")
    print(f"Very high (>0.3): {(all_results['similarity_score'] > 0.3).sum()} chunks")
    print(f"High (0.2-0.3): {((all_results['similarity_score'] >= 0.2) & (all_results['similarity_score'] <= 0.3)).sum()} chunks")
    print(f"Medium (0.1-0.2): {((all_results['similarity_score'] >= 0.1) & (all_results['similarity_score'] < 0.2)).sum()} chunks")
    print(f"Low (<0.1): {(all_results['similarity_score'] < 0.1).sum()} chunks")

if __name__ == "__main__":
    main() 