from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import os
import json

def process_document_with_chunking(source_path, chunk_size=512, chunk_overlap=100):
    """
    Process document: conversion + chunking
    
    Args:
        source_path: Input document path
        chunk_size: Target chunk size (in tokens)
        chunk_overlap: Chunk overlap size
    
    Returns:
        dict: Contains original document, chunks and other info
    """
    
    print(f"Processing document: {source_path}")
    
    # Step 1: Document conversion
    converter = DocumentConverter()
    result = converter.convert(source_path)
    
    # Step 2: Get document object and basic info
    doc = result.document
    markdown_content = doc.export_to_markdown()
    
    print(f"Document conversion completed, {len(markdown_content)} characters total")
    
    # Step 3: Initialize HybridChunker
    chunker = HybridChunker(
        chunk_size=chunk_size,
        overlap_size=chunk_overlap,
        split_by_page=True,  # Split by page boundaries
        respect_section_boundaries=True  # Respect section boundaries
    )
    
    # Step 4: Chunk the document
    chunks = chunker.chunk(doc)
    
    print(f"Chunking completed, generated {len(chunks)} chunks")
    
    # Step 5: Process and analyze chunks
    chunk_info = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text.strip()
        chunk_data = {
            'id': i,
            'text': chunk_text,
            'length': len(chunk_text),
            'page': getattr(chunk, 'page', None),
            'section': getattr(chunk, 'section', None),
            'metadata': {
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
                'has_tables': '|' in chunk_text or 'Table' in chunk_text,
                'has_lists': any(line.strip().startswith(('-', '*', '•')) for line in chunk_text.split('\n'))
            }
        }
        chunk_info.append(chunk_data)
    
    # Step 6: Generate output filenames
    base_name = os.path.splitext(source_path)[0]
    
    # Save original markdown
    markdown_file = f"{base_name}_converted.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # Save chunking results as JSON
    chunks_file = f"{base_name}_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_info, f, ensure_ascii=False, indent=2)
    
    # Save readable chunks file
    readable_chunks_file = f"{base_name}_chunks_readable.txt"
    with open(readable_chunks_file, 'w', encoding='utf-8') as f:
        f.write(f"Document Chunking Results\n{'='*50}\n\n")
        f.write(f"Original document: {source_path}\n")
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write(f"Average chunk size: {sum(c['length'] for c in chunk_info) / len(chunk_info):.0f} characters\n\n")
        
        for chunk_data in chunk_info:
            f.write(f"--- Chunk {chunk_data['id'] + 1} ---\n")
            f.write(f"Length: {chunk_data['length']} characters, {chunk_data['metadata']['word_count']} words\n")
            if chunk_data['page']:
                f.write(f"Page: {chunk_data['page']}\n")
            f.write(f"Content:\n{chunk_data['text']}\n\n")
    
    # Step 7: Print statistics
    print(f"\nDocument processing completed!")
    print(f"├── Original document: {markdown_file}")
    print(f"├── Chunks JSON: {chunks_file}")
    print(f"└── Readable chunks: {readable_chunks_file}")
    
    print(f"\nChunking statistics:")
    print(f"├── Total chunks: {len(chunks)}")
    print(f"├── Average chunk size: {sum(c['length'] for c in chunk_info) / len(chunk_info):.0f} characters")
    print(f"├── Largest chunk: {max(c['length'] for c in chunk_info)} characters")
    print(f"├── Smallest chunk: {min(c['length'] for c in chunk_info)} characters")
    print(f"├── Chunks with tables: {sum(1 for c in chunk_info if c['metadata']['has_tables'])}")
    print(f"└── Chunks with lists: {sum(1 for c in chunk_info if c['metadata']['has_lists'])}")
    
    return {
        'document': doc,
        'chunks': chunks,
        'chunk_info': chunk_info,
        'files': {
            'markdown': markdown_file,
            'chunks_json': chunks_file,
            'chunks_readable': readable_chunks_file
        }
    }

def analyze_chunks_quality(chunk_info):
    """Analyze chunk quality"""
    print(f"\nChunk quality analysis:")
    print(f"{'='*40}")
    
    lengths = [c['length'] for c in chunk_info]
    
    print(f"Size distribution:")
    print(f"├── Average: {sum(lengths) / len(lengths):.0f} characters")
    print(f"├── Median: {sorted(lengths)[len(lengths)//2]} characters")
    print(f"├── Standard deviation: {(sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths))**0.5:.0f}")
    
    # Check for chunks that are too small or too large
    too_small = [c for c in chunk_info if c['length'] < 100]
    too_large = [c for c in chunk_info if c['length'] > 1000]
    
    if too_small:
        print(f"├── Too small chunks ({len(too_small)}): may need to adjust chunk_size")
    if too_large:
        print(f"├── Too large chunks ({len(too_large)}): may need to adjust splitting strategy")
    
    # Content type analysis
    table_chunks = sum(1 for c in chunk_info if c['metadata']['has_tables'])
    list_chunks = sum(1 for c in chunk_info if c['metadata']['has_lists'])
    
    print(f"├── Table chunks: {table_chunks} ({table_chunks/len(chunk_info)*100:.1f}%)")
    print(f"└── List chunks: {list_chunks} ({list_chunks/len(chunk_info)*100:.1f}%)")

if __name__ == "__main__":
    # Example usage
    source_file = "TALR7983-0923-accelerated-protection-pds-8-sep-2023.pdf"
    
    if os.path.exists(source_file):
        result = process_document_with_chunking(
            source_file, 
            chunk_size=512,  # Can be adjusted as needed
            chunk_overlap=50
        )
        
        # Analyze chunk quality
        analyze_chunks_quality(result['chunk_info'])
        
    else:
        print(f"File {source_file} does not exist")
        print("Please place the PDF file in the current directory, or modify the source_file variable") 