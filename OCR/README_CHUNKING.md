# Docling HybridChunker Usage Guide

## Overview

This project demonstrates how to use Docling's HybridChunker for intelligent document chunking, particularly suitable for Claims processing scenarios.

## File Description

### Core Files
- `docling_chunking.py` - Complete chunking processing script
- `chunking_demo.ipynb` - Interactive demonstration and analysis notebook
- `docling.py` - Basic document conversion script

### Output Files
- `*_chunks.json` - Structured chunking data
- `*_chunks.txt` - Human-readable chunking text
- `*_converted.md` - Original converted Markdown

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# Convert document
converter = DocumentConverter()
result = converter.convert("your_document.pdf")
doc = result.document

# Configure chunker
chunker = HybridChunker(
    chunk_size=512,                    # Target chunk size
    overlap_size=50,                   # Overlap size
    split_by_page=True,                # Split by page boundaries
    respect_section_boundaries=True    # Respect section boundaries
)

# Execute chunking
chunks = chunker.chunk(doc)
```

### 3. Run Complete Example

```bash
cd OCR
python docling_chunking.py
```

## HybridChunker Core Parameters

### Main Configuration
- **chunk_size**: Target chunk size (in tokens)
- **overlap_size**: Overlap size between chunks to maintain context continuity
- **split_by_page**: Whether to split by page boundaries
- **respect_section_boundaries**: Whether to respect document section structure

### Chunking Strategy Selection

| Strategy | chunk_size | overlap_size | Use Case |
|----------|------------|--------------|----------|
| Small chunks | 256 | 25 | Fine-grained retrieval, Q&A systems |
| Medium chunks | 512 | 50 | Balanced performance, general processing |
| Large chunks | 1024 | 100 | Context preservation, summarization |

## Special Features for Claims Processing

### Content Classification
HybridChunker can automatically identify and classify:
- Claims-related content (policy, coverage, deductible, etc.)
- Financial information (amounts, rates, percentages, etc.)
- Legal terms (terms, conditions, clauses, etc.)
- Tables and data structures

### Example Code
```python
# Claims-specific analysis
def analyze_claims_content(chunks):
    claims_keywords = ['claim', 'policy', 'premium', 'coverage']
    # ... analysis logic
```

## Best Practices

### 1. Choosing Appropriate Chunk Size
- **Small documents** (< 10 pages): chunk_size=256-512
- **Medium documents** (10-50 pages): chunk_size=512-1024  
- **Large documents** (> 50 pages): chunk_size=1024-2048

### 2. Overlap Strategy
- For scenarios requiring context preservation, use 10-20% overlap
- For independent processing scenarios, reduce overlap

### 3. Boundary Handling
- Enable `split_by_page` to avoid cross-page semantic fragmentation
- Enable `respect_section_boundaries` to maintain document structure integrity

## Output Formats

### JSON Format
```json
{
  "id": 0,
  "text": "Chunk content text",
  "length": 234,
  "word_count": 45,
  "has_tables": false,
  "has_lists": true,
  "metadata": {
    "chunk_type": "text"
  }
}
```

### Text Format
```
--- Chunk 1 [text] ---
Length: 234 characters, 45 words
✓ Contains lists
Content:
...actual chunk content...
```

## Advanced Features

### 1. Batch Processing
```python
import glob
pdf_files = glob.glob("*.pdf")
for pdf_file in pdf_files:
    process_document_with_chunking(pdf_file)
```

### 2. Custom Chunking Logic
```python
class CustomChunker(HybridChunker):
    def should_split_here(self, text, position):
        # Custom splitting logic
        return super().should_split_here(text, position)
```

### 3. Quality Assessment
Use built-in analysis functions to evaluate chunking quality:
- Length distribution statistics
- Content type identification
- Detection of oversized/undersized chunks

## Performance Optimization

### 1. Memory Management
For large documents, consider:
- Batch processing
- Timely release of unnecessary objects
- Using generator patterns

### 2. Processing Speed
- Smaller chunk_size produces more chunks and longer processing time
- Disabling unnecessary boundary checks can improve speed

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure `docling[chunking]` is installed
2. **Memory insufficient**: Reduce chunk_size or process in batches
3. **Poor chunking quality**: Adjust parameters or use custom logic

### Debugging Tips
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check chunking statistics
analyze_chunks_quality(chunks)
```

## Next Steps

1. Try different parameter combinations
2. Integrate into your Claims processing pipeline
3. Add custom content analysis logic
4. Consider integration with vector databases for semantic search 