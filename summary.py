#!/usr/bin/env python3
"""
Chunk Summarization Script

This script processes JSONL files containing document chunks and adds AI-generated
summaries as metadata to each chunk using OpenAI's GPT API.

Usage:
    python summary.py --input_file path/to/input.jsonl --output_file path/to/output.jsonl

Features:
- Processes chunks from JSONL files
- Generates concise summaries using OpenAI GPT
- Adds summary as new metadata field
- Handles rate limiting and errors gracefully
- Shows progress with tqdm
"""

import json
import argparse
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ChunkSummarizer:
    """Handles the summarization of document chunks using OpenAI GPT."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the summarizer.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
            model: OpenAI model to use for summarization.
        """
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.rate_limit_delay = 1  # seconds between requests to avoid rate limiting
    
    def generate_summary(self, chunk: Dict[str, Any]) -> str:
        """
        Generate a summary for a single chunk.
        
        Args:
            chunk: Dictionary containing chunk data
            
        Returns:
            Generated summary as string
        """
        content = chunk.get('content', '')
        content_label = chunk.get('content_label', 'text')
        section_title = chunk.get('Section_Title', 'N/A')
        subheading = chunk.get('Subheading', 'N/A')
        
        # Create context-aware prompt
        prompt = self._create_summary_prompt(content, content_label, section_title, subheading)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, informative summaries of insurance document content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3,
                timeout=30
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Add rate limiting delay
            time.sleep(self.rate_limit_delay)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    def _create_summary_prompt(self, content: str, content_label: str, section_title: str, subheading: str) -> str:
        """
        Create a context-aware prompt for summarization.
        
        Args:
            content: The main content to summarize
            content_label: Type of content (text, image, table, list, etc.)
            section_title: Section title for context
            subheading: Subheading for context
            
        Returns:
            Formatted prompt string
        """
        context_info = []
        if section_title != 'N/A':
            context_info.append(f"Section: {section_title}")
        if subheading != 'N/A':
            context_info.append(f"Subheading: {subheading}")
        if content_label:
            context_info.append(f"Content type: {content_label}")
        
        context = " | ".join(context_info) if context_info else "General content"
        
        prompt = f"""Please create a concise summary (1-2 sentences, max 150 words) of the following insurance document content.

Context: {context}

Content to summarize:
{content}

Focus on the key information, benefits, conditions, or requirements mentioned. Make the summary informative and specific to insurance terms and concepts."""

        return prompt

def load_chunks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return chunks

def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save chunks to a JSONL file.
    
    Args:
        chunks: List of chunk dictionaries
        file_path: Path to save the JSONL file
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                json.dump(chunk, f, ensure_ascii=False)
                f.write('\n')
        print(f"Successfully saved {len(chunks)} chunks to {file_path}")
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")

def process_chunks_with_summaries(input_file: str, output_file: str, model: str = "gpt-4o-mini") -> None:
    """
    Process chunks from input file and add summaries, save to output file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        model: OpenAI model to use
    """
    print(f"Loading chunks from {input_file}...")
    chunks = load_chunks_from_jsonl(input_file)
    
    if not chunks:
        print("No chunks loaded. Exiting.")
        return
    
    print(f"Loaded {len(chunks)} chunks.")
    
    # Initialize summarizer
    print(f"Initializing summarizer with model: {model}")
    summarizer = ChunkSummarizer(model=model)
    
    # Process chunks with progress bar
    print("Generating summaries...")
    processed_chunks = []
    
    for chunk in tqdm(chunks, desc="Processing chunks"):
        try:
            # Generate summary
            summary = summarizer.generate_summary(chunk)
            
            # Add summary to chunk metadata
            chunk_with_summary = chunk.copy()
            chunk_with_summary['ai_summary'] = summary
            
            processed_chunks.append(chunk_with_summary)
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            if processed_chunks:
                print(f"Saving {len(processed_chunks)} processed chunks...")
                save_chunks_to_jsonl(processed_chunks, output_file)
            return
        except Exception as e:
            print(f"Error processing chunk: {e}")
            # Still add the chunk but mark summary as failed
            chunk_with_summary = chunk.copy()
            chunk_with_summary['ai_summary'] = f"Failed to generate summary: {str(e)}"
            processed_chunks.append(chunk_with_summary)
    
    # Save processed chunks
    print(f"Saving processed chunks to {output_file}...")
    save_chunks_to_jsonl(processed_chunks, output_file)

def main():
    """Main function to handle command line arguments and orchestrate the summarization process."""
    parser = argparse.ArgumentParser(
        description="Add AI-generated summaries to document chunks in JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process single file
    python summary.py --input_file ocr/Data_Json/TAL_chunks.jsonl --output_file ocr/Data_Json/TAL_chunks_with_summaries.jsonl
    
    # Use specific model
    python summary.py --input_file input.jsonl --output_file output.jsonl --model gpt-4o-mini
    
    # Process all JSONL files in directory
    python summary.py --process_all --input_dir ocr/Data_Json --output_dir ocr/Data_Json/summarized
        """
    )
    
    parser.add_argument(
        '--input_file', 
        type=str, 
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='gpt-4o-mini',
        help='OpenAI model to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--process_all', 
        action='store_true',
        help='Process all JSONL files in input directory'
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='ocr/Data_Json',
        help='Input directory for batch processing (default: ocr/Data_Json)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Output directory for batch processing'
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key in your environment or .env file.")
        return
    
    if args.process_all:
        # Batch processing mode
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / "summarized"
        
        if not input_dir.exists():
            print(f"Error: Input directory {input_dir} does not exist.")
            return
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSONL files
        jsonl_files = list(input_dir.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No JSONL files found in {input_dir}")
            return
        
        print(f"Found {len(jsonl_files)} JSONL files to process:")
        for file in jsonl_files:
            print(f"  - {file.name}")
        
        # Process each file
        for input_file in jsonl_files:
            output_file = output_dir / f"{input_file.stem}_with_summaries.jsonl"
            print(f"\n{'='*60}")
            print(f"Processing: {input_file.name}")
            print(f"Output: {output_file}")
            process_chunks_with_summaries(str(input_file), str(output_file), args.model)
    
    else:
        # Single file processing mode
        if not args.input_file or not args.output_file:
            print("Error: --input_file and --output_file are required for single file processing.")
            print("Use --process_all for batch processing.")
            parser.print_help()
            return
        
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} does not exist.")
            return
        
        process_chunks_with_summaries(args.input_file, args.output_file, args.model)

if __name__ == "__main__":
    main()
