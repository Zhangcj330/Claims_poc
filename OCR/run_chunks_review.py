#!/usr/bin/env python3
"""
OCR Chunks Review - Main Script

This script demonstrates how to use ChunksReviewer to review and improve existing OCR chunks.

Usage:
    python run_chunks_review.py

Configuration:
    Please make sure GOOGLE_API_KEY is set in .env file
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunks_reviewer import ChunksReviewer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Review and improve OCR chunks using Gemini 2.5')
    parser.add_argument('--jsonl', '-j', 
                       default='Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl',
                       help='Path to input JSONL chunks file')
    parser.add_argument('--pdf', '-p',
                       default='TAL_AcceleratedProtection_2022-08-05.pdf', 
                       help='Path to original PDF file')
    parser.add_argument('--output', '-o',
                       help='Output directory (default: same as input)')
    parser.add_argument('--threshold', '-t', type=float, default=75.0,
                       help='Confidence threshold (default: 75.0)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show file information without processing')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: Please set GOOGLE_API_KEY in .env file")
        print("💡 Tip: Create .env file in project root with the following content:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        return 1
    
    # Check input files
    jsonl_path = args.jsonl
    pdf_path = args.pdf
    
    print("🚀 OCR Chunks Review System")
    print("=" * 50)
    print(f"📄 JSONL file: {jsonl_path}")
    print(f"📄 PDF file: {pdf_path}")
    print(f"🎯 Confidence threshold: {args.threshold}")
    
    # Check if files exist
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: JSONL file not found: {jsonl_path}")
        return 1
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return 1
    
    print("✅ Input file validation passed")
    
    # If dry run, only show file information
    if args.dry_run:
        print("\n🔍 Dry Run - File Information:")
        
        # Initialize reviewer (without API calls)
        try:
            reviewer = ChunksReviewer()
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return 1
        
        # Load chunks
        chunks = reviewer.load_chunks_from_jsonl(jsonl_path)
        page_groups = reviewer.group_chunks_by_page(chunks)
        
        print(f"📊 Total chunks: {len(chunks)}")
        print(f"📄 Total pages: {len(page_groups)}")
        print(f"📄 Page range: {min(page_groups.keys())} - {max(page_groups.keys())}")
        
        # Show chunks distribution per page
        print("\n📄 Chunks distribution by page:")
        for page_no in sorted(page_groups.keys())[:10]:  # Show only first 10 pages
            chunk_count = len(page_groups[page_no])
            print(f"   Page {page_no:2d}: {chunk_count} chunks")
        
        if len(page_groups) > 10:
            print(f"   ... and {len(page_groups) - 10} more pages")
        
        print("\n💡 To start actual processing, run without --dry-run parameter")
        return 0
    
    # Actual processing
    try:
        print("\n🚀 Starting processing...")
        
        # Initialize reviewer
        reviewer = ChunksReviewer()
        print("✅ ChunksReviewer initialized successfully")
        
        # Run review and improvement
        results = reviewer.review_and_improve_chunks(
            jsonl_path=jsonl_path,
            pdf_path=pdf_path,
            output_dir=args.output,
            confidence_threshold=args.threshold
        )
        
        # Show results
        print("\n🎉 Processing completed!")
        summary = results['summary']
        
        print("\n📊 Results Summary:")
        print("=" * 30)
        print(f"📄 Total pages: {summary['total_pages']}")
        print(f"🔧 Pages improved: {summary['pages_improved']}")
        print(f"📈 Improvement rate: {summary['improvement_rate']*100:.1f}%")
        print(f"⭐ Average confidence: {summary['average_confidence']:.1f}")
        
        print("\n📁 Generated files:")
        files = results['files']
        print(f"📄 Improved chunks: {files['improved_chunks']}")
        print(f"📊 Detailed report: {files['review_report']}")
        
        print("\n✨ Recommendations:")
        if summary['improvement_rate'] > 0.3:
            print("🔧 Many pages needed improvement, consider checking original OCR quality")
        elif summary['improvement_rate'] > 0.1:
            print("👍 Some pages improved, overall quality is good")
        else:
            print("🎯 Excellent quality, minimal improvements needed")
        
        if summary['average_confidence'] < 70:
            print("⚠️  Average confidence is low, consider checking document quality or adjusting parameters")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 