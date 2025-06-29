#!/usr/bin/env python3
"""
Standalone validation tool for JSONL chunk files
Usage: python validate_chunks.py <jsonl_file_path>
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import gemini_ocr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_ocr import GeminiPDSProcessor, validate_file_standalone

def validate_single_file(file_path: str):
    """Validate a single JSONL file"""
    print(f"🔍 Validating file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        result = validate_file_standalone(file_path)
        return result['is_valid']
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def validate_directory(directory: str):
    """Validate all JSONL files in a directory"""
    print(f"🔍 Validating all JSONL files in: {directory}")
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return
    
    jsonl_files = list(Path(directory).glob("**/*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {directory}")
        return
    
    valid_count = 0
    total_count = len(jsonl_files)
    
    print(f"\nFound {total_count} JSONL files:")
    print("=" * 60)
    
    for file_path in jsonl_files:
        print(f"\n📄 {file_path.name}")
        if validate_single_file(str(file_path)):
            valid_count += 1
            print("✅ VALID")
        else:
            print("❌ INVALID")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {valid_count}/{total_count} files are valid")
    
    if valid_count == total_count:
        print("🎉 All files passed validation!")
    else:
        print(f"⚠️  {total_count - valid_count} files have issues that need attention")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python validate_chunks.py <file_path>           # Validate single file")
        print("  python validate_chunks.py <directory_path>     # Validate all JSONL files in directory")
        print("\nExamples:")
        print("  python validate_chunks.py Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl")
        print("  python validate_chunks.py Data_Json/")
        return 1
    
    target_path = sys.argv[1]
    
    if os.path.isfile(target_path):
        # Validate single file
        success = validate_single_file(target_path)
        return 0 if success else 1
    elif os.path.isdir(target_path):
        # Validate directory
        validate_directory(target_path)
        return 0
    else:
        print(f"❌ Path not found: {target_path}")
        return 1

if __name__ == "__main__":
    exit(main()) 