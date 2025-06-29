#!/usr/bin/env python3
"""
OCR Chunks Review - Quick Test Script

This script is used to quickly test if the chunks reviewer system 
is working properly without consuming many API calls.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """Test environment configuration"""
    print("🧪 Testing environment configuration...")
    
    # Check .env file
    load_dotenv()
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Missing GOOGLE_API_KEY environment variable")
        print("💡 Please create .env file in project root and set API key")
        return False
    else:
        api_key = os.getenv('GOOGLE_API_KEY')
        print(f"✅ API Key is set ({api_key[:10]}...)")
    
    return True

def test_dependencies():
    """Test required packages"""
    print("\n🧪 Testing dependencies...")
    
    required_packages = [
        'fitz',  # PyMuPDF
        'PIL',   # Pillow
        'langchain_google_genai',
        'dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz
            elif package == 'PIL':
                from PIL import Image
            elif package == 'langchain_google_genai':
                from langchain_google_genai import ChatGoogleGenerativeAI
            elif package == 'dotenv':
                from dotenv import load_dotenv
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Please run: pip install python-dotenv langchain-google-genai PyMuPDF Pillow")
        return False
    
    return True

def test_files():
    """Test input files"""
    print("\n🧪 Testing input files...")
    
    test_files = [
        "Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl",
        "TAL_AcceleratedProtection_2022-08-05.pdf"
    ]
    
    all_exist = True
    for file_path in test_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (not found)")
            all_exist = False
    
    if not all_exist:
        print("\n💡 Test files not found, but system can still work with other files")
    
    return True

def test_system_initialization():
    """Test system initialization"""
    print("\n🧪 Testing system initialization...")
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        print("✅ ChunksReviewer initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_file_loading():
    """Test file loading functionality"""
    print("\n🧪 Testing file loading functionality...")
    
    jsonl_file = "Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"⚠️  Test file not found: {jsonl_file}")
        return True
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        
        # Test loading chunks
        chunks = reviewer.load_chunks_from_jsonl(jsonl_file)
        print(f"✅ Successfully loaded {len(chunks)} chunks")
        
        # Test grouping
        page_groups = reviewer.group_chunks_by_page(chunks)
        print(f"✅ Successfully grouped into {len(page_groups)} pages")
        
        # Test metadata extraction
        metadata = reviewer.extract_global_metadata(chunks)
        print(f"✅ Successfully extracted metadata: {metadata.get('Insurer', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ File loading failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing functionality"""
    print("\n🧪 Testing PDF processing functionality...")
    
    pdf_file = "TAL_AcceleratedProtection_2022-08-05.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"⚠️  Test file not found: {pdf_file}")
        return True
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        
        # Test page to image conversion
        page_image = reviewer.pdf_page_to_image(pdf_file, page_no=1)
        if page_image:
            print(f"✅ PDF page to image conversion successful: {page_image.size}")
            
            # Test image encoding
            encoded = reviewer.encode_image(page_image)
            print(f"✅ Image encoding successful: {len(encoded)} characters")
        else:
            print("❌ PDF page to image conversion failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 OCR Chunks Review System - Quick Test")
    print("=" * 60)
    
    tests = [
        ("Environment Configuration", test_environment),
        ("Dependencies", test_dependencies), 
        ("Input Files", test_files),
        ("System Initialization", test_system_initialization),
        ("File Loading", test_file_loading),
        ("PDF Processing", test_pdf_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use")
        print("\n💡 Next steps:")
        print("   1. Run: python run_chunks_review.py --dry-run")
        print("   2. Or use: python run_chunks_review.py")
        return 0
    else:
        print("⚠️  Some tests failed, please resolve issues based on the above messages")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 