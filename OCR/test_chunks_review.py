#!/usr/bin/env python3
"""
OCR Chunks Review - 快速测试脚本

这个脚本用于快速测试 chunks reviewer 系统是否正常工作，
无需消耗大量 API 调用。
"""

import os
import sys
from dotenv import load_dotenv

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_environment():
    """测试环境配置"""
    print("🧪 测试环境配置...")
    
    # 检查 .env 文件
    load_dotenv()
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ 缺少 GOOGLE_API_KEY 环境变量")
        print("💡 请在项目根目录创建 .env 文件并设置 API key")
        return False
    else:
        api_key = os.getenv('GOOGLE_API_KEY')
        print(f"✅ API Key 已设置 ({api_key[:10]}...)")
    
    return True

def test_dependencies():
    """测试依赖包"""
    print("\n🧪 测试依赖包...")
    
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
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("💡 请运行: pip install python-dotenv langchain-google-genai PyMuPDF Pillow")
        return False
    
    return True

def test_files():
    """测试输入文件"""
    print("\n🧪 测试输入文件...")
    
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
            print(f"❌ {file_path} (不存在)")
            all_exist = False
    
    if not all_exist:
        print("\n💡 测试文件不存在，但系统仍可使用其他文件")
    
    return True

def test_system_initialization():
    """测试系统初始化"""
    print("\n🧪 测试系统初始化...")
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        print("✅ ChunksReviewer 初始化成功")
        return True
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return False

def test_file_loading():
    """测试文件加载功能"""
    print("\n🧪 测试文件加载功能...")
    
    jsonl_file = "Data_Json/TAL_AcceleratedProtection_2022-08-05_chunks.jsonl"
    
    if not os.path.exists(jsonl_file):
        print(f"⚠️  测试文件不存在: {jsonl_file}")
        return True
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        
        # 测试加载 chunks
        chunks = reviewer.load_chunks_from_jsonl(jsonl_file)
        print(f"✅ 成功加载 {len(chunks)} 个 chunks")
        
        # 测试分组
        page_groups = reviewer.group_chunks_by_page(chunks)
        print(f"✅ 成功分组到 {len(page_groups)} 个页面")
        
        # 测试元数据提取
        metadata = reviewer.extract_global_metadata(chunks)
        print(f"✅ 成功提取元数据: {metadata.get('Insurer', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        return False

def test_pdf_processing():
    """测试 PDF 处理功能"""
    print("\n🧪 测试 PDF 处理功能...")
    
    pdf_file = "TAL_AcceleratedProtection_2022-08-05.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"⚠️  测试文件不存在: {pdf_file}")
        return True
    
    try:
        from chunks_reviewer import ChunksReviewer
        reviewer = ChunksReviewer()
        
        # 测试页面转图像
        page_image = reviewer.pdf_page_to_image(pdf_file, page_no=1)
        if page_image:
            print(f"✅ PDF 页面转图像成功: {page_image.size}")
            
            # 测试图像编码
            encoded = reviewer.encode_image(page_image)
            print(f"✅ 图像编码成功: {len(encoded)} characters")
        else:
            print("❌ PDF 页面转图像失败")
            return False
        
        return True
    except Exception as e:
        print(f"❌ PDF 处理失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🚀 OCR Chunks Review System - 快速测试")
    print("=" * 60)
    
    tests = [
        ("环境配置", test_environment),
        ("依赖包", test_dependencies), 
        ("输入文件", test_files),
        ("系统初始化", test_system_initialization),
        ("文件加载", test_file_loading),
        ("PDF处理", test_pdf_processing)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用")
        print("\n💡 下一步:")
        print("   1. 运行: python run_chunks_review.py --dry-run")
        print("   2. 或使用: python run_chunks_review.py")
        return 0
    else:
        print("⚠️  部分测试未通过，请根据上述提示解决问题")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 