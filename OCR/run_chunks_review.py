#!/usr/bin/env python3
"""
OCR Chunks Review - 运行脚本

这个脚本演示如何使用 ChunksReviewer 来检查和改进现有的 OCR chunks。

使用方法:
    python run_chunks_review.py

配置文件:
    请确保在 .env 文件中设置了 GOOGLE_API_KEY
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunks_reviewer import ChunksReviewer

def main():
    # 解析命令行参数
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
    
    # 加载环境变量
    load_dotenv()
    
    # 检查 API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ 错误: 请在 .env 文件中设置 GOOGLE_API_KEY")
        print("💡 提示: 在项目根目录创建 .env 文件，内容如下:")
        print("   GOOGLE_API_KEY=your_api_key_here")
        return 1
    
    # 检查输入文件
    jsonl_path = args.jsonl
    pdf_path = args.pdf
    
    print("🚀 OCR Chunks Review System")
    print("=" * 50)
    print(f"📄 JSONL 文件: {jsonl_path}")
    print(f"📄 PDF 文件: {pdf_path}")
    print(f"🎯 置信度阈值: {args.threshold}")
    
    # 检查文件是否存在
    if not os.path.exists(jsonl_path):
        print(f"❌ 错误: JSONL 文件不存在: {jsonl_path}")
        return 1
    
    if not os.path.exists(pdf_path):
        print(f"❌ 错误: PDF 文件不存在: {pdf_path}")
        return 1
    
    print("✅ 输入文件检查通过")
    
    # 如果是 dry run，只显示文件信息
    if args.dry_run:
        print("\n🔍 Dry Run - 文件信息:")
        
        # 初始化 reviewer（不调用 API）
        try:
            reviewer = ChunksReviewer()
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return 1
        
        # 加载 chunks
        chunks = reviewer.load_chunks_from_jsonl(jsonl_path)
        page_groups = reviewer.group_chunks_by_page(chunks)
        
        print(f"📊 总 chunks: {len(chunks)}")
        print(f"📄 总页面数: {len(page_groups)}")
        print(f"📄 页面范围: {min(page_groups.keys())} - {max(page_groups.keys())}")
        
        # 显示每页的 chunks 数量
        print("\n📄 各页面 chunks 分布:")
        for page_no in sorted(page_groups.keys())[:10]:  # 只显示前10页
            chunk_count = len(page_groups[page_no])
            print(f"   第 {page_no:2d} 页: {chunk_count} chunks")
        
        if len(page_groups) > 10:
            print(f"   ... 还有 {len(page_groups) - 10} 页")
        
        print("\n💡 要开始实际处理，请运行不带 --dry-run 参数的命令")
        return 0
    
    # 实际处理
    try:
        print("\n🚀 开始处理...")
        
        # 初始化 reviewer
        reviewer = ChunksReviewer()
        print("✅ ChunksReviewer 初始化成功")
        
        # 运行审查和改进
        results = reviewer.review_and_improve_chunks(
            jsonl_path=jsonl_path,
            pdf_path=pdf_path,
            output_dir=args.output,
            confidence_threshold=args.threshold
        )
        
        # 显示结果
        print("\n🎉 处理完成!")
        summary = results['summary']
        
        print("\n📊 结果摘要:")
        print("=" * 30)
        print(f"📄 总页面数: {summary['total_pages']}")
        print(f"🔧 改进页面数: {summary['pages_improved']}")
        print(f"📈 改进率: {summary['improvement_rate']*100:.1f}%")
        print(f"⭐ 平均置信度: {summary['average_confidence']:.1f}")
        
        print("\n📁 生成的文件:")
        files = results['files']
        print(f"📄 改进后的 chunks: {files['improved_chunks']}")
        print(f"📊 详细报告: {files['review_report']}")
        
        print("\n✨ 建议:")
        if summary['improvement_rate'] > 0.3:
            print("🔧 有较多页面需要改进，建议检查原始 OCR 质量")
        elif summary['improvement_rate'] > 0.1:
            print("👍 部分页面有改进，整体质量不错")
        else:
            print("🎯 质量很好，几乎不需要改进")
        
        if summary['average_confidence'] < 70:
            print("⚠️  平均置信度较低，建议检查文档质量或调整参数")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        return 1
    except Exception as e:
        print(f"\n❌ 处理出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 