#!/usr/bin/env python3
"""
为JSONL文件中的每个chunk添加chunk_id，从0开始递增
"""

import json
import sys
from pathlib import Path

def add_chunk_ids(input_file, output_file=None):
    """
    为JSONL文件中的每个JSON对象添加chunk_id
    
    Args:
        input_file (str): 输入的JSONL文件路径
        output_file (str, optional): 输出文件路径，如果为None则在原文件名后加_with_ids
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"错误: 文件 {input_file} 不存在")
        return False
    
    # 如果没有指定输出文件，则在原文件名后加_with_ids
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_with_ids{input_path.suffix}"
    else:
        output_path = Path(output_file)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            chunk_id = 0
            processed_count = 0
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 解析JSON
                    chunk_data = json.loads(line)
                    
                    # 添加chunk_id
                    chunk_data['chunk_id'] = chunk_id
                    
                    # 写入输出文件
                    json.dump(chunk_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    chunk_id += 1
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON格式错误: {e}")
                    continue
            
            print(f"处理完成!")
            print(f"输入文件: {input_path}")
            print(f"输出文件: {output_path}")
            print(f"处理了 {processed_count} 个chunks")
            print(f"添加的chunk_id范围: 0 到 {chunk_id - 1}")
            
            return True
            
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("使用方法: python add_chunk_ids.py <input_file> [output_file]")
        print("例子: python add_chunk_ids.py data.jsonl")
        print("例子: python add_chunk_ids.py data.jsonl data_with_ids.jsonl")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = add_chunk_ids(input_file, output_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 