#!/usr/bin/env python3
"""
分析GGUF文件，列出所有张量名称
用于查找embedding等权重的正确名称
"""

import sys
from pathlib import Path

# 添加gguf-py到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'third_party' / 'llama.cpp' / 'gguf-py'))

from gguf.gguf_reader import GGUFReader

def analyze_gguf_tensors(gguf_file_path):
    """分析GGUF文件并列出所有张量名称"""
    reader = GGUFReader(gguf_file_path)
    
    print(f"=== GGUF文件分析: {gguf_file_path} ===\n")
    print(f"张量总数: {len(reader.tensors)}\n")
    
    # 查找embedding相关的张量
    print("=== Embedding相关张量 ===")
    embedding_tensors = []
    for tensor in reader.tensors:
        name_lower = tensor.name.lower()
        if any(keyword in name_lower for keyword in ['embed', 'tok', 'token']):
            embedding_tensors.append(tensor)
            print(f"  {tensor.name:50} | Shape: {tensor.shape} | Type: {tensor.tensor_type.name}")
    
    if not embedding_tensors:
        print("  未找到embedding相关张量")
    
    # 查找output/lm_head相关张量
    print("\n=== Output/LM Head相关张量 ===")
    output_tensors = []
    for tensor in reader.tensors:
        name_lower = tensor.name.lower()
        if any(keyword in name_lower for keyword in ['output', 'lm_head', 'head']):
            output_tensors.append(tensor)
            print(f"  {tensor.name:50} | Shape: {tensor.shape} | Type: {tensor.tensor_type.name}")
    
    if not output_tensors:
        print("  未找到output相关张量")
    
    # 查找norm相关张量
    print("\n=== Norm相关张量 ===")
    norm_tensors = []
    for tensor in reader.tensors:
        name_lower = tensor.name.lower()
        if 'norm' in name_lower:
            norm_tensors.append(tensor)
            print(f"  {tensor.name:50} | Shape: {tensor.shape} | Type: {tensor.tensor_type.name}")
    
    # 列出前几个层的张量（用于确认命名格式）
    print("\n=== 前5个层的张量（用于确认命名格式） ===")
    layer_tensors = {}
    for tensor in reader.tensors:
        # 查找 blk.0, blk.1 等格式
        if tensor.name.startswith('blk.'):
            parts = tensor.name.split('.')
            if len(parts) >= 2:
                try:
                    layer_num = int(parts[1])
                    if layer_num < 5:  # 只显示前5层
                        if layer_num not in layer_tensors:
                            layer_tensors[layer_num] = []
                        layer_tensors[layer_num].append(tensor.name)
                except ValueError:
                    pass
    
    for layer_num in sorted(layer_tensors.keys()):
        print(f"\n层 {layer_num}:")
        for tensor_name in sorted(layer_tensors[layer_num]):
            tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
            if tensor:
                print(f"  {tensor_name:50} | Shape: {tensor.shape} | Type: {tensor.tensor_type.name}")
    
    # 列出所有张量名称（前50个）
    print("\n=== 所有张量名称（前50个） ===")
    for i, tensor in enumerate(reader.tensors[:50]):
        print(f"  [{i:3}] {tensor.name:50} | Shape: {tensor.shape} | Type: {tensor.tensor_type.name}")
    
    if len(reader.tensors) > 50:
        print(f"\n  ... 还有 {len(reader.tensors) - 50} 个张量未显示")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: analyze_gguf_tensors.py <path_to_gguf_file>")
        sys.exit(1)
    
    gguf_file_path = sys.argv[1]
    analyze_gguf_tensors(gguf_file_path)
