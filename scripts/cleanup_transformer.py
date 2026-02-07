#!/usr/bin/env python3
"""
清理 transformer.cpp 中的调试代码
保留核心功能，删除调试日志
"""

import re
import sys

def remove_debug_blocks(content):
    """删除调试代码块"""
    
    # 删除单行 DEBUG 日志
    content = re.sub(r'\s*// DEBUG:.*\n', '\n', content)
    content = re.sub(r'\s*CLLM_INFO\("\[CPU DEBUG\].*\);\s*\n', '\n', content)
    content = re.sub(r'\s*CLLM_INFO\("\[LAYER_DEBUG\].*\);\s*\n', '\n', content)
    content = re.sub(r'\s*CLLM_INFO\("\[GPU DEBUG\].*\);\s*\n', '\n', content)
    content = re.sub(r'\s*CLLM_INFO\("\[DEBUG\].*\);\s*\n', '\n', content)
    content = re.sub(r'\s*CLLM_DEBUG\(.*\);\s*\n', '\n', content)
    
    # 删除多行调试代码块（统计信息打印）
    patterns = [
        # 删除统计信息打印块
        r'\s*// Debug: 检查 embedding 统计\s*\{[^}]+\}',
        r'\s*// DEBUG: 保存所有层的[^}]+\{[^}]+\}',
        r'\s*// DEBUG: 第一次残差连接后统计[^}]+\{[^}]+\}',
        r'\s*// DEBUG: Attention 输出统计[^}]+\{[^}]+\}',
        r'\s*// DEBUG: 打印 Layer[^}]+\{[^}]+\}',
        r'\s*// DEBUG: 计算并打印[^}]+\{[^}]+\}',
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '\n', content, flags=re.DOTALL)
    
    # 删除 if (startPos == 0) 的调试块
    content = re.sub(
        r'\s*if \(startPos == 0\) \{\s*'
        r'float minVal = [^;]+;\s*'
        r'float maxVal = [^;]+;\s*'
        r'double sum = 0;\s*'
        r'for \(int j = 0; j < [^;]+; \+\+j\) \{[^}]+\}\s*'
        r'float mean = sum / [^;]+;\s*'
        r'CLLM_INFO\([^;]+\);\s*'
        r'\}',
        '\n',
        content
    )
    
    return content

def main():
    input_file = "/Users/dannypan/PycharmProjects/cLLM/src/kylin/hf/transformer.cpp"
    output_file = "/Users/dannypan/PycharmProjects/cLLM/src/kylin/hf/transformer_clean.cpp"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    
    # 清理调试代码
    cleaned = remove_debug_blocks(content)
    
    # 删除多余的空行
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    cleaned_lines = len(cleaned.split('\n'))
    
    with open(output_file, 'w') as f:
        f.write(cleaned)
    
    print(f"Original: {original_lines} lines")
    print(f"Cleaned: {cleaned_lines} lines")
    print(f"Removed: {original_lines - cleaned_lines} lines")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
