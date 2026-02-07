#!/usr/bin/env python3
"""
清理 transformer.cpp 和 ggml_backend.cpp 中的调试代码
"""

import re

def clean_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_lines = len(content.split('\n'))
    
    # 1. 删除单行 DEBUG 日志 (CLLM_INFO/CLLM_DEBUG 包含 DEBUG 关键字)
    patterns = [
        # 删除 CLLM_INFO 包含 DEBUG 的行
        r'\s*CLLM_INFO\("[^"]*DEBUG[^"]*"[^;]*\);\s*\n',
        # 删除 CLLM_DEBUG 行
        r'\s*CLLM_DEBUG\([^;]*\);\s*\n',
        # 删除 // DEBUG: 注释行
        r'\s*//\s*DEBUG:[^\n]*\n',
        # 删除 // Debug: 注释行
        r'\s*//\s*Debug:[^\n]*\n',
    ]
    
    for pattern in patterns:
        content = re.sub(pattern, '\n', content, flags=re.IGNORECASE)
    
    # 2. 删除统计信息打印块 (if (startPos == 0) { ... CLLM_INFO ... })
    # 匹配统计信息块
    stat_pattern = r'\s*if \(startPos == 0\) \{\s*float minVal[^;]+;\s*float maxVal[^;]+;\s*double sum = 0;\s*for \(int j = 0; j < [^;]+; \+\+j\) \{[^}]+\}\s*float mean = sum / [^;]+;\s*CLLM_INFO\([^;]+\);\s*\}'
    content = re.sub(stat_pattern, '\n', content)
    
    # 3. 删除其他变体的统计块
    stat_pattern2 = r'\s*if \(startPos == 0 \|\| startPos == 1\) \{\s*float minVal[^;]+;\s*float maxVal[^;]+;\s*double sum = 0;\s*for \(int j = 0; j < [^;]+; \+\+j\) \{[^}]+\}\s*float mean = sum / [^;]+;\s*CLLM_INFO\([^;]+\);\s*\}'
    content = re.sub(stat_pattern2, '\n', content)
    
    # 4. 删除多余的空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    cleaned_lines = len(content.split('\n'))
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"{filepath}:")
    print(f"  Original: {original_lines} lines")
    print(f"  Cleaned: {cleaned_lines} lines")
    print(f"  Removed: {original_lines - cleaned_lines} lines")
    return original_lines - cleaned_lines

if __name__ == "__main__":
    files = [
        "/Users/dannypan/PycharmProjects/cLLM/src/kylin/hf/transformer.cpp",
        "/Users/dannypan/PycharmProjects/cLLM/src/kylin/hf/ggml_backend.cpp"
    ]
    
    total_removed = 0
    for filepath in files:
        try:
            removed = clean_file(filepath)
            total_removed += removed
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\nTotal lines removed: {total_removed}")
