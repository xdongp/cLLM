#!/usr/bin/env python3
"""测试服务器流式输出是否是真流式"""

import requests
import json
import time

def test_streaming(max_tokens):
    url = "http://localhost:8085/generate_stream"
    data = {"prompt": "解释人工智能", "max_tokens": max_tokens, "stream": True}
    
    start = time.time()
    response = requests.post(url, json=data, stream=True, timeout=300)
    ttfb = time.time() - start  # Time To First Byte
    
    # 读取所有数据
    content = b""
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            content += chunk
    
    total_time = time.time() - start
    transfer_time = total_time - ttfb
    
    # 统计 token 数
    lines = content.decode("utf-8").strip().split("\n")
    token_count = sum(1 for l in lines if l.startswith("data:") and "token" in l)
    
    return {
        "max_tokens": max_tokens,
        "actual_tokens": token_count,
        "ttfb": ttfb,
        "transfer_time": transfer_time,
        "total_time": total_time,
        "bytes": len(content)
    }

if __name__ == "__main__":
    print("流式输出时间分析")
    print("=" * 70)
    header = f"{'max_tokens':>12} | {'实际tokens':>10} | {'TTFB(s)':>10} | {'传输(s)':>10} | {'总时间(s)':>10}"
    print(header)
    print("-" * 70)

    for tokens in [50, 100, 200, 300]:
        r = test_streaming(tokens)
        row = f"{r['max_tokens']:>12} | {r['actual_tokens']:>10} | {r['ttfb']:>10.3f} | {r['transfer_time']:>10.3f} | {r['total_time']:>10.3f}"
        print(row)

    print("-" * 70)
    print()
    print("分析:")
    print("- 如果是【真流式】: TTFB 应该很短(<0.5s)，传输时间 ≈ 生成时间")
    print("- 如果是【伪流式】: TTFB ≈ 生成时间，传输时间很短(<0.5s)")
