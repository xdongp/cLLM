#!/usr/bin/env python3
"""
自研HTTP服务器全面测试脚本
使用Python requests库进行测试
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import sys

class TestStats:
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.lock = threading.Lock()
    
    def add_result(self, passed: bool):
        with self.lock:
            self.total += 1
            if passed:
                self.passed += 1
            else:
                self.failed += 1
    
    def print(self):
        print("\n" + "=" * 50)
        print("测试统计:")
        print(f"  总计: {self.total}")
        print(f"  通过: {self.passed}")
        print(f"  失败: {self.failed}")
        if self.total > 0:
            print(f"  成功率: {100.0 * self.passed / self.total:.2f}%")
        print("=" * 50 + "\n")

class HttpClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def get(self, path: str) -> Tuple[int, str]:
        """GET请求"""
        try:
            url = self.base_url + path
            response = self.session.get(url, timeout=self.timeout)
            return response.status_code, response.text
        except Exception as e:
            return 0, str(e)
    
    def post(self, path: str, data: dict) -> Tuple[int, str]:
        """POST请求"""
        try:
            url = self.base_url + path
            response = self.session.post(url, json=data, timeout=self.timeout)
            return response.status_code, response.text
        except Exception as e:
            return 0, str(e)

# 测试统计
stats = TestStats()

def test(name: str, func):
    """测试辅助函数"""
    print(f"[TEST] {name}...", end='', flush=True)
    try:
        result = func()
        if result:
            print(" ✓ PASSED")
            stats.add_result(True)
        else:
            print(" ✗ FAILED")
            stats.add_result(False)
    except Exception as e:
        print(f" ✗ FAILED (exception: {e})")
        stats.add_result(False)

# ============================================================================
# 测试用例
# ============================================================================

def test_health_endpoint(client: HttpClient) -> bool:
    """测试1: 健康检查端点"""
    status_code, response = client.get("/health")
    if status_code != 200:
        return False
    try:
        data = json.loads(response)
        return data.get("success") == True
    except:
        return False

def test_generate_basic(client: HttpClient) -> bool:
    """测试2: 生成端点（基本）"""
    status_code, response = client.post("/generate", {
        "prompt": "Hello",
        "max_tokens": 5
    })
    if status_code != 200:
        return False
    try:
        data = json.loads(response)
        return data.get("success") == True and "data" in data and "text" in data["data"]
    except:
        return False

def test_encode_endpoint(client: HttpClient) -> bool:
    """测试3: 编码端点"""
    status_code, response = client.post("/encode", {
        "text": "Hello world"
    })
    if status_code != 200:
        return False
    try:
        data = json.loads(response)
        return data.get("success") == True and "data" in data and "tokens" in data["data"]
    except:
        return False

def test_404_not_found(client: HttpClient) -> bool:
    """测试4: 404 Not Found"""
    status_code, _ = client.get("/nonexistent")
    return status_code == 404

def test_invalid_json(client: HttpClient) -> bool:
    """测试5: 无效JSON"""
    try:
        url = client.base_url + "/generate"
        response = client.session.post(url, data="{invalid json}", timeout=client.timeout)
        return response.status_code == 400
    except:
        return False

def test_empty_request_body(client: HttpClient) -> bool:
    """测试6: 空请求体"""
    try:
        url = client.base_url + "/generate"
        response = client.session.post(url, data="", timeout=client.timeout)
        return response.status_code == 400
    except:
        return False

def test_large_request_body(client: HttpClient) -> bool:
    """测试7: 大请求体"""
    large_prompt = "A" * 10000  # 10KB
    status_code, _ = client.post("/generate", {
        "prompt": large_prompt,
        "max_tokens": 5
    })
    # 可能被拒绝，但不应崩溃
    return status_code in [200, 400]

def test_concurrent_requests(client: HttpClient, num_threads: int = 10, requests_per_thread: int = 10) -> bool:
    """测试8: 并发请求"""
    success_count = 0
    fail_count = 0
    
    def worker(thread_id: int):
        nonlocal success_count, fail_count
        for i in range(requests_per_thread):
            status_code, response = client.post("/generate", {
                "prompt": f"Test {thread_id}-{i}",
                "max_tokens": 3
            })
            if status_code == 200:
                try:
                    data = json.loads(response)
                    if data.get("success") == True:
                        with stats.lock:
                            success_count += 1
                        return
                except:
                    pass
            with stats.lock:
                fail_count += 1
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    total = num_threads * requests_per_thread
    success_rate = success_count / total if total > 0 else 0
    
    print(f"  (并发: {num_threads}线程 × {requests_per_thread}请求 = {total}请求, "
          f"成功: {success_count}, 成功率: {success_rate*100:.1f}%)", end='', flush=True)
    
    return success_rate >= 0.95  # 95%成功率

def test_keep_alive(client: HttpClient) -> bool:
    """测试9: Keep-Alive连接"""
    for i in range(5):
        status_code, _ = client.get("/health")
        if status_code != 200:
            return False
        time.sleep(0.01)
    return True

def test_stress_test(client: HttpClient, num_requests: int = 100) -> bool:
    """测试10: 压力测试"""
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    def worker(request_id: int):
        nonlocal success_count, fail_count
        status_code, response = client.post("/generate", {
            "prompt": f"Stress test {request_id}",
            "max_tokens": 3
        })
        if status_code == 200:
            try:
                data = json.loads(response)
                if data.get("success") == True:
                    with stats.lock:
                        success_count += 1
                    return
            except:
                pass
        with stats.lock:
            fail_count += 1
    
    threads = []
    for i in range(num_requests):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    duration = end_time - start_time
    total = num_requests
    success_rate = success_count / total if total > 0 else 0
    qps = total / duration if duration > 0 else 0
    
    print(f"  (请求数: {num_requests}, 成功: {success_count}, "
          f"耗时: {duration:.2f}s, QPS: {qps:.1f}, "
          f"成功率: {success_rate*100:.1f}%)", end='', flush=True)
    
    return success_rate >= 0.90  # 90%成功率

def test_benchmark_endpoint(client: HttpClient) -> bool:
    """测试11: Benchmark端点"""
    status_code, response = client.post("/benchmark", {
        "requests": 5,
        "concurrency": 2,
        "max_tokens": 10,
        "prompt": "Test"
    })
    if status_code != 200:
        return False
    try:
        data = json.loads(response)
        return data.get("success") == True
    except:
        return False

def test_stream_endpoint(client: HttpClient) -> bool:
    """测试12: 流式端点（如果支持）"""
    try:
        url = client.base_url + "/generate_stream"
        response = client.session.post(url, json={
            "prompt": "Hello",
            "max_tokens": 5,
            "stream": True
        }, stream=True, timeout=client.timeout)
        return response.status_code == 200
    except:
        return False

# ============================================================================
# 主函数
# ============================================================================

def main():
    server_url = "http://localhost:8080"
    
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    
    print("=" * 50)
    print("自研HTTP服务器全面测试")
    print("=" * 50)
    print(f"服务器地址: {server_url}")
    print("开始测试...")
    print()
    
    client = HttpClient(server_url)
    
    # 等待服务器启动
    print("等待服务器就绪...", end='', flush=True)
    for i in range(10):
        status_code, _ = client.get("/health")
        if status_code == 200:
            print(" ✓")
            break
        time.sleep(1)
        print(".", end='', flush=True)
    else:
        print(" ✗ 服务器未就绪")
        return 1
    print()
    
    # 基本功能测试
    print("\n[基本功能测试]")
    test("健康检查端点", lambda: test_health_endpoint(client))
    test("生成端点（基本）", lambda: test_generate_basic(client))
    test("编码端点", lambda: test_encode_endpoint(client))
    test("Benchmark端点", lambda: test_benchmark_endpoint(client))
    test("流式端点", lambda: test_stream_endpoint(client))
    
    # 错误处理测试
    print("\n[错误处理测试]")
    test("404 Not Found", lambda: test_404_not_found(client))
    test("无效JSON", lambda: test_invalid_json(client))
    test("空请求体", lambda: test_empty_request_body(client))
    
    # 边界条件测试
    print("\n[边界条件测试]")
    test("大请求体", lambda: test_large_request_body(client))
    
    # 并发测试
    print("\n[并发测试]")
    test("并发请求（10线程×10请求）", lambda: test_concurrent_requests(client, 10, 10))
    test("并发请求（20线程×5请求）", lambda: test_concurrent_requests(client, 20, 5))
    test("并发请求（50线程×2请求）", lambda: test_concurrent_requests(client, 50, 2))
    
    # 连接测试
    print("\n[连接测试]")
    test("Keep-Alive连接", lambda: test_keep_alive(client))
    
    # 压力测试
    print("\n[压力测试]")
    test("压力测试（100请求）", lambda: test_stress_test(client, 100))
    test("压力测试（200请求）", lambda: test_stress_test(client, 200))
    test("压力测试（500请求）", lambda: test_stress_test(client, 500))
    
    # 打印统计
    stats.print()
    
    return 1 if stats.failed > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
