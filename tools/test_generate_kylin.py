#!/usr/bin/env python3
"""
测试 cLLM /generate 接口 - 使用 Kylin backend 和 GGUF 格式
"""

import requests
import json
import time
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("generate-test")


class GenerateTester:
    """生成接口测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/generate"
        self.health_url = f"{self.base_url}/health"
    
    def check_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ 服务器健康检查通过")
                return True
            else:
                logger.error(f"❌ 服务器健康检查失败: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ 无法连接到服务器: {e}")
            return False
    
    def test_generate(self, prompt: str, max_tokens: int = 10, temperature: float = 0.0) -> dict:
        """测试生成接口"""
        logger.info(f"发送请求: prompt='{prompt}', max_tokens={max_tokens}, temperature={temperature}")
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                self.generate_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logger.info(f"响应时间: {response_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"响应内容: {json.dumps(result, indent=2, ensure_ascii=False)}")
                
                data = result.get("data", {})
                generated_text = data.get("text", "")
                tokens_per_second = data.get("tokens_per_second", 0)
                
                logger.info(f"✅ 生成成功")
                logger.info(f"生成文本: '{generated_text}'")
                logger.info(f"Tokens/秒: {tokens_per_second:.2f}")
                
                return {
                    "success": True,
                    "generated_text": generated_text,
                    "tokens_per_second": tokens_per_second,
                    "response_time": response_time
                }
            else:
                logger.error(f"❌ 请求失败: HTTP {response.status_code}")
                logger.error(f"响应内容: {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            logger.error(f"❌ 请求异常: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_tests(self):
        """运行测试用例"""
        logger.info("="*60)
        logger.info("开始测试 /generate 接口")
        logger.info("="*60)
        
        test_cases = [
            {"prompt": "Hello", "max_tokens": 5, "temperature": 0.0},
            {"prompt": "你好", "max_tokens": 5, "temperature": 0.0},
            {"prompt": "The capital of France is", "max_tokens": 10, "temperature": 0.0},
            {"prompt": "1+1=", "max_tokens": 5, "temperature": 0.0},
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"测试用例 {i}/{len(test_cases)}")
            logger.info(f"{'='*60}")
            
            result = self.test_generate(**test_case)
            results.append(result)
            
            if i < len(test_cases):
                time.sleep(1)
        
        logger.info(f"\n{'='*60}")
        logger.info("测试总结")
        logger.info(f"{'='*60}")
        
        success_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)
        
        logger.info(f"成功: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("✅ 所有测试通过!")
            return True
        else:
            logger.error("❌ 部分测试失败!")
            return False


def main():
    parser = argparse.ArgumentParser(description="测试 cLLM /generate 接口")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器URL")
    parser.add_argument("--prompt", default="Hello", help="测试提示词")
    parser.add_argument("--max-tokens", type=int, default=5, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    
    args = parser.parse_args()
    
    tester = GenerateTester(args.url)
    
    if not tester.check_health():
        logger.error("服务器健康检查失败，请确保服务器已启动")
        return 1
    
    if args.prompt:
        result = tester.test_generate(args.prompt, args.max_tokens, args.temperature)
        return 0 if result.get("success", False) else 1
    else:
        success = tester.run_tests()
        return 0 if success else 1


if __name__ == "__main__":
    exit(main())
