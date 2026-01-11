#!/usr/bin/env python3
"""
cLLM 测试数据生成器

功能：
- 生成 Tokenizer 测试数据
- 生成推理测试数据
- 生成性能测试数据
- 生成压力测试数据

使用方法：
    python3 scripts/generate_test_data.py
"""

import json
import random
import string
import os
from pathlib import Path


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, output_dir="tests/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = {
            "tokenizer": {},
            "inference": {},
            "performance": {},
            "stress": {},
            "scenarios": {}
        }
    
    def generate_tokenizer_data(self):
        """生成 Tokenizer 测试数据"""
        print("📝 生成 Tokenizer 测试数据...")
        
        # 短文本（多语言）
        short_texts = [
            "Hello, world!",
            "你好，世界！",
            "こんにちは、世界！",
            "Bonjour le monde!",
            "¡Hola mundo!",
            "Test input",
            "Simple test",
            "Quick brown fox",
        ]
        
        # 长文本
        long_texts = [
            " ".join(["This is a long text for testing tokenizer performance."] * 50),
            " ".join(["这是一个用于测试分词器性能的长文本。"] * 50),
            " ".join(["Mixed English and 中文 text for testing."] * 30),
        ]
        
        # 特殊情况
        special_cases = [
            "😀🎉🚀💻",  # Emoji
            "Text with\nnewlines\nand\ttabs",  # 换行和制表符
            "Mixed中英文日本語text",  # 多语言混合
            "Special chars: !@#$%^&*()",  # 特殊字符
            "Numbers: 0123456789",  # 数字
            "URL: https://example.com/path?param=value",  # URL
            "Email: test@example.com",  # Email
            "Code: def hello(): print('Hello')",  # 代码
        ]
        
        # 边界情况
        boundary_cases = [
            "",  # 空字符串
            " ",  # 单个空格
            "   ",  # 多个空格
            "\n",  # 单个换行
            "\t",  # 单个制表符
            "A",  # 单个字符
            "很",  # 单个中文字符
            "a" * 1000,  # 很长的单个单词
        ]
        
        self.data["tokenizer"] = {
            "short_texts": short_texts,
            "long_texts": long_texts,
            "special_cases": special_cases,
            "boundary_cases": boundary_cases
        }
        
        print(f"  ✅ 生成 {len(short_texts)} 个短文本")
        print(f"  ✅ 生成 {len(long_texts)} 个长文本")
        print(f"  ✅ 生成 {len(special_cases)} 个特殊情况")
        print(f"  ✅ 生成 {len(boundary_cases)} 个边界情况")
    
    def generate_inference_data(self):
        """生成推理测试数据"""
        print("🧠 生成推理测试数据...")
        
        prompts = [
            {
                "id": "qa_factual_1",
                "text": "What is the capital of France?",
                "max_length": 50,
                "temperature": 0.3,
                "expected_keywords": ["Paris"],
                "category": "qa_factual"
            },
            {
                "id": "qa_reasoning_1",
                "text": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "max_length": 100,
                "temperature": 0.3,
                "expected_keywords": ["5 minutes"],
                "category": "qa_reasoning"
            },
            {
                "id": "chat_casual_1",
                "text": "Hello! How are you today?",
                "max_length": 100,
                "temperature": 0.7,
                "expected_keywords": ["Hello", "good", "fine"],
                "category": "chat_casual"
            },
            {
                "id": "code_generation_1",
                "text": "Write a Python function to calculate the factorial of a number",
                "max_length": 200,
                "temperature": 0.3,
                "expected_keywords": ["def", "factorial", "return"],
                "category": "code_generation"
            },
            {
                "id": "summarization_1",
                "text": "Summarize the following text in one sentence: Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.",
                "max_length": 50,
                "temperature": 0.5,
                "expected_keywords": ["AI", "machines", "intelligence"],
                "category": "summarization"
            },
            {
                "id": "translation_1",
                "text": "Translate to Chinese: Hello, how are you?",
                "max_length": 50,
                "temperature": 0.3,
                "expected_keywords": ["你好"],
                "category": "translation"
            },
            {
                "id": "chinese_1",
                "text": "请介绍一下人工智能的发展历史",
                "max_length": 200,
                "temperature": 0.7,
                "expected_keywords": ["人工智能", "发展", "历史"],
                "category": "chinese_qa"
            },
        ]
        
        self.data["inference"]["prompts"] = prompts
        
        # 按类别统计
        categories = {}
        for p in prompts:
            cat = p["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"  ✅ 生成 {len(prompts)} 个推理测试用例")
        for cat, count in categories.items():
            print(f"    - {cat}: {count} 个")
    
    def generate_performance_data(self):
        """生成性能测试数据"""
        print("⚡ 生成性能测试数据...")
        
        self.data["performance"] = {
            "batch_sizes": [1, 2, 4, 8, 16, 32],
            "sequence_lengths": [10, 50, 100, 200, 500, 1000, 2000],
            "concurrency_levels": [1, 5, 10, 20, 50, 100],
            "test_durations_seconds": [10, 30, 60, 300],
        }
        
        print("  ✅ 生成性能测试配置")
        print(f"    - Batch sizes: {self.data['performance']['batch_sizes']}")
        print(f"    - Sequence lengths: {self.data['performance']['sequence_lengths']}")
        print(f"    - Concurrency levels: {self.data['performance']['concurrency_levels']}")
    
    def generate_stress_data(self):
        """生成压力测试数据"""
        print("💪 生成压力测试数据...")
        
        self.data["stress"] = {
            "duration_minutes": [5, 15, 30, 60, 120],
            "request_rates": [10, 50, 100, 200, 500, 1000],
            "payload_sizes": [100, 500, 1000, 5000, 10000],
            "patterns": [
                {
                    "name": "constant_load",
                    "description": "恒定负载",
                    "rate": 100,
                    "duration": 300
                },
                {
                    "name": "spike_load",
                    "description": "尖峰负载",
                    "base_rate": 50,
                    "spike_rate": 500,
                    "spike_duration": 60
                },
                {
                    "name": "ramp_up",
                    "description": "逐步增加负载",
                    "start_rate": 10,
                    "end_rate": 200,
                    "duration": 300
                }
            ]
        }
        
        print("  ✅ 生成压力测试配置")
        print(f"    - 测试模式: {len(self.data['stress']['patterns'])} 种")
    
    def generate_scenario_data(self):
        """生成场景测试数据"""
        print("🎬 生成场景测试数据...")
        
        scenarios = {
            "single_turn_qa": {
                "name": "单轮问答",
                "conversations": [
                    {
                        "user": "What is machine learning?",
                        "expected_keywords": ["machine learning", "algorithm", "data"]
                    },
                    {
                        "user": "什么是深度学习？",
                        "expected_keywords": ["深度学习", "神经网络"]
                    }
                ]
            },
            "multi_turn_chat": {
                "name": "多轮对话",
                "conversations": [
                    {
                        "turns": [
                            {"role": "user", "content": "Hello!"},
                            {"role": "assistant", "content": "Hello! How can I help you?"},
                            {"role": "user", "content": "Can you write code?"},
                            {"role": "assistant", "content": "Yes, I can help with programming!"},
                            {"role": "user", "content": "Write a hello world in Python"}
                        ]
                    }
                ]
            },
            "code_assistance": {
                "name": "代码辅助",
                "tasks": [
                    {
                        "description": "Write a sorting algorithm",
                        "language": "Python",
                        "expected_keywords": ["def", "sort", "return"]
                    },
                    {
                        "description": "Debug this code: for i in range(10 print(i)",
                        "language": "Python",
                        "expected_keywords": ["syntax", "error", "parentheses"]
                    }
                ]
            }
        }
        
        self.data["scenarios"] = scenarios
        
        print(f"  ✅ 生成 {len(scenarios)} 个测试场景")
        for name, scenario in scenarios.items():
            print(f"    - {scenario['name']}")
    
    def save(self):
        """保存测试数据"""
        print("💾 保存测试数据...")
        
        # 保存主文件
        main_file = self.output_dir / "test_cases.json"
        with open(main_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 主文件: {main_file}")
        
        # 保存分类文件
        for category, data in self.data.items():
            category_file = self.output_dir / f"{category}_test_data.json"
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {category} 数据: {category_file}")
    
    def generate_all(self):
        """生成所有测试数据"""
        print("=" * 60)
        print("🚀 开始生成测试数据")
        print("=" * 60)
        
        self.generate_tokenizer_data()
        self.generate_inference_data()
        self.generate_performance_data()
        self.generate_stress_data()
        self.generate_scenario_data()
        
        self.save()
        
        print("=" * 60)
        print("✅ 测试数据生成完成！")
        print("=" * 60)
    
    def print_summary(self):
        """打印摘要"""
        print("\n📊 测试数据摘要:")
        print(f"  - Tokenizer 测试用例: {sum(len(v) for v in self.data['tokenizer'].values())} 个")
        print(f"  - 推理测试用例: {len(self.data['inference'].get('prompts', []))} 个")
        print(f"  - 性能测试配置: 已生成")
        print(f"  - 压力测试配置: {len(self.data['stress'].get('patterns', []))} 种模式")
        print(f"  - 场景测试: {len(self.data['scenarios'])} 个场景")
        print(f"\n📁 输出目录: {self.output_dir}")


def main():
    """主函数"""
    generator = TestDataGenerator()
    generator.generate_all()
    generator.print_summary()


if __name__ == "__main__":
    main()
