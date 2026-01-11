#!/usr/bin/env python3
"""
Test script for Qwen3 0.6B INT8 model with xLLM
"""
import sys
import os

# 将祖父目录添加到路径中
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, grandparent_dir)

def test_model():
    """Test the Qwen3 model with xLLM"""
    try:
        from xllm.model_executor import ModelExecutor
        
        # 使用INT8量化初始化模型执行器
        model_path = "./qwen3-0.6b-int8"
        model_executor = ModelExecutor(model_path, quantization="int8")
        
        print("✓ ModelExecutor initialized successfully")
        print("✓ Qwen3 0.6B INT8 model loaded")
        
        # 测试基本功能
        test_text = "人工智能是"
        encoded = model_executor.encode(test_text)
        print(f"✓ Encoding test: '{test_text}' -> {encoded[:10]}...")
        
        print("\n🎉 Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Qwen3 0.6B INT8 model with xLLM...")
    print("="*50)
    test_model()
