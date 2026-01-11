#!/usr/bin/env python3
"""
使用xLLM测试Qwen3-0.6B模型的脚本
"""
import sys
import os
import time

# 将祖父目录添加到路径中，以便我们可以导入xllm模块
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, grandparent_dir)

def test_model_loading():
    """Test loading the Qwen3 model"""
    print("Testing Qwen3-0.6B model loading...")
    print("="*50)
    
    try:
        from xllm.model_executor import ModelExecutor
        
        # 下载的Qwen3模型路径
        model_path = "./Qwen/Qwen3-0.6B"
        print(f"Loading model from: {model_path}")
        
        # 初始化模型执行器
        start_time = time.time()
        model_executor = ModelExecutor(model_path)
        end_time = time.time()
        
        print(f"✓ Model loaded successfully in {end_time - start_time:.2f} seconds")
        print(f"Model path: {model_executor.model_path}")
        print(f"Device: {model_executor.device}")
        
        return model_executor
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_tokenization():
    """Test tokenization functionality"""
    print("\nTesting tokenization...")
    print("="*30)
    
    try:
        from xllm.tokenizer_manager import TokenizerManager
        
        # 初始化分词器管理器
        model_path = "./Qwen/Qwen3-0.6B"
        tokenizer_manager = TokenizerManager(model_path)
        
        # 测试编码
        test_text = "人工智能是计算机科学的一个分支"
        encoded = tokenizer_manager.encode(test_text)
        print(f"Input text: {test_text}")
        print(f"Encoded tokens: {encoded[:10]}... (showing first 10)")
        print(f"Total tokens: {len(encoded)}")
        
        # 测试解码
        decoded = tokenizer_manager.decode(encoded)
        print(f"Decoded text: {decoded}")
        
        print("✓ Tokenization test passed")
        return True
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

def test_simple_inference():
    """Test simple inference functionality"""
    print("\nTesting simple inference...")
    print("="*30)
    
    try:
        from xllm.model_executor import ModelExecutor
        
        # 初始化模型执行器
        model_path = "./Qwen/Qwen3-0.6B"
        model_executor = ModelExecutor(model_path)
        
        # 创建简单测试输入（使用前几个令牌）
        test_input_ids = [1, 2, 3, 4, 5]  # 用于测试的简单令牌ID
        batch_inputs = {
            "input_ids": test_input_ids,
            "request_positions": [(0, len(test_input_ids))],
            "batch_size": 1
        }
        
        print(f"Input tokens: {test_input_ids}")
        
        # 运行前向传递
        start_time = time.time()
        outputs = model_executor.forward(batch_inputs)
        end_time = time.time()
        
        print(f"✓ Forward pass completed in {end_time - start_time:.2f} seconds")
        print(f"Output keys: {list(outputs.keys())}")
        
        if "logits" in outputs:
            logits = outputs["logits"]
            print(f"Logits shape: {logits.shape}")
            print("✓ Inference test passed")
            return True
        else:
            print("❌ Unexpected output format")
            return False
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Qwen3-0.6B Model Test with xLLM")
    print("="*40)
    
    # 测试模型加载
    model_executor = test_model_loading()
    if not model_executor:
        return
    
    # 测试分词
    tokenization_success = test_tokenization()
    
    # 测试简单推理
    inference_success = test_simple_inference()
    
    # 摘要
    print("\n" + "="*40)
    print("TEST SUMMARY")
    print("="*40)
    if tokenization_success and inference_success:
        print("🎉 All tests passed!")
        print("✅ Qwen3-0.6B model is ready for use with xLLM")
    else:
        print("❌ Some tests failed.")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()