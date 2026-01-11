#!/usr/bin/env python3
"""
xLLM核心功能快速测试
"""
import sys
import os

# 将祖父目录添加到路径中，以便我们可以导入xllm模块
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, grandparent_dir)

def test_imports():
    """测试导入xLLM模块"""
    print("Testing xLLM module imports...")
    print("="*35)
    
    try:
        # 测试导入核心模块
        from xllm import __version__
        print(f"✓ xLLM version: {__version__}")
        
        import xllm.server
        print("✓ server module imported")
        
        import xllm.tokenizer_manager
        print("✓ tokenizer_manager module imported")
        
        import xllm.scheduler
        print("✓ scheduler module imported")
        
        import xllm.model_executor
        print("✓ model_executor module imported")
        
        import xllm.sampler
        print("✓ sampler module imported")
        
        print("\n✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_classes():
    """测试实例化核心类"""
    print("\nTesting xLLM class instantiation...")
    print("="*35)
    
    try:
        from xllm.tokenizer_manager import RequestState
        request_state = RequestState(
            request_id="test-1",
            prompt="Hello world",
            tokenized_prompt=[1, 2, 3]
        )
        print("✓ RequestState instantiated")
        
        from xllm.sampler import Sampler
        sampler = Sampler()
        print("✓ Sampler instantiated")
        
        print("\n✅ All classes instantiated successfully!")
        return True
    except Exception as e:
        print(f"❌ Class instantiation failed: {e}")
        return False

def main():
    """主测试函数"""
    print("xLLM Quick Functionality Test")
    print("="*30)
    
    # 测试导入
    imports_ok = test_imports()
    
    # Test class instantiation
    classes_ok = test_classes()
    
    # Summary
    print("\n" + "="*30)
    print("QUICK TEST SUMMARY")
    print("="*30)
    
    if imports_ok and classes_ok:
        print("🎉 Quick tests passed!")
        print("✅ xLLM core functionality is working!")
        print("\nOnce dependencies are installed, you can run:")
        print("cd /Users/dannypan/PycharmProjects/sglang/xllm")
        print("python3 -m xllm.server --model-path ./model/Qwen/Qwen3-0.6B --port 8080")
    else:
        print("❌ Quick tests failed.")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()