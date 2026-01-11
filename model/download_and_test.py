#!/usr/bin/env python3
"""
下载并测试Qwen3 0.6B INT8模型的脚本
"""
import os
import sys
import subprocess
import argparse

# 将祖父目录添加到路径中，以便我们可以导入xllm模块
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, grandparent_dir)

def create_model_directory():
    """Create model directory if it doesn't exist"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3-0.6b-int8")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")
    else:
        print(f"Model directory already exists: {model_dir}")
    return model_dir

def download_model_instructions():
    """Print instructions for downloading the model"""
    print("\n" + "="*60)
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("To download the Qwen3 0.6B INT8 model, you can use one of the following methods:")
    print("\n1. Using Hugging Face Transformers:")
    print("   - Visit: https://huggingface.co/Qwen/Qwen3-0.6B")
    print("   - Download the INT8 quantized version")
    print("\n2. Using git-lfs (if available):")
    print("   git clone https://huggingface.co/Qwen/Qwen3-0.6B")
    print("   cd Qwen3-0.6B")
    print("\n3. Manual download:")
    print("   - Go to the Hugging Face model page")
    print("   - Download the model files manually")
    print("\nAfter downloading, place the model files in the 'model/qwen3-0.6b-int8' directory.")
    print("="*60)

def test_model_loading():
    """Test if the model can be loaded"""
    try:
        print("\nTesting model loading...")
        
        # 获取模型目录
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3-0.6b-int8")
        
        if not os.path.exists(model_dir):
            print(f"Model directory not found: {model_dir}")
            print("Please download the model first!")
            return False
        
        # 检查模型文件是否存在
        required_files = ["config.json", "pytorch_model.bin"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                missing_files.append(file)
        
        if missing_files:
            print(f"Missing required model files: {missing_files}")
            print("Please ensure all model files are downloaded correctly.")
            return False
        
        print(f"✓ Model directory found: {model_dir}")
        print("✓ Required model files present")
        print("Model is ready for testing!")
        return True
        
    except Exception as e:
        print(f"Error testing model loading: {e}")
        return False

def create_test_script():
    """Create a simple test script for the model"""
    test_script_content = '''#!/usr/bin/env python3
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
        
        print("\\n🎉 Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Qwen3 0.6B INT8 model with xLLM...")
    print("="*50)
    test_model()
'''
    
    test_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_model.py")
    with open(test_script_path, 'w') as f:
        f.write(test_script_content)
    
    print(f"Created test script: {test_script_path}")

def main():
    """Main function"""
    print("xLLM Model Setup Script")
    print("="*30)
    
    # 创建模型目录
    model_dir = create_model_directory()
    
    # 显示下载说明
    download_model_instructions()
    
    # 测试模型加载
    test_model_loading()
    
    # 创建测试脚本
    create_test_script()
    
    print("\nSetup complete! Please follow the download instructions above.")

if __name__ == "__main__":
    main()