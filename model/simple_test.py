#!/usr/bin/env python3
"""
Qwen3-0.6B模型文件简单测试脚本
"""
import os
import json

def check_model_files():
    """Check if all required model files are present"""
    print("Checking Qwen3-0.6B model files...")
    print("="*40)
    
    model_path = "./Qwen/Qwen3-0.6B"
    
    # 模型所需的文件
    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    missing_files = []
    found_files = []
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            found_files.append((file, f"{file_size / (1024*1024):.1f} MB"))
            print(f"✓ {file} ({file_size / (1024*1024):.1f} MB)")
        else:
            missing_files.append(file)
            print(f"✗ {file} (MISSING)")
    
    print(f"\nFound {len(found_files)} files, Missing {len(missing_files)} files")
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    else:
        print("✅ All required model files are present!")
        return True

def check_config():
    """Check model configuration"""
    print("\nChecking model configuration...")
    print("="*30)
    
    config_path = "./Qwen/Qwen3-0.6B/config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("Model configuration:")
        print(f"  Model type: {config.get('model_type', 'Unknown')}")
        print(f"  Hidden size: {config.get('hidden_size', 'Unknown')}")
        print(f"  Number of heads: {config.get('num_attention_heads', 'Unknown')}")
        print(f"  Number of layers: {config.get('num_hidden_layers', 'Unknown')}")
        print(f"  Vocabulary size: {config.get('vocab_size', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to read config: {e}")
        return False

def check_tokenizer():
    """Check tokenizer files"""
    print("\nChecking tokenizer files...")
    print("="*25)
    
    tokenizer_config_path = "./Qwen/Qwen3-0.6B/tokenizer_config.json"
    
    try:
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)
        
        print("Tokenizer configuration:")
        print(f"  Tokenizer class: {config.get('tokenizer_class', 'Unknown')}")
        print(f"  Model max length: {config.get('model_max_length', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to read tokenizer config: {e}")
        return False

def main():
    """Main function"""
    print("Qwen3-0.6B Model File Verification")
    print("="*35)
    
    # 检查模型文件
    files_ok = check_model_files()
    
    # 检查配置
    config_ok = check_config()
    
    # 检查分词器
    tokenizer_ok = check_tokenizer()
    
    # 摘要
    print("\n" + "="*35)
    print("VERIFICATION SUMMARY")
    print("="*35)
    
    if files_ok and config_ok and tokenizer_ok:
        print("🎉 All checks passed!")
        print("✅ Qwen3-0.6B model is properly downloaded and ready for use!")
        print("\nNext steps:")
        print("1. Install xLLM dependencies:")
        print("   cd /Users/dannypan/PycharmProjects/sglang/xllm")
        print("   python3 -m pip install -r requirements.txt")
        print("\n2. Test with xLLM:")
        print("   cd /Users/dannypan/PycharmProjects/sglang/xllm")
        print("   python3 -m xllm.server --model-path ./model/Qwen/Qwen3-0.6B --port 8080")
    else:
        print("❌ Some checks failed.")
        print("Please verify the model files and try again.")

if __name__ == "__main__":
    main()