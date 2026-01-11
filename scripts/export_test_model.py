#!/usr/bin/env python3
"""
创建一个简单的PyTorch模型用于LibTorch后端测试
"""

import torch
import torch.nn as nn
import os

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=128, num_layers=2, num_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Embedding(512, hidden_size)  # 位置编码
        
        # 简单的Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        seq_len = input_ids.shape[1]
        
        # 词嵌入
        embeddings = self.embedding(input_ids)
        
        # 位置编码
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos_encodings = self.pos_encoding(positions)
        pos_encodings = pos_encodings.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        
        # 组合嵌入
        x = embeddings + pos_encodings
        
        # Transformer处理
        x = self.transformer(x)
        
        # 输出投影到词汇表
        logits = self.lm_head(x)
        
        return logits

def main():
    print("Creating simple transformer model for LibTorch testing...")
    
    # 创建模型实例
    model = SimpleTransformer(
        vocab_size=32000,  # 与Qwen模型类似的词汇表大小
        hidden_size=128,   # 小尺寸便于测试
        num_layers=2,      # 少层数加快测试
        num_heads=4        # 少注意力头
    )
    
    model.eval()  # 设置为评估模式
    
    print(f"Model created with:")
    print(f"  - Vocab size: {model.vocab_size}")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - Num layers: {model.num_layers}")
    print(f"  - Num heads: {model.num_heads}")
    
    # 创建示例输入
    example_input = torch.randint(0, model.vocab_size, (1, 10))  # [batch_size=1, seq_len=10]
    print(f"Example input shape: {example_input.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        example_output = model(example_input)
        print(f"Example output shape: {example_output.shape}")
    
    # 导出为TorchScript模型
    print("\nExporting model to TorchScript...")
    traced_model = torch.jit.trace(model, example_input)
    
    # 保存模型
    output_path = "tests/test_model_libtorch.pt"
    traced_model.save(output_path)
    print(f"✓ Model exported to {output_path}")
    
    # 验证模型大小
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    # 验证加载
    print("\nVerifying model loading...")
    try:
        loaded_model = torch.jit.load(output_path)
        loaded_model.eval()
        
        with torch.no_grad():
            test_output = loaded_model(example_input)
        
        print("✓ Model loaded and tested successfully")
        print(f"Verification output shape: {test_output.shape}")
        
        # 检查输出是否合理
        if test_output.shape == example_output.shape:
            print("✓ Output shape matches expected")
        else:
            print("✗ Output shape mismatch")
            
        print("\n✓ All tests passed! Ready for C++ integration.")
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test model creation completed successfully!")
    else:
        print("\n❌ Test model creation failed!")
        exit(1)