#!/usr/bin/env python3
"""
分析Qwen3-0.6B模型内存使用情况
"""

def calculate_model_memory():
    """计算模型理论内存需求"""
    # Qwen3-0.6B 配置
    config = {
        'vocab_size': 151936,
        'hidden_size': 1024,
        'num_layers': 28,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 3072,
        'max_seq_len': 4096,
        'head_dim': 1024 // 16,  # 64
    }
    
    print("=" * 60)
    print("Qwen3-0.6B 模型内存分析")
    print("=" * 60)
    print(f"\n模型配置:")
    print(f"  - Vocab Size: {config['vocab_size']:,}")
    print(f"  - Hidden Size: {config['hidden_size']}")
    print(f"  - Num Layers: {config['num_layers']}")
    print(f"  - Num Attention Heads: {config['num_attention_heads']}")
    print(f"  - Num KV Heads: {config['num_key_value_heads']}")
    print(f"  - Intermediate Size: {config['intermediate_size']}")
    print(f"  - Max Seq Length: {config['max_seq_len']}")
    print(f"  - Head Dim: {config['head_dim']}")
    
    # 计算各组件参数量
    h = config['hidden_size']
    i = config['intermediate_size']
    l = config['num_layers']
    v = config['vocab_size']
    n_head = config['num_attention_heads']
    n_kv_head = config['num_key_value_heads']
    head_dim = config['head_dim']
    
    # Embedding
    embed_params = v * h
    
    # 每层参数
    # Attention: Q, K, V, O projections
    q_proj = n_head * head_dim * h
    k_proj = n_kv_head * head_dim * h
    v_proj = n_kv_head * head_dim * h
    o_proj = h * n_head * head_dim
    attention_params_per_layer = q_proj + k_proj + v_proj + o_proj
    
    # FFN: gate, up, down
    gate_proj = i * h
    up_proj = i * h
    down_proj = h * i
    ffn_params_per_layer = gate_proj + up_proj + down_proj
    
    # LayerNorm (2 per layer + 1 final)
    layernorm_params = (2 * l + 1) * h
    
    # 总参数量
    total_params = embed_params + l * (attention_params_per_layer + ffn_params_per_layer) + layernorm_params
    
    print(f"\n" + "=" * 60)
    print("参数量分析")
    print("=" * 60)
    print(f"\nEmbedding: {embed_params:,} ({embed_params/1e6:.2f}M)")
    print(f"  - vocab_size × hidden_size = {v} × {h}")
    
    print(f"\n每层Attention: {attention_params_per_layer:,} ({attention_params_per_layer/1e6:.2f}M)")
    print(f"  - Q Proj: {q_proj:,}")
    print(f"  - K Proj: {k_proj:,}")
    print(f"  - V Proj: {v_proj:,}")
    print(f"  - O Proj: {o_proj:,}")
    
    print(f"\n每层FFN: {ffn_params_per_layer:,} ({ffn_params_per_layer/1e6:.2f}M)")
    print(f"  - Gate: {gate_proj:,}")
    print(f"  - Up: {up_proj:,}")
    print(f"  - Down: {down_proj:,}")
    
    print(f"\n总Attention: {l * attention_params_per_layer:,} ({l * attention_params_per_layer/1e6:.2f}M)")
    print(f"总FFN: {l * ffn_params_per_layer:,} ({l * ffn_params_per_layer/1e6:.2f}M)")
    print(f"LayerNorm: {layernorm_params:,} ({layernorm_params/1e6:.2f}M)")
    
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 内存计算 (不同精度)
    print(f"\n" + "=" * 60)
    print("理论内存需求")
    print("=" * 60)
    
    # BF16 (原始权重)
    bf16_bytes = total_params * 2
    print(f"\nBF16 (原始): {bf16_bytes:,} bytes = {bf16_bytes/1024/1024:.2f} MB = {bf16_bytes/1024/1024/1024:.2f} GB")
    
    # F32 (当前实现)
    f32_bytes = total_params * 4
    print(f"F32 (当前): {f32_bytes:,} bytes = {f32_bytes/1024/1024:.2f} MB = {f32_bytes/1024/1024/1024:.2f} GB")
    
    # KV Cache
    kv_cache_per_layer = 2 * config['max_seq_len'] * n_kv_head * head_dim  # K + V
    total_kv_cache = l * kv_cache_per_layer
    kv_cache_bytes = total_kv_cache * 4  # F32
    print(f"\nKV Cache (F32, max_seq={config['max_seq_len']}):")
    print(f"  每层: {kv_cache_per_layer:,} floats = {kv_cache_per_layer*4/1024/1024:.2f} MB")
    print(f"  总计: {total_kv_cache:,} floats = {kv_cache_bytes/1024/1024:.2f} MB = {kv_cache_bytes/1024/1024/1024:.2f} GB")
    
    # 工作缓冲区
    work_buffer = (
        h * 10 +  # hiddenStates, residual, normOutput, attnOutput, ffnOutput, bufferA, bufferB等
        (n_head * head_dim + 2 * n_kv_head * head_dim) +  # qkvBuffer
        n_head * head_dim * config['max_seq_len'] +  # attnScores
        i * 4 +  # gateBuffer, upBuffer, gateUpBuffer等
        v  # logitsBuffer
    )
    work_buffer_bytes = work_buffer * 4
    print(f"\n工作缓冲区 (估计): {work_buffer_bytes/1024/1024:.2f} MB")
    
    # 总内存
    total_memory = f32_bytes + kv_cache_bytes + work_buffer_bytes
    print(f"\n" + "=" * 60)
    print("内存汇总")
    print("=" * 60)
    print(f"\n权重 (F32): {f32_bytes/1024/1024/1024:.2f} GB")
    print(f"KV Cache: {kv_cache_bytes/1024/1024/1024:.2f} GB")
    print(f"工作缓冲区: {work_buffer_bytes/1024/1024:.2f} MB")
    print(f"理论总计: {total_memory/1024/1024/1024:.2f} GB")
    
    # 实际测量
    actual_memory = 9115074560  # 从time -l输出
    print(f"\n实际测量: {actual_memory/1024/1024/1024:.2f} GB")
    print(f"开销比例: {actual_memory/total_memory:.2f}x")
    
    # 优化潜力
    print(f"\n" + "=" * 60)
    print("优化潜力分析")
    print("=" * 60)
    
    # 如果保持BF16
    bf16_total = bf16_bytes + kv_cache_bytes + work_buffer_bytes
    print(f"\n如果保持BF16权重:")
    print(f"  理论内存: {bf16_total/1024/1024/1024:.2f} GB")
    print(f"  节省: {(total_memory - bf16_total)/1024/1024/1024:.2f} GB ({(total_memory - bf16_total)/total_memory*100:.1f}%)")
    
    # 如果动态KV Cache (按实际序列长度)
    avg_seq_len = 512  # 假设平均序列长度
    dynamic_kv_cache = l * 2 * avg_seq_len * n_kv_head * head_dim * 4
    dynamic_total = f32_bytes + dynamic_kv_cache + work_buffer_bytes
    print(f"\n如果动态KV Cache (avg_seq={avg_seq_len}):")
    print(f"  KV Cache: {dynamic_kv_cache/1024/1024:.2f} MB")
    print(f"  理论内存: {dynamic_total/1024/1024/1024:.2f} GB")
    print(f"  节省: {(total_memory - dynamic_total)/1024/1024/1024:.2f} GB")
    
    # 如果INT8量化
    int8_bytes = total_params * 1
    int8_total = int8_bytes + kv_cache_bytes + work_buffer_bytes
    print(f"\n如果INT8量化权重:")
    print(f"  理论内存: {int8_total/1024/1024/1024:.2f} GB")
    print(f"  节省: {(total_memory - int8_total)/1024/1024/1024:.2f} GB ({(total_memory - int8_total)/total_memory*100:.1f}%)")

if __name__ == "__main__":
    calculate_model_memory()
