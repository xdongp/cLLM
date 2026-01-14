# GGUF文件分析工具使用说明

## 工具位置

分析工具位于：`tools/analyze_gguf_tensors.py`

## 使用方法

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
python3 tools/analyze_gguf_tensors.py model/Qwen/qwen3-0.6b-q4_k_m.gguf
```

## 工具功能

该工具会分析GGUF文件并显示：

1. **Embedding相关张量**：查找所有包含 "embed"、"tok"、"token" 的张量
2. **Output/LM Head相关张量**：查找输出层相关张量
3. **Norm相关张量**：查找归一化层相关张量
4. **前5层的张量**：显示前5层的所有张量，用于确认命名格式
5. **所有张量名称（前50个）**：列出前50个张量的详细信息

## 输出示例

工具会输出类似以下内容：

```
=== Embedding相关张量 ===
  tok_emb.weight                    | Shape: [151936, 768] | Type: Q4_K

=== 前5层的张量（用于确认命名格式） ===
层 0:
  blk.0.attn_q.weight               | Shape: [768, 768] | Type: Q4_K
  blk.0.attn_k.weight               | Shape: [768, 768] | Type: Q4_K
  ...
```

## 根据输出更新代码

根据工具输出的实际张量名称，更新 `src/model/gguf_loader_new.cpp` 中的 `embeddingNames` 列表。

## 替代方法：使用llama.cpp的工具

如果上面的工具无法运行，也可以使用llama.cpp自带的工具：

```bash
cd third_party/llama.cpp/gguf-py
python3 examples/reader.py ../../model/Qwen/qwen3-0.6b-q4_k_m.gguf
```

或者使用gguf_dump：

```bash
cd third_party/llama.cpp/gguf-py
python3 -m gguf.scripts.gguf_dump ../../model/Qwen/qwen3-0.6b-q4_k_m.gguf
```
