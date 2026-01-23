# Kylin 后端调试指南

## 当前状态

✅ **已完成的功能：**
- Embedding 和 Layer 0 输出的详细统计日志
- 自动化对比工具
- 日志分析工具
- 安全的数据访问（避免段错误）

❌ **已知问题：**
- Kylin 和 llama_cpp 的输出不一致
- 需要进一步对比中间结果来定位问题

## 快速开始

### 1. 运行完整对比测试

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
./tools/full_comparison.sh "Hi" 1 0.0
```

这会：
- 测试 Kylin 后端
- 测试 llama_cpp 后端
- 生成详细对比报告
- 保存日志文件

### 2. 查看对比报告

```bash
cat /tmp/comparison_report.txt
```

报告包含：
- 输出文本对比
- Embedding 统计信息
- Layer 0 输出统计信息
- 数值分布分析

### 3. 分析详细日志

```bash
python3 tools/analyze_debug_logs.py /tmp/kylin_full_comparison.log
```

## 调试信息说明

### Embedding 统计

**正常值范围：**
- Min/Max: 通常在 [-1, 1] 范围内
- Mean: 应该接近 0（通常在 [-0.1, 0.1]）
- NaN/Inf: 必须为 0
- Shape: `[1024, 1]` (对于 Qwen3-0.6B)

**当前观察值（示例）：**
```
Min: -0.085917
Max: 0.088907
Mean: 0.000529
NaN: 0
Inf: 0
```

✅ 这些值看起来正常

### Layer 0 输出统计

**正常值范围：**
- Min/Max: 可能比 embedding 范围大，但通常 < 10
- Mean: 可能不为 0，但不应过大
- NaN/Inf: 必须为 0
- Shape: `[1024, 1]`

**当前观察值（示例）：**
```
Min: -1.192264
Max: 1.058421
Mean: -0.006728
NaN: 0
Inf: 0
```

✅ 这些值看起来正常

## 问题定位步骤

### 步骤 1: 对比 Embedding 输出

如果 embedding 输出不同，问题在：
- Tokenizer 实现
- Embedding 权重加载
- `ggml_get_rows` 的使用

**检查方法：**
```bash
# 查看 embedding 统计
grep "\[Kylin Debug\] Embedding" /tmp/kylin_full_comparison.log

# 对比前10个值
# 如果值完全不同，检查 tokenizer 和 embedding 权重
```

### 步骤 2: 对比 Layer 0 输出

如果 Layer 0 输出不同，问题可能在：
- Q/K/V 投影
- Q/K norm 应用
- RoPE 实现
- 注意力计算
- GQA 扩展

**检查方法：**
```bash
# 查看 layer 0 统计
grep "\[Kylin Debug\] Layer 0" /tmp/kylin_full_comparison.log

# 如果统计信息正常但输出不同，检查：
# 1. 注意力计算的维度
# 2. Softmax 的应用
# 3. 残差连接
```

### 步骤 3: 逐步对比

使用更长的 prompt 来观察差异：

```bash
# 测试更长的 prompt
./tools/full_comparison.sh "What is" 3 0.0

# 查看每一步的输出
# 如果第一步就不同，问题在 embedding 或第一层
# 如果后续步骤才不同，问题在后续层或 KV cache
```

## 常见问题排查

### 问题：Embedding 值异常

**可能原因：**
1. Tokenizer 返回的 token ID 不正确
2. Embedding 权重未正确加载
3. `ggml_get_rows` 参数错误

**检查方法：**
```bash
# 检查 tokenizer 输出
# 在代码中添加日志打印 inputIds

# 检查 embedding 权重
# 验证 tokEmbed_ 是否正确加载
```

### 问题：Layer 0 输出异常

**可能原因：**
1. Q/K/V 投影维度错误
2. Q/K norm 未正确应用
3. RoPE 参数错误
4. 注意力计算错误
5. GQA 扩展错误

**检查方法：**
```bash
# 查看注意力计算的调试日志
grep "\[Attention L0\]" /tmp/kylin_full_comparison.log

# 检查：
# - Q/K/V 的形状
# - RoPE 参数
# - GQA 扩展
# - 注意力分数
```

### 问题：输出文本完全不同

**可能原因：**
1. 累积误差（每层的小误差累积）
2. 数值精度问题
3. 计算顺序问题

**检查方法：**
```bash
# 对比每一层的输出（需要添加更多日志）
# 或者对比 logits 输出

# 查看 logits 统计
grep "Logits stats" /tmp/kylin_full_comparison.log
```

## 下一步建议

1. **添加更多层的调试日志**
   - 在每层后打印输出统计
   - 对比每一层的差异

2. **对比 logits 输出**
   - 添加 logits 的详细统计
   - 对比 top-k tokens

3. **添加注意力分数日志**
   - 打印注意力权重的统计
   - 检查注意力模式

4. **数值精度对比**
   - 使用更高精度（FP32）进行对比
   - 检查量化误差

## 工具使用示例

### 示例 1: 快速对比
```bash
./tools/full_comparison.sh "Hello" 3 0.0
```

### 示例 2: 分析特定日志
```bash
python3 tools/analyze_debug_logs.py /tmp/kylin_debug.log
```

### 示例 3: 查看原始日志
```bash
# 查看所有调试信息
grep "\[Kylin Debug\]" /tmp/kylin_full_comparison.log

# 查看注意力计算信息
grep "\[Attention" /tmp/kylin_full_comparison.log | head -20
```

## 注意事项

1. **使用 CPU 后端进行调试**
   - Metal 后端可能有段错误
   - CPU 后端更稳定，便于调试

2. **使用 temperature=0.0**
   - 确保确定性输出
   - 便于对比

3. **使用短 prompt**
   - 减少日志量
   - 加快测试速度

4. **保存日志文件**
   - 日志文件可能很大
   - 定期清理旧日志
