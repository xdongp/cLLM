# 调试工具使用指南

本目录包含用于对比和调试 Kylin 和 llama_cpp 后端的工具。

## 工具列表

### 1. `full_comparison.sh` - 完整对比测试
自动测试两个后端并生成详细对比报告。

**使用方法：**
```bash
./tools/full_comparison.sh "Hi" 1 0.0
```

**参数：**
- 第1个参数：prompt（默认："Hi"）
- 第2个参数：max_tokens（默认：1）
- 第3个参数：temperature（默认：0.0）

**输出：**
- 在终端显示对比结果
- 生成报告文件：`/tmp/comparison_report.txt`
- 保存日志文件：`/tmp/kylin_full_comparison.log` 和 `/tmp/llama_full_comparison.log`

### 2. `analyze_debug_logs.py` - 日志分析工具
解析 Kylin 调试日志并生成统计报告。

**使用方法：**
```bash
python3 tools/analyze_debug_logs.py /tmp/kylin_debug.log
```

**功能：**
- 提取 embedding 统计信息（min, max, mean, NaN, Inf）
- 提取 Layer 0 输出统计信息
- 分析数值分布和潜在问题
- 显示前10个值用于对比

### 3. `compare_debug_outputs.sh` - 快速对比
快速对比两个后端的输出和调试信息。

**使用方法：**
```bash
./tools/compare_debug_outputs.sh "Hi" 1 0.0
```

### 4. `simple_test.sh` - 简单测试
最基本的后端对比测试。

**使用方法：**
```bash
./tools/simple_test.sh "Hi" 3 0.0
```

## 调试日志说明

Kylin 后端会在日志中输出以下调试信息：

### Embedding 统计
```
[Kylin Debug] Embedding stats: min=-0.085917, max=0.088907, mean=0.000529, nan=0, inf=0, shape=[1024,1]
[Kylin Debug] Embedding first 10 values: 0.009665 -0.003393 ...
```

### Layer 0 输出统计
```
[Kylin Debug] Layer 0 output stats: min=-1.192264, max=1.058421, mean=-0.006728, nan=0, inf=0, shape=[1024,1]
[Kylin Debug] Layer 0 output first 10 values: 0.002139 -0.105501 ...
```

### 形状信息
```
[Kylin Debug] Embedding shape: [1024, 1], type=f32
[Kylin Debug] Layer 0 output shape: [1024, 1], type=f32
```

## 对比分析要点

### 1. Embedding 输出对比
- **正常范围**：embedding 值通常在 [-1, 1] 范围内
- **均值**：应该接近 0
- **NaN/Inf**：应该为 0
- **形状**：应该是 `[hidden_size, seq_len]`

### 2. Layer 0 输出对比
- **正常范围**：经过第一层后，值范围可能扩大，但不应过大（通常 < 10）
- **均值**：可能不为 0，但不应过大
- **NaN/Inf**：应该为 0
- **形状**：应该是 `[hidden_size, seq_len]`

### 3. 输出文本对比
- 使用 temperature=0.0 进行确定性对比
- 如果输出不同，检查：
  - Embedding 是否相同
  - Layer 0 输出是否相同
  - 后续层的计算是否正确

## 常见问题排查

### 问题1：Embedding 值异常
- 检查 tokenizer 是否正确
- 检查 embedding 权重是否正确加载
- 检查 `ggml_get_rows` 是否正确使用

### 问题2：Layer 0 输出异常
- 检查 Q/K/V 投影是否正确
- 检查 Q/K norm 是否正确应用
- 检查 RoPE 参数是否正确
- 检查注意力计算是否正确

### 问题3：输出文本完全不同
- 对比 embedding 输出
- 对比第一层输出
- 逐步检查每一层的输出

## 示例工作流程

1. **运行完整对比测试：**
   ```bash
   ./tools/full_comparison.sh "What is the capital of France?" 5 0.0
   ```

2. **查看报告：**
   ```bash
   cat /tmp/comparison_report.txt
   ```

3. **分析详细日志：**
   ```bash
   python3 tools/analyze_debug_logs.py /tmp/kylin_full_comparison.log
   ```

4. **手动检查日志：**
   ```bash
   grep "\[Kylin Debug\]" /tmp/kylin_full_comparison.log
   ```

## 注意事项

- 确保服务器有足够时间启动（脚本中已包含等待时间）
- 如果遇到段错误，检查是否使用了 Metal 后端（可以切换到 CPU）
- 日志文件可能很大，注意磁盘空间
- 对比时使用相同的 prompt 和参数
