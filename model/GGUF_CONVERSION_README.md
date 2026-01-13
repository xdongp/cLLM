# Qwen3-0.6B 模型转换为 GGUF 格式指南

## 概述

本指南说明如何使用 `convert_qwen_to_gguf.py` 脚本将 HuggingFace 格式的 Qwen3-0.6B 模型转换为 GGUF 格式。

## 前置要求

### 1. 依赖安装

确保已安装以下 Python 包：

```bash
pip install torch transformers numpy
```

### 2. llama.cpp 可用性

脚本需要 llama.cpp 的转换工具。确保以下之一可用：

- **选项 A（推荐）**: llama.cpp 在 `third_party/llama.cpp/` 目录下
  ```bash
  git submodule update --init --recursive third_party/llama.cpp
  ```

- **选项 B**: llama.cpp 已安装为 Python 包
  ```bash
  pip install llama-cpp-python
  ```

- **选项 C**: `convert_hf_to_gguf.py` 在系统 PATH 中

## 使用方法

### 基本用法

```bash
# 转换到 F32 格式（完整精度）
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-f32.gguf \
    --outtype f32
```

### 支持的输出类型

| 类型 | 说明 | 文件大小 | 精度 | 推荐场景 |
|------|------|---------|------|---------|
| `f32` | 完整精度 (FP32) | 最大 | 最高 | 开发测试 |
| `f16` | 半精度 (FP16) | ~50% | 高 | 生产环境 |
| `bf16` | Brain Float 16 | ~50% | 高 | 训练/推理 |
| `q8_0` | 8-bit 量化 | ~25% | 中 | 资源受限 |
| `q4_k_m` | 4-bit K-quant (中等质量) | ~25% | 中高 | **推荐用于4-bit量化** |
| `tq1_0` | Tiny Quant 1.0 | ~12.5% | 低 | 极低资源 |
| `tq2_0` | Tiny Quant 2.0 | ~25% | 中 | 低资源 |
| `auto` | 自动选择 | 可变 | 可变 | 不确定时 |

### 常用命令示例

#### 1. 转换为 F32 格式（完整精度）

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-f32.gguf \
    --outtype f32
```

**适用场景**: 开发测试、精度要求最高的场景

#### 2. 转换为 F16 格式（半精度，推荐）

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-f16.gguf \
    --outtype f16
```

**适用场景**: 生产环境，平衡精度和性能

#### 3. 转换为 Q8_0 格式（8-bit 量化）

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-q8_0.gguf \
    --outtype q8_0
```

**适用场景**: 资源受限环境，需要较小文件大小

#### 4. 转换为 Q4_K_M 格式（4-bit 量化，推荐）

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-q4_k_m.gguf \
    --outtype q4_k_m
```

**适用场景**: 需要最小文件大小，同时保持较好精度（推荐用于4-bit量化）

**注意**: 如果直接转换失败，可能需要两步转换：
1. 先转换为 F16: `--outtype f16`
2. 然后使用 llama.cpp 的量化工具转换为 Q4_K_M

#### 5. 使用默认路径（简化命令）

```bash
# 使用默认路径，输出到 model/Qwen/qwen3-0.6b-f32.gguf
python model/convert_qwen_to_gguf.py --outtype f32
```

#### 6. 启用详细输出

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-f32.gguf \
    --outtype f32 \
    --verbose
```

#### 7. 仅导出词汇表（测试用）

```bash
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-vocab.gguf \
    --vocab-only
```

#### 8. 检查依赖

```bash
python model/convert_qwen_to_gguf.py --check-deps
```

## 转换流程

转换过程包括以下步骤：

1. **验证模型目录**: 检查 `config.json` 等必需文件
2. **加载模型配置**: 从 HuggingFace 配置读取模型参数
3. **加载模型权重**: 从 `model.safetensors` 或 `pytorch_model.bin` 加载权重
4. **转换权重格式**: 将 HuggingFace 权重转换为 GGUF 格式
5. **写入 GGUF 文件**: 生成最终的 `.gguf` 文件

## 输出文件说明

转换完成后，会生成以下文件：

- **主文件**: `qwen3-0.6b-{outtype}.gguf` - 包含模型权重和元数据
- **大小**: 取决于量化类型（F32 约 2.4GB，F16 约 1.2GB，Q8_0 约 600MB）

## 验证转换结果

转换完成后，可以使用以下方法验证：

### 1. 检查文件大小

```bash
ls -lh model/Qwen/qwen3-0.6b-*.gguf
```

### 2. 使用 llama.cpp 验证（如果可用）

```bash
# 检查 GGUF 文件信息
python third_party/llama.cpp/convert_hf_to_gguf.py --help
```

### 3. 使用 cLLM 加载测试

转换后的 GGUF 文件可以在 cLLM 项目中使用：

```cpp
// C++ 代码示例
GGUFLoader loader("model/Qwen/qwen3-0.6b-f32.gguf", config);
loader.load();
ModelWeights weights;
loader.loadWeights(weights);
```

## 常见问题

### Q1: 找不到 convert_hf_to_gguf.py

**解决方案**:
```bash
# 确保 llama.cpp 子模块已初始化
git submodule update --init --recursive third_party/llama.cpp
```

### Q2: 转换失败，提示缺少依赖

**解决方案**:
```bash
pip install torch transformers numpy
```

### Q3: 转换速度慢

**原因**: 大模型转换需要时间，F32 格式转换可能需要几分钟到十几分钟。

**优化建议**:
- 使用 F16 或 Q8_0 格式（转换更快）
- 确保有足够的磁盘空间
- 使用 SSD 存储

### Q4: 内存不足

**解决方案**:
- 关闭其他占用内存的程序
- 使用量化格式（Q8_0）减少内存需求
- 增加系统交换空间

### Q5: 转换后的文件无法加载

**检查清单**:
1. 确认转换过程没有错误
2. 检查输出文件大小是否合理
3. 验证 GGUF 文件格式版本
4. 检查 cLLM 代码中的 GGUF 加载器实现

## 性能对比

| 格式 | 文件大小 | 加载时间 | 推理速度 | 内存占用 |
|------|---------|---------|---------|---------|
| F32 | 2.4 GB | 慢 | 慢 | 高 |
| F16 | 1.2 GB | 中 | 中 | 中 |
| Q8_0 | 600 MB | 快 | 快 | 低 |
| Q4_K_M | 300 MB | 快 | 快 | 低 |

## 下一步

转换完成后，可以：

1. **测试加载**: 使用 cLLM 的 GGUFLoader 加载模型
2. **性能测试**: 对比不同格式的性能
3. **集成到项目**: 在 ModelLoaderFactory 中使用 GGUF 格式

## 相关文档

- [GGUF 格式支持详细设计](../docs/design/GGUF格式支持详细设计.md)
- [GGUF 格式支持任务分解](../docs/design/GGUF格式支持任务分解.md)
- [llama.cpp 转换工具文档](https://github.com/ggerganov/llama.cpp/blob/master/gguf.md)

---

**注意**: 转换过程可能需要较长时间，请耐心等待。建议在转换前确保有足够的磁盘空间（至少是模型大小的 2 倍）。
