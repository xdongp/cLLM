# GGUF 格式支持快速参考卡片

> **快速查找**: 本文档提供 GGUF 格式支持功能的快速参考，详细内容请查看 [文档索引](./GGUF格式支持文档索引.md)

---

## 📋 文档清单

| 文档 | 路径 | 用途 |
|------|------|------|
| **详细设计** | [GGUF格式支持详细设计.md](./GGUF格式支持详细设计.md) | 完整设计方案 |
| **任务分解** | [GGUF格式支持任务分解.md](./GGUF格式支持任务分解.md) | 开发任务清单 |
| **文档索引** | [GGUF格式支持文档索引.md](./GGUF格式支持文档索引.md) | 所有文档导航 |
| **快速参考** | [GGUF格式支持快速参考.md](./GGUF格式支持快速参考.md) | 本文档 |
| **转换指南** | [../../model/GGUF_CONVERSION_README.md](../../model/GGUF_CONVERSION_README.md) | 模型转换工具 |

---

## 🎯 核心概念速查

### 架构组件

```
ModelLoaderFactory (工厂)
    ↓
IModelLoader (接口)
    ↓
GGUFLoader (实现)
    ↓
ModelWeights (通用数据结构)
    ↓
后端适配器
    ├── Kylin 适配器 → kylin::Tensor
    └── LibTorch 适配器 → torch::Tensor
```

### 关键数据结构

```cpp
// 通用权重数据（后端无关）
struct WeightData {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::string name;
    WeightDType dtype;
};

struct ModelWeights {
    WeightData embedding;
    std::vector<LayerWeights> layers;
    WeightData finalNorm;
    WeightData lmHead;
};
```

### 核心接口

```cpp
class IModelLoader {
    virtual bool load() = 0;
    virtual bool loadWeights(ModelWeights& weights) = 0;
    virtual const ModelConfig& getConfig() const = 0;
};
```

---

## 🚀 快速开始

### 1. 转换模型为 GGUF 格式

```bash
# 转换为 F16 格式（推荐）
python model/convert_qwen_to_gguf.py \
    --model-dir model/Qwen/Qwen3-0.6B \
    --output model/Qwen/qwen3-0.6b-f16.gguf \
    --outtype f16
```

### 2. 在代码中使用

```cpp
// 创建加载器
auto loader = ModelLoaderFactory::create("model.qwen3-0.6b-f16.gguf");
loader->load();

// 加载权重
ModelWeights weights;
loader->loadWeights(weights);

// 转换为后端张量
#ifdef ENABLE_KYLIN_BACKEND
    auto kylin_tensors = convertToKylinTensors(weights);
#endif

#ifdef ENABLE_LIBTORCH_BACKEND
    auto torch_tensors = convertToTorchTensors(weights, device);
#endif
```

---

## 📊 开发阶段速查

| 阶段 | 目标 | 关键任务 | 文档位置 |
|------|------|---------|---------|
| **阶段1** | 基础架构 | ModelLoaderFactory、IModelLoader | [任务分解-阶段1](./GGUF格式支持任务分解.md#阶段1-基础架构搭建) |
| **阶段2** | GGUF加载器 | GGUFLoader实现、文件解析 | [任务分解-阶段2](./GGUF格式支持任务分解.md#阶段2-gguf-加载器实现) |
| **阶段3** | 量化支持 | 反量化算法、量化类型 | [任务分解-阶段3](./GGUF格式支持任务分解.md#阶段3-量化反量化支持) |
| **阶段4** | 后端集成 | Kylin适配、LibTorch适配 | [任务分解-阶段4](./GGUF格式支持任务分解.md#阶段4-后端集成) |
| **阶段5** | 测试验证 | 单元测试、集成测试 | [任务分解-阶段5](./GGUF格式支持任务分解.md#阶段5-测试与验证) |

---

## 🔧 常用命令

### 模型转换

```bash
# 检查依赖
python model/convert_qwen_to_gguf.py --check-deps

# 转换为 F32（完整精度）
python model/convert_qwen_to_gguf.py --outtype f32

# 转换为 F16（推荐）
python model/convert_qwen_to_gguf.py --outtype f16

# 转换为 Q8_0（量化）
python model/convert_qwen_to_gguf.py --outtype q8_0

# 详细输出
python model/convert_qwen_to_gguf.py --outtype f16 --verbose
```

### 文件位置

- **转换脚本**: `model/convert_qwen_to_gguf.py`
- **使用文档**: `model/GGUF_CONVERSION_README.md`
- **设计文档**: `docs/design/GGUF格式支持详细设计.md`
- **任务分解**: `docs/design/GGUF格式支持任务分解.md`

---

## 📐 设计要点

### 1. 后端无关设计
- ✅ 使用 `ModelWeights` 通用数据结构
- ✅ `IModelLoader` 不依赖具体后端
- ✅ 通过适配器转换为后端张量

### 2. 可扩展架构
- ✅ 工厂模式支持多种格式
- ✅ 策略模式支持不同量化类型
- ✅ 适配器模式支持多后端

### 3. 性能优化
- ✅ 内存映射加载大文件
- ✅ 延迟加载权重
- ✅ 量化支持减少内存

---

## 🔍 问题排查

### 转换失败
1. 检查依赖: `python model/convert_qwen_to_gguf.py --check-deps`
2. 确认模型目录存在: `ls model/Qwen/Qwen3-0.6B/`
3. 查看详细错误: 添加 `--verbose` 参数

### 加载失败
1. 检查 GGUF 文件完整性
2. 验证文件格式版本
3. 查看日志输出

### 性能问题
1. 使用量化格式（Q8_0）
2. 启用内存映射
3. 优化后端选择

---

## 📚 相关资源

- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **GGUF 规范**: https://github.com/ggerganov/llama.cpp/blob/master/gguf.md
- **项目文档导航**: [../../00_文档导航.md](../../00_文档导航.md)

---

## 📝 更新记录

| 日期 | 更新内容 |
|------|---------|
| 2026-01-13 | 创建快速参考文档 |

---

**提示**: 需要详细信息时，请查看 [完整文档索引](./GGUF格式支持文档索引.md)
