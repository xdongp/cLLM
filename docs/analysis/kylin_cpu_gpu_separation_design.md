# src/kylin CPU/GPU 代码分离设计方案

## 自动执行规则

> **重要提示**：本文档包含完整的 CPU/GPU 分离实施计划。执行时应遵循以下自动执行规则：

### 自动执行原则

1. **无需询问**：执行过程中不需要询问用户建议，全程自动决策
2. **逐步执行**：严格按照文档中的步骤顺序执行（步骤 1.1 → 1.2 → ... → 5.2）
3. **测试驱动**：每步完成后必须执行测试命令验证
4. **自动提交**：测试通过后自动执行 `git add -A && git commit -m "Step X.Y: 描述 - CPU/GPU tests passed"`
5. **失败回滚**：如果测试失败，自动执行 `git reset --hard HEAD~1` 回滚，然后尝试其他实现方法
6. **持续执行**：不回滚到起点，只在当前步骤失败时回滚到上一步，尝试其他方法后继续

### 成功标准

- **编译成功**：`cmake .. && make` 无错误
- **CPU 测试通过**：`./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常中文文本
- **GPU 测试通过**：`CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常中文文本
- **Stage 27 通过**：`./bin/kylin_test_suite --gtest_filter="*Stage27*"` 测试通过

### 失败处理流程

```
执行步骤 X.Y
    ↓
测试是否通过？
    ↓ 是
自动提交 git commit
    ↓
进入下一步 X.Y+1
    ↓
测试是否通过？
    ↓ 否
自动回滚 git reset --hard HEAD~1
    ↓
尝试其他实现方法
    ↓
重新测试
```

### 注意事项

- 不要修改文档中定义的接口设计
- 保持向后兼容，不破坏现有公共 API
- 每步修改尽量小，便于回滚
- 保留所有原始功能，只重构代码组织方式

---


## 一、当前代码结构分析

### 1.1 目录结构

```
src/kylin/
├── core/              # 核心基础设施
│   ├── kernels.cpp          # CPU 基础算子 (matmul, rmsnorm, softmax)
│   ├── ggml_kernels.cpp     # GGML 内核封装
│   └── quantization.cpp     # 量化支持
├── ops/               # 算子实现
│   ├── attention.cpp        # CPU Attention 实现
│   ├── feed_forward.cpp     # CPU FFN 实现
│   ├── rope.cpp             # CPU RoPE 实现
│   └── kv_cache_ops.cpp     # KV Cache 操作
├── hf/                # HuggingFace 模型支持
│   ├── transformer.cpp      # HFTransformerModel - CPU/GPU 混合实现 ❌
│   ├── ggml_backend.cpp     # GGML GPU 后端实现
│   └── kv_cache_pool.cpp    # KV Cache 池
├── model/             # 模型组件
│   ├── transformer_model.cpp
│   └── transformer_block.cpp
└── gguf/              # GGUF 格式支持
    ├── transformer.cpp
    └── ggml_operator.cpp
```

### 1.2 核心问题：CPU/GPU 代码混合

**文件：src/kylin/hf/transformer.cpp**

```cpp
class HFTransformerModel {
    // 构造函数中根据 device 分支
    if (device == DeviceType::Metal) {
#ifdef GGML_USE_METAL
        gpuBackend_ = std::make_unique<GGMLGPUBackend>();
#endif
    }
    
    // 权重预转换逻辑混杂
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::FP16) {
            convertWeightsToFP16();
            if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
                preconvertWeights();  // GPU需要FP32
            }
        }
    }
};

// forwardBatch 中再次分支
std::vector<std::vector<float>> HFTransformerModel::forwardBatch(...) {
    if (deviceType_ == DeviceType::Metal && gpuBackend_) {
        return forwardBatchGPU(tokenIds, positions, requestIds);
    }
    // CPU 实现...
}
```

**问题：**
1. 一个类同时处理 CPU 和 GPU 逻辑
2. 多处 `if (deviceType_ == DeviceType::Metal)` 分支判断
3. 代码耦合度高，难以维护和测试
4. GPU 实现分散在 `hf/ggml_backend.cpp`

### 1.3 重复实现分析

| 组件 | CPU 实现 | GPU 实现 | 问题 |
|------|----------|----------|------|
| Attention | `ops/attention.cpp` | `hf/ggml_backend.cpp` | 重复实现 |
| RoPE | `ops/rope.cpp` | `hf/ggml_backend.cpp` | 重复实现 |
| FFN | `ops/feed_forward.cpp` | `hf/ggml_backend.cpp` | 重复实现 |
| RMS Norm | `core/kernels.cpp` | `hf/ggml_backend.cpp` | 重复实现 |

## 二、设计目标

1. **完全分离 CPU 和 GPU 代码**
2. **统一后端接口**
3. **模型与后端解耦**
4. **便于测试和维护**

## 三、设计方案

### 3.1 核心设计原则

1. **策略模式**：统一接口，不同后端实现
2. **组合优于继承**：模型由后端组合而成
3. **单一职责**：每个类只做一件事

### 3.2 目标架构

保持现有目录结构，通过接口抽象实现分离：

```
src/kylin/
├── backend/           # 后端接口层（新增）
│   ├── backend_interface.h    # 统一后端接口
│   ├── cpu_backend.h          # CPU 后端
│   ├── cpu_backend.cpp        # CPU 后端实现
│   ├── gpu_backend.h          # GPU 后端
│   └── gpu_backend.cpp        # GPU 后端实现（封装 ggml_backend）
├── core/              # 核心基础设施（不变）
│   ├── kernels.cpp            # CPU 基础算子
│   └── ggml_kernels.cpp       # GGML 封装
├── ops/               # 算子层（不变）
│   ├── attention.cpp          # CPU Attention
│   ├── feed_forward.cpp       # CPU FFN
│   └── rope.cpp               # CPU RoPE
├── hf/                # HF 模型（重构）
│   ├── transformer.h          # 模型定义（与后端无关）
│   ├── transformer.cpp        # 模型实现（使用后端接口）
│   ├── ggml_backend.h         # GGML GPU 实现（保留）
│   └── ggml_backend.cpp       # GGML GPU 实现（保留）
└── model/             # 模型组件（不变）
```

### 3.3 关键接口设计

#### 3.3.1 后端接口

```cpp
// backend/backend_interface.h

class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;
    
    // 初始化
    virtual bool initialize(const HFModelConfig& config) = 0;
    virtual void shutdown() = 0;
    
    // 权重管理
    virtual bool loadWeights(const ModelWeights& weights) = 0;
    
    // 前向推理
    virtual std::vector<float> forward(
        const std::vector<int32_t>& inputIds,
        int requestId
    ) = 0;
    
    virtual std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) = 0;
    
    // KV Cache 管理
    virtual void resetKVCache(int requestId) = 0;
    virtual void releaseKVCache(int requestId) = 0;
    
    // 信息查询
    virtual std::string getName() const = 0;
    virtual bool isGPU() const = 0;
};

// 后端工厂
class BackendFactory {
public:
    static std::unique_ptr<IComputeBackend> create(DeviceType device);
};
```

#### 3.3.2 CPU 后端实现

```cpp
// backend/cpu_backend.h

class CPUBackend : public IComputeBackend {
public:
    bool initialize(const HFModelConfig& config) override;
    void shutdown() override;
    
    bool loadWeights(const ModelWeights& weights) override;
    
    std::vector<float> forward(
        const std::vector<int32_t>& inputIds,
        int requestId
    ) override;
    
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) override;
    
    void resetKVCache(int requestId) override;
    void releaseKVCache(int requestId) override;
    
    std::string getName() const override { return "CPU"; }
    bool isGPU() const override { return false; }

private:
    HFModelConfig config_;
    ModelWeights weights_;
    
    // CPU 特有的成员
    std::vector<float> kCache_;
    std::vector<float> vCache_;
    std::vector<float> ropeFreqsCos_;
    std::vector<float> ropeFreqsSin_;
    
    // 工作缓冲区
    std::vector<float> hiddenStates_;
    std::vector<float> qkvBuffer_;
    // ... 其他缓冲区
    
    // 核心计算函数
    void rmsNorm(const float* input, float* output, const float* weight, int size);
    void matmul(const float* A, const float* B, float* C, int M, int N, int K);
    void applyRoPE(float* q, float* k, int pos, int headDim);
    void attention(...);
    void ffn(...);
};
```

#### 3.3.3 GPU 后端实现

```cpp
// backend/gpu_backend.h

class GPUBackend : public IComputeBackend {
public:
    GPUBackend();
    ~GPUBackend();
    
    bool initialize(const HFModelConfig& config) override;
    void shutdown() override;
    
    bool loadWeights(const ModelWeights& weights) override;
    
    std::vector<float> forward(
        const std::vector<int32_t>& inputIds,
        int requestId
    ) override;
    
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) override;
    
    void resetKVCache(int requestId) override;
    void releaseKVCache(int requestId) override;
    
    std::string getName() const override { return "GPU"; }
    bool isGPU() const override { return true; }

private:
    // 封装现有的 GGMLGPUBackend
    std::unique_ptr<GGMLGPUBackend> ggmlBackend_;
};
```

#### 3.3.4 模型类（与后端解耦）

```cpp
// hf/transformer.h

class HFTransformerModel {
public:
    HFTransformerModel(const std::string& modelDir, 
                       DeviceType device = DeviceType::CPU,
                       QuantType quantType = QuantType::FP32);
    ~HFTransformerModel();
    
    bool isLoaded() const { return loaded_; }
    
    // 前向推理 - 委托给后端
    std::vector<float> forward(const std::vector<int32_t>& inputIds);
    std::vector<float> forwardWithRequestId(const std::vector<int32_t>& inputIds, size_t requestId);
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<size_t>& requestIds
    );
    
    // 释放 KV Cache
    void releaseKVCache(size_t requestId);
    int getKVCacheCurrentLength(size_t requestId) const;
    
    // 获取配置
    const HFModelConfig& config() const { return config_; }

private:
    HFModelConfig config_;
    std::unique_ptr<IComputeBackend> backend_;  // 后端接口
    bool loaded_ = false;
    
    // 加载权重
    ModelWeights loadWeights(const std::string& modelDir);
};
```

## 四、收益分析

### 4.1 代码质量

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| transformer.cpp 行数 | ~1000+ | ~200 | -80% |
| CPU/GPU 分支数 | 10+ | 0 | -100% |
| 类职责 | 混合 | 单一 | ✓ |

### 4.2 维护性

1. **新增后端**：只需实现 `IComputeBackend` 接口
2. **Bug 修复**：定位更快，影响范围明确
3. **单元测试**：可以单独测试每个后端

### 4.3 性能

- 无性能损失，只是代码组织方式改变
- CPU 路径：无 GPU 相关代码，缓存更友好
- GPU 路径：无 CPU 回退逻辑，减少分支预测失败

## 五、风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 引入 Bug | 中 | 高 | 充分测试，保留回滚方案 |
| 性能回退 | 低 | 中 | 基准测试对比 |
| 工期延长 | 低 | 低 | 分步骤实施，每步可验证 |

## 六、逐步实施计划（含详细测试步骤和 Git 提交）

### 测试命令

**每次修改后必须执行：**

```bash
# 1. 编译
mkdir -p build && cd build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)

# 2. CPU 功能测试
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40

# 3. GPU 功能测试（Metal）
CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40

# 4. kylin_test_suite Stage 27 测试
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

### Git 提交规范

**每步骤完成后必须提交：**

```bash
# 测试通过后提交
git add -A
git commit -m "Step X: 描述 - CPU/GPU 测试通过"

# 如果测试失败，回滚到上一步
git reset --hard HEAD~1
```

---

### 步骤 1.1：创建后端接口头文件

**目标**：创建 `backend_interface.h`，定义统一接口

**操作**：
1. 创建 `include/cllm/kylin/backend/backend_interface.h`
   - 定义 `IComputeBackend` 纯虚接口
   - 定义 `BackendFactory` 工厂类
   - 定义 `ModelWeights` 数据结构

2. 更新 `CMakeLists.txt`（只添加头文件路径，不添加源文件）

**验证标准**：
- [ ] `cmake ..` 成功
- [ ] `make` 成功（不影响现有代码）
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 1.1: Create backend interface header - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 1.2：创建 CPU 后端头文件（空实现）

**目标**：创建 `cpu_backend.h`，声明 CPUBackend 类

**操作**：
1. 创建 `include/cllm/kylin/backend/cpu_backend.h`
   - 声明 `CPUBackend` 类，继承 `IComputeBackend`
   - 所有方法暂时返回 false/空值

2. 更新 `CMakeLists.txt`

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 1.2: Create CPU backend header (empty impl) - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 1.3：创建 GPU 后端头文件（空实现）

**目标**：创建 `gpu_backend.h`，声明 GPUBackend 类

**操作**：
1. 创建 `include/cllm/kylin/backend/gpu_backend.h`
   - 声明 `GPUBackend` 类，继承 `IComputeBackend`
   - 所有方法暂时返回 false/空值

2. 更新 `CMakeLists.txt`

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 1.3: Create GPU backend header (empty impl) - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 1.4：创建后端工厂

**目标**：创建 `backend_factory.cpp`，实现工厂模式

**操作**：
1. 创建 `src/kylin/backend/backend_factory.cpp`
   - 实现 `BackendFactory::create()`
   - 暂时返回 nullptr

2. 更新 `CMakeLists.txt`，添加源文件

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 1.4: Create backend factory - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.1：分析 CPU 逻辑

**目标**：分析 `transformer.cpp`，识别 CPU 特有逻辑

**操作**：
1. 阅读 `src/kylin/hf/transformer.cpp`
2. 识别 CPU 特有的成员变量
3. 识别 CPU 计算函数
4. 记录需要移到 CPUBackend 的代码

**输出**：创建文档 `docs/analysis/cpu_logic_extraction.md`

**验证标准**：
- [ ] 分析文档完成
- [ ] 未修改任何代码

**提交**：
```bash
git add docs/analysis/cpu_logic_extraction.md
git commit -m "Step 2.1: Analyze CPU logic in transformer.cpp - analysis complete"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.2：移植 RMS Norm 到 CPU 后端

**目标**：将 RMS Norm 实现移到 CPUBackend

**操作**：
1. 在 `cpu_backend.h` 中添加 `rmsNorm()` 声明
2. 在 `cpu_backend.cpp` 中实现 `rmsNorm()`
   - 复制 `core/kernels.cpp` 中的实现
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.2: Move RMS Norm to CPUBackend - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.3：移植 Matmul 到 CPU 后端

**目标**：将 Matmul 实现移到 CPUBackend

**操作**：
1. 在 `cpu_backend.h` 中添加 `matmul()` 声明
2. 在 `cpu_backend.cpp` 中实现 `matmul()`
   - 复制 `core/kernels.cpp` 中的实现
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.3: Move Matmul to CPUBackend - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.4：移植 RoPE 到 CPU 后端

**目标**：将 RoPE 实现移到 CPUBackend

**操作**：
1. 在 `cpu_backend.h` 中添加 `applyRoPE()` 声明
2. 在 `cpu_backend.cpp` 中实现 `applyRoPE()`
   - 复制 `ops/rope.cpp` 中的实现
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.4: Move RoPE to CPUBackend - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.5：移植 Attention 到 CPU 后端

**目标**：将 Attention 实现移到 CPUBackend

**操作**：
1. 在 `cpu_backend.h` 中添加 `attention()` 声明
2. 在 `cpu_backend.cpp` 中实现 `attention()`
   - 复制 `ops/attention.cpp` 中的实现
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.5: Move Attention to CPUBackend - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.6：移植 FFN 到 CPU 后端

**目标**：将 FFN 实现移到 CPUBackend

**操作**：
1. 在 `cpu_backend.h` 中添加 `ffn()` 声明
2. 在 `cpu_backend.cpp` 中实现 `ffn()`
   - 复制 `ops/feed_forward.cpp` 中的实现
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.6: Move FFN to CPUBackend - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.7：实现 CPUBackend 初始化

**目标**：实现 CPUBackend 的 `initialize()` 方法

**操作**：
1. 实现 `initialize()` - 初始化 CPU 缓冲区
2. 实现 `shutdown()` - 释放资源
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.7: Implement CPUBackend initialize/shutdown - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.8：实现 CPUBackend 权重加载

**目标**：实现 CPUBackend 的 `loadWeights()` 方法

**操作**：
1. 实现 `loadWeights()` - 加载权重到 CPU
2. 定义 `ModelWeights` 数据结构
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.8: Implement CPUBackend loadWeights - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.9：实现 CPUBackend KV Cache 管理

**目标**：实现 CPUBackend 的 KV Cache 管理方法

**操作**：
1. 实现 `resetKVCache()` - 重置 KV Cache
2. 实现 `releaseKVCache()` - 释放 KV Cache
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 2.9: Implement CPUBackend KV Cache management - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.10：实现 CPUBackend forward

**目标**：实现 CPUBackend 的 `forward()` 方法

**操作**：
1. 实现 `forward()` - 单序列推理
2. 组合 RMS Norm, Attention, FFN 等组件
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常文本（非乱码）
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` 通过

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 2.10: Implement CPUBackend forward - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 2.11：实现 CPUBackend forwardBatch

**目标**：实现 CPUBackend 的 `forwardBatch()` 方法

**操作**：
1. 实现 `forwardBatch()` - 批量推理
2. 循环调用 `forward()` 或实现真正的批量处理
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` 通过

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 2.11: Implement CPUBackend forwardBatch - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.1：分析 GPU 逻辑

**目标**：分析 `ggml_backend.cpp`，识别 GPU 特有逻辑

**操作**：
1. 阅读 `src/kylin/hf/ggml_backend.cpp`
2. 识别 `GGMLGPUBackend` 的接口
3. 记录需要封装的方法

**输出**：创建文档 `docs/analysis/gpu_logic_extraction.md`

**验证标准**：
- [ ] 分析文档完成
- [ ] 未修改任何代码

**提交**：
```bash
git add docs/analysis/gpu_logic_extraction.md
git commit -m "Step 3.1: Analyze GPU logic in ggml_backend.cpp - analysis complete"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.2：实现 GPUBackend 初始化

**目标**：实现 GPUBackend 的 `initialize()` 方法

**操作**：
1. 在 `gpu_backend.h` 中添加 `GGMLGPUBackend` 成员
2. 实现 `initialize()` - 创建 `GGMLGPUBackend`
3. 实现 `shutdown()` - 释放资源
4. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 3.2: Implement GPUBackend initialize/shutdown - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.3：实现 GPUBackend 权重加载

**目标**：实现 GPUBackend 的 `loadWeights()` 方法

**操作**：
1. 实现 `loadWeights()` - 调用 `GGMLGPUBackend::uploadWeights()`
2. 转换参数格式
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 3.3: Implement GPUBackend loadWeights - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.4：实现 GPUBackend KV Cache 管理

**目标**：实现 GPUBackend 的 KV Cache 管理方法

**操作**：
1. 实现 `resetKVCache()` - 调用 `GGMLGPUBackend` 方法
2. 实现 `releaseKVCache()` - 调用 `GGMLGPUBackend` 方法
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 3.4: Implement GPUBackend KV Cache management - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.5：实现 GPUBackend forward

**目标**：实现 GPUBackend 的 `forward()` 方法

**操作**：
1. 实现 `forward()` - 调用 `GGMLGPUBackend::forward()`
2. 转换参数格式
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常文本

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 3.5: Implement GPUBackend forward - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 3.6：实现 GPUBackend forwardBatch

**目标**：实现 GPUBackend 的 `forwardBatch()` 方法

**操作**：
1. 实现 `forwardBatch()` - 调用 `GGMLGPUBackend::forwardBatchGPU()`
2. 转换参数格式
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常文本
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` GPU 测试通过

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 3.6: Implement GPUBackend forwardBatch - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 4.1：修改 transformer.h

**目标**：修改模型类头文件，使用后端接口

**操作**：
1. 修改 `include/cllm/kylin/hf/transformer.h`
   - 移除 CPU/GPU 特有的成员变量
   - 添加 `std::unique_ptr<IComputeBackend> backend_`
   - 保持公共接口不变

2. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 4.1: Modify transformer.h to use backend interface - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 4.2：修改 transformer.cpp 构造函数

**目标**：修改构造函数，使用 BackendFactory

**操作**：
1. 修改 `src/kylin/hf/transformer.cpp`
   - 构造函数使用 `BackendFactory::create(device)`
   - 删除 `if (deviceType_ == DeviceType::Metal)` 分支
   - 初始化后端

2. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 4.2: Modify transformer.cpp constructor - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 4.3：修改 transformer.cpp forward 方法

**目标**：修改 forward 方法，委托给后端

**操作**：
1. 修改 `forward()` - 调用 `backend_->forward()`
2. 删除 CPU 计算逻辑
3. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 输出正常文本

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
```

**提交**：
```bash
git add -A
git commit -m "Step 4.3: Modify transformer.cpp forward - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 4.4：修改 transformer.cpp forwardBatch 方法

**目标**：修改 forwardBatch 方法，委托给后端

**操作**：
1. 修改 `forwardBatch()` - 调用 `backend_->forwardBatch()`
2. 删除 `forwardBatchGPU()` 调用
3. 删除 CPU 批量计算逻辑
4. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` 通过

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 4.4: Modify transformer.cpp forwardBatch - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 4.5：清理 transformer.cpp 冗余代码

**目标**：删除已移到后端的冗余代码

**操作**：
1. 删除 CPU 计算函数（rmsNorm, matmul, attention, ffn 等）
2. 删除 GPU 相关代码
3. 删除未使用的成员变量
4. 编译测试

**验证标准**：
- [ ] 编译通过
- [ ] `./bin/show_model_output` 正常工作
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` 通过

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
rm -rf * && cmake .. && make -j$(sysctl -n hw.ncpu)
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 4.5: Clean up transformer.cpp redundant code - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 5.1：完整功能测试

**目标**：验证 CPU 和 GPU 功能完整

**操作**：
1. 运行完整测试套件
2. 对比 CPU 和 GPU 输出质量

**验证标准**：
- [ ] CPU 测试：`./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 正常
- [ ] GPU 测试：`CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40` 正常
- [ ] `./bin/kylin_test_suite --gtest_filter="*Stage27*"` 通过
- [ ] CPU 和 GPU 输出质量一致

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build

# CPU 测试
./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40

# GPU 测试
CLLM_DEVICE=metal ./bin/show_model_output ../models/qwen3-8b-fp32 "你好" 2>&1 | tail -40

# 完整测试
./bin/kylin_test_suite --gtest_filter="*Stage27*"
```

**提交**：
```bash
git add -A
git commit -m "Step 5.1: Complete functional testing - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

### 步骤 5.2：性能基准测试

**目标**：验证性能不低于重构前

**操作**：
1. 运行性能测试
2. 对比重构前后的性能数据

**验证标准**：
- [ ] CPU 性能不低于重构前
- [ ] GPU 性能不低于重构前

**测试命令**：
```bash
cd /Users/dannypan/PycharmProjects/cLLM/build
# 运行性能测试（如果有）
./bin/kylin_test_suite --gtest_filter="*Performance*"
```

**提交**：
```bash
git add -A
git commit -m "Step 5.2: Performance benchmark - CPU/GPU tests passed"
```

**回滚方案**：
```bash
git reset --hard HEAD~1
```

---

## 七、总结

通过引入 `IComputeBackend` 接口，将 CPU 和 GPU 实现完全分离，模型类只负责协调，后端负责具体计算。这样可以大幅提升代码质量和可维护性。

### 关键成功因素

1. **逐步实施**：每步都有明确的验证标准
2. **充分测试**：每次修改后都运行 show_model_output 和 kylin_test_suite
3. **Git 提交**：每步成功后立即提交，失败则回滚
4. **向后兼容**：公共接口保持不变

### 预期收益

- transformer.cpp 行数减少 80%
- CPU/GPU 分支数为 0
- 单一职责，易于测试和维护

### Git 提交历史示例

```
Step 1.1: Create backend interface header - CPU/GPU tests passed
Step 1.2: Create CPU backend header (empty impl) - CPU/GPU tests passed
Step 1.3: Create GPU backend header (empty impl) - CPU/GPU tests passed
Step 1.4: Create backend factory - CPU/GPU tests passed
Step 2.1: Analyze CPU logic in transformer.cpp - analysis complete
Step 2.2: Move RMS Norm to CPUBackend - CPU/GPU tests passed
...
Step 5.2: Performance benchmark - CPU/GPU tests passed
```