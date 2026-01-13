# Phase0_执行计划：基础架构搭建

## 文档信息
- **创建日期**: 2026-01-13
- **基于文档**: `docs/design/GGUF格式支持任务分解.md`
- **版本**: v1.0
- **状态**: 执行计划

---

## 1. 阶段目标
建立后端无关的模型加载架构，实现通用权重数据结构和工厂模式接口，为后续 GGUF 格式支持奠定基础。

## 2. 目标范围
- 实现通用权重数据结构 (`ModelWeights`, `WeightData`, `LayerWeights`)
- 实现 `IModelLoader` 接口（后端无关）
- 实现 `ModelLoaderFactory` 工厂类
- 扩展 `LibTorchBackend` 添加 GGUF 支持接口
- 实现格式检测逻辑
- 编写基础单元测试

## 3. 前置条件
- ✅ 设计文档已完成并审核通过
- ✅ 开发环境已配置（CMake、编译器、依赖库）
- ✅ 现有代码库可正常编译运行

## 4. 预期成果
- 通用权重数据结构已实现并通过测试
- `IModelLoader` 接口定义完成，支持 `loadWeights()` 方法
- `ModelLoaderFactory` 可以检测格式并创建加载器
- `LibTorchBackend` 扩展接口已定义
- 基础单元测试覆盖率达到 80%+

---

## 5. 详细任务清单

### 任务1.1: 实现通用权重数据结构
- **任务ID**: T1.1
- **执行内容**:
  - 创建 `include/cllm/model/weight_data.h`
  - 实现 `WeightData` 结构体
  - 实现 `LayerWeights` 结构体
  - 实现 `ModelWeights` 结构体
  - 实现辅助方法（`findWeight()`, `getAllWeights()`, `isValid()`）
- **技术要求**:
  - 使用标准 C++17
  - 遵循项目编码规范
  - 包含完整的 Doxygen 注释
- **交付标准**:
  - 头文件通过编译
  - 所有方法有单元测试
  - 代码审查通过
- **负责人角色**: C++ 开发工程师
- **预估工时**: 2 天
- **依赖关系**: 无

### 任务1.2: 实现 IModelLoader 接口
- **任务ID**: T1.2
- **执行内容**:
  - 创建 `include/cllm/model/loader_interface.h`
  - 定义 `IModelLoader` 抽象基类
  - 实现 `loadWeights(ModelWeights&)` 纯虚方法
  - 实现 `loadInto()` 便捷方法（默认实现）
  - 实现 `loadToTorchTensorDict()` 便捷方法（条件编译）
  - 实现转换辅助方法（`convertToKylinTensors()`, `convertToTorchTensors()`）
- **技术要求**:
  - 使用虚函数实现多态
  - 支持条件编译（`#ifdef ENABLE_LIBTORCH_BACKEND`）
  - 异常安全保证
- **交付标准**:
  - 接口定义完整
  - 编译通过
  - 有接口使用示例
- **负责人角色**: C++ 架构师
- **预估工时**: 3 天
- **依赖关系**: T1.1

### 任务1.3: 实现 ModelLoaderFactory 工厂类
- **任务ID**: T1.3
- **执行内容**:
  - 创建 `include/cllm/model/loader_factory.h`
  - 创建 `src/model/loader_factory.cpp`
  - 实现 `ModelFormat` 枚举
  - 实现 `detectFormat()` 方法
  - 实现 `createLoader()` 工厂方法
  - 实现各格式的创建方法（`createBinaryLoader()`, `createGGUFLoader()`, `createSafetensorsLoader()`）
  - 实现 `isFormatSupported()` 方法
- **技术要求**:
  - 使用工厂模式
  - 支持运行时格式检测
  - 支持条件编译
- **交付标准**:
  - 可以正确检测文件格式
  - 可以创建对应的加载器实例
  - 有单元测试覆盖
- **负责人角色**: C++ 开发工程师
- **预估工时**: 2 天
- **依赖关系**: T1.2

### 任务1.4: 扩展 LibTorchBackend 接口
- **任务ID**: T1.4
- **执行内容**:
  - 修改 `include/cllm/inference/libtorch_backend.h`
  - 添加 `loadWeightsFromDict()` 方法声明
  - 添加 `loadFromGGUF()` 方法声明
  - 添加 `buildModel()` 私有方法声明
  - 更新类文档说明
- **技术要求**:
  - 保持向后兼容
  - 使用条件编译保护
- **交付标准**:
  - 接口定义完整
  - 编译通过
  - 文档更新
- **负责人角色**: C++ 开发工程师
- **预估工时**: 1 天
- **依赖关系**: T1.2

### 任务1.5: 实现格式检测逻辑
- **任务ID**: T1.5
- **执行内容**:
  - 在 `ModelLoaderFactory` 中实现文件扩展名检测
  - 实现文件魔数检测（可选，用于更准确的格式识别）
  - 处理边界情况（文件不存在、权限不足等）
- **技术要求**:
  - 支持 `.gguf`, `.bin`, `.safetensors` 扩展名
  - 错误处理完善
- **交付标准**:
  - 格式检测准确率 100%
  - 有单元测试
- **负责人角色**: C++ 开发工程师
- **预估工时**: 1 天
- **依赖关系**: T1.3

### 任务1.6: 编写基础单元测试
- **任务ID**: T1.6
- **执行内容**:
  - 创建 `tests/test_weight_data.cpp`
  - 创建 `tests/test_loader_factory.cpp`
  - 编写 `WeightData` 测试用例
  - 编写 `ModelWeights` 测试用例
  - 编写 `ModelLoaderFactory` 测试用例
- **技术要求**:
  - 使用 Google Test 框架
  - 测试覆盖率达到 80%+
- **交付标准**:
  - 所有测试用例通过
  - 测试覆盖率报告
- **负责人角色**: 测试工程师
- **预估工时**: 2 天
- **依赖关系**: T1.1, T1.2, T1.3

---

## 6. 并行执行策略

### 可并行任务
- T1.1 (通用权重数据结构) 和 T1.2 (IModelLoader接口) 可以并行开始
- T1.3 (ModelLoaderFactory) 依赖 T1.2，但可以与 T1.4 (LibTorchBackend扩展) 并行
- T1.6 (基础单元测试) 可以在 T1.1-T1.3 完成后开始

---

## 7. 关键路径分析

### 关键路径
```
T1.1 → T1.2 → T1.3 → T1.5 → T1.6
```

**关键路径总时长**: 约 2-3 周

---

## 8. 资源分配建议

### 人员配置
- C++ 开发工程师: 2-3 人
- 测试工程师: 1 人

### 技能要求
- 熟悉 C++17
- 了解设计模式
- 有模型加载经验
- 熟悉 Google Test 框架

---

## 9. 风险与应对

### 技术风险
| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 数据结构设计不合理 | 中 | 中 | 提前进行设计评审 |
| 接口扩展性不足 | 中 | 中 | 采用面向接口设计 |

### 进度风险
| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 开发周期延长 | 低 | 中 | 分阶段实施，优先核心功能 |
| 人员流动 | 低 | 高 | 知识共享，代码文档化 |

---

## 10. 交付物清单

### 代码交付物
- [ ] `include/cllm/model/weight_data.h` - 通用权重数据结构
- [ ] `include/cllm/model/loader_interface.h` - 加载器接口
- [ ] `include/cllm/model/loader_factory.h` - 工厂类
- [ ] `src/model/loader_factory.cpp` - 工厂实现

### 测试交付物
- [ ] `tests/test_weight_data.cpp` - 权重数据测试
- [ ] `tests/test_loader_factory.cpp` - 工厂测试
- [ ] 测试覆盖率报告

---

**文档结束**