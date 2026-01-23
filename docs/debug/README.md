# Kylin 后端调试文档索引

## 📚 文档列表

### 1. [kylin_debug_plan.md](./kylin_debug_plan.md)
**Kylin 后端分阶段调试规划**

- 完整的 9 个测试阶段定义
- 每个阶段的目标、测试内容和验证方法
- 实施计划和优先级
- 成功标准

**适合**: 了解整体调试策略和计划

### 2. [kylin_flow_diagram.md](./kylin_flow_diagram.md)
**Kylin 完整流程调用图**

- 初始化流程的详细调用链
- 推理流程的完整数据流
- 张量形状变化图
- 关键检查点定义

**适合**: 理解 Kylin 的完整执行流程

### 3. [kylin_stage_implementation_guide.md](./kylin_stage_implementation_guide.md)
**分阶段测试实施指南**

- 每个阶段的具体测试用例（C++ 代码）
- 详细的验证方法
- 实施步骤和代码示例
- 预期结果定义

**适合**: 实施具体测试用例时参考

---

## 🛠️ 工具列表

### 1. `tools/kylin_stage_test.cpp`
**分阶段测试框架（C++）**

- 按阶段执行测试
- 自动生成测试报告
- 支持与 llama_cpp 对比

**状态**: ⏳ 待完善（框架已创建，需要添加更多测试用例）

### 2. `tools/run_kylin_stages.sh`
**自动化测试脚本**

- 自动运行所有阶段
- 生成阶段报告
- 失败时停止并报告

**状态**: ✅ 已创建

### 3. `tools/full_comparison.sh`
**完整对比测试**

- 自动测试 Kylin 和 llama_cpp
- 生成详细对比报告
- 提取调试信息

**状态**: ✅ 已完成

### 4. `tools/analyze_debug_logs.py`
**日志分析工具**

- 解析 Kylin 调试日志
- 生成统计报告
- 分析数值分布

**状态**: ✅ 已完成

---

## 📊 测试阶段概览

| Stage | 名称 | 状态 | 优先级 |
|-------|------|------|--------|
| **Stage 0** | 基础环境验证 | ✅ 完成 | - |
| **Stage 1** | 模型加载验证 | ✅ 完成 | - |
| **Stage 2** | Token Embedding 验证 | 🔄 进行中 | 高 |
| **Stage 3** | 第一层 Transformer Block | 🔄 进行中 | 高 |
| **Stage 4** | 注意力计算详细验证 | ⏳ 待开始 | 高 |
| **Stage 5** | FFN 计算验证 | ⏳ 待开始 | 中 |
| **Stage 6** | 多层累积验证 | ⏳ 待开始 | 中 |
| **Stage 7** | 最终输出验证 | 🔄 部分完成 | 中 |
| **Stage 8** | 增量推理验证 | ⏳ 待开始 | 中 |
| **Stage 9** | 端到端对比 | 🔄 进行中 | 高 |

---

## 🚀 快速开始

### 1. 查看调试规划
```bash
cat docs/debug/kylin_debug_plan.md
```

### 2. 查看流程图
```bash
cat docs/debug/kylin_flow_diagram.md
```

### 3. 运行完整对比测试
```bash
./tools/full_comparison.sh "Hi" 5 0.0
```

### 4. 分析调试日志
```bash
python3 tools/analyze_debug_logs.py /tmp/kylin_full_comparison.log
```

---

## 📝 当前进展

### ✅ 已完成

1. **调试规划文档**
   - 完整的 9 个阶段定义
   - 详细的流程调用图
   - 实施指南

2. **基础调试工具**
   - Embedding 和 Layer 0 的统计日志
   - 对比工具
   - 日志分析工具

3. **测试框架**
   - 分阶段测试框架（C++）
   - 自动化测试脚本

### 🔄 进行中

1. **Stage 2-3 的详细验证**
   - 已添加基础日志
   - 需要完善对比功能

2. **Stage 9 的端到端对比**
   - 已有基础对比工具
   - 需要添加更多统计信息

### ⏳ 待开始

1. **Stage 4-6 的详细验证**
   - 需要添加注意力计算的详细日志
   - 需要添加 FFN 的详细日志
   - 需要添加多层输出的验证

2. **Stage 8 的增量推理验证**
   - 需要添加 KV Cache 状态验证
   - 需要添加增量推理的测试用例

---

## 🎯 下一步行动

### 立即行动（今天）

1. **完善 Stage 2-3 的验证**
   - 添加 embedding 的详细对比
   - 添加 Layer 0 输出的详细分析
   - 实现与 llama_cpp 的自动对比

2. **实现 Stage 4.1-4.3 的验证**
   - 添加 QKV 投影的日志
   - 添加 Q/K 归一化的日志
   - 添加 RoPE 的验证

### 短期目标（1-2天）

1. **完成 Stage 4 的所有子阶段**
   - 实现注意力计算的完整验证
   - 添加所有中间节点的统计

2. **实现 Stage 5-6 的验证**
   - 添加 FFN 的详细日志
   - 实现多层输出的验证

### 中期目标（3-5天）

1. **完成所有阶段的验证**
2. **定位并修复所有问题**
3. **实现性能基准测试**

---

## 📖 使用指南

### 对于开发者

1. **开始调试前**：阅读 `kylin_debug_plan.md` 了解整体策略
2. **实施测试时**：参考 `kylin_stage_implementation_guide.md` 的具体用例
3. **理解流程时**：查看 `kylin_flow_diagram.md` 的调用图
4. **运行测试时**：使用 `tools/run_kylin_stages.sh` 自动化测试

### 对于测试人员

1. **运行完整测试**：使用 `tools/full_comparison.sh`
2. **分析结果**：使用 `tools/analyze_debug_logs.py`
3. **查看报告**：检查 `/tmp/comparison_report.txt`

---

## 🔍 调试技巧

### 1. 使用 CPU 后端进行调试
Metal 后端可能有段错误，建议使用 CPU 后端：
```yaml
backend:
  kylin:
    device_backend: "cpu"
```

### 2. 使用 temperature=0.0
确保确定性输出，便于对比：
```json
{"prompt": "Hi", "max_tokens": 5, "temperature": 0.0}
```

### 3. 查看详细日志
启用 DEBUG 级别日志：
```bash
./build/bin/cllm_server --config config/config.yaml 2>&1 | grep "\[Kylin Debug\]"
```

### 4. 逐步启用功能
如果遇到问题，可以：
1. 先禁用可能有问题的功能（如 Q/K norm）
2. 验证基础功能正确
3. 逐步启用功能，每次验证

---

## 📞 问题反馈

如果发现问题或需要帮助：
1. 查看相关阶段的文档
2. 检查日志文件
3. 运行对比测试
4. 查看测试报告

---

**最后更新**: 2026-01-23  
**文档版本**: v1.0
