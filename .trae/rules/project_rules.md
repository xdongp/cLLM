# cLLM Project Rules for Trae

## 核心约束

Trae 在协助 cLLM 项目开发时，必须遵守以下核心约束：

### 1. 问题分解原则
- **遇到复杂问题时，必须将问题分解，而不是实现简化版**
- 将大任务拆分为可管理的子任务
- 使用 TODO 列表跟踪进度
- 每个子任务完成后进行验证

### 2. 代码质量规范
- **禁止硬编码**：使用配置文件或常量定义
- **避免重复代码**：提取公共函数
- **命名规范**：使用有意义的变量名和函数名
- **类型安全**：确保类型正确，避免隐式转换

### 3. 项目特定约束
- **禁止删除 `.codebuddy/` 目录及其内容**
- **禁止重写整个文件** (必须使用精确替换)
- **禁止创建临时脚本文件** (如 `benchmark_*.py`, `test_*.py`)
- **禁止使用裸指针** (使用智能指针)
- **禁止使用全局变量**
- **禁止在头文件中实现大段代码** (除模板外)
- **禁止循环依赖**
- **禁止主动提交代码** (除非用户明确要求)

## 工作流程规则

### 修复-测试-验证循环
```
用户: "修复某个bug"
AI: 修复代码
用户: "请测试"
AI: 自动执行:
  - 编译项目
  - 运行测试
  - 如果失败，分析错误并修复
  - 重复直到成功
```

### 多步骤任务自动完成
```
用户: "实现功能X"
AI: 
  - 创建TODO列表
  - 逐步实现每个步骤
  - 每完成一步，自动验证
  - 继续下一步直到完成
```

### 错误自动处理
```
如果遇到编译错误:
  - 分析错误原因
  - 修复问题
  - 重新编译
  - 重复直到成功

如果遇到运行时错误:
  - 查看日志
  - 定位问题
  - 修复并测试
```

## 参考文档

Trae 必须遵守以下 .codebuddy 规则文件：

### 核心约束 (必须遵守)
- [00_core_constraints.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/always/00_core_constraints.md) - 核心约束规则
- [01_architecture_rules.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/always/01_architecture_rules.md) - 架构设计约束
- [02_workflow_standards.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/always/02_workflow_standards.md) - 工作流程标准

### 手动规则 (按需参考)
- [code_generation_standards.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/manual/code_generation_standards.md) - 代码生成标准
- [performance_optimization.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/manual/performance_optimization.md) - 性能优化规则
- [refactoring_guide.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/manual/refactoring_guide.md) - 代码重构规则

### 特定功能规则
- [tokenizer_integration.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/rules/requested/tokenizer_integration.md) - Tokenizer集成规则

### 项目文档
- [README.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/README.md) - CodeBuddy 使用说明
- [QUICK_REFERENCE.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/QUICK_REFERENCE.md) - 快速参考
- [USAGE_GUIDE.md](/Users/dannypan/PycharmProjects/cLLM/.codebuddy/USAGE_GUIDE.md) - 使用指南

## 项目特定规则

### C++ 代码规范
1. **内存管理**：使用智能指针，避免内存泄漏
2. **错误处理**：检查返回值，使用异常处理关键错误
3. **性能优化**：
   - 减少内存拷贝
   - 使用缓存友好的数据布局
   - 并行化计算密集型任务
4. **量化支持**：
   - LayerNorm 权重保持 F32
   - 线性层权重可量化到 INT8/FP16
   - 量化参数（scale, zero_point）必须正确传递

### 配置文件规范
- 使用 YAML 格式
- 必须包含 backend 类型和参数
- 量化类型：fp32, fp16, int8

### 测试规范
1. 每次修改后必须编译验证
2. 功能修改后必须测试输出正确性
3. 性能优化后必须对比基准数据

## 常用命令

### 编译
```bash
cd build && cmake --build . --target cllm_server -j8
```

### 测试服务器
```bash
# 启动服务器
./bin/cllm_server --config config_kylin_int8.yaml

# 测试生成
curl -s http://localhost:18081/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20}'
```

### 内存检查
```bash
# 检查进程内存
ps aux | grep cllm_server
```

## 用户指令映射

| 用户指令 | AI 行动 |
|---------|--------|
| "继续" | 继续执行当前任务的下一步 |
| "请测试" | 编译、测试、验证当前修改 |
| "如果失败，请修复" | 遇到错误时自动修复 |
| "完成后继续优化" | 完成当前任务后自动进入优化阶段 |

## 关键经验总结

### INT8 量化实现要点
1. 只量化线性层权重（q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj）
2. LayerNorm 权重必须保持 F32
3. 量化参数数组大小：3个全局权重 + 每层11个权重
4. 反量化公式：output = scale * (int8_value - zero_point)

### 性能优化方向
1. 内存效率：减少不必要的内存分配与复制
2. 计算优化：使用 SIMD/NEON 指令
3. 并行化：使用 OpenMP 并行化独立计算
4. 缓存优化：改善数据局部性
