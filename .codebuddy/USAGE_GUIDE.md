# CodeBuddy 使用指南 (cLLM项目)

## 🎯 快速开始

### 1. 检查配置

```bash
# 确认.codebuddy目录存在
ls -la .codebuddy/

# 应该看到以下结构:
# .codebuddy/
# ├── README.md
# ├── project.yaml
# ├── rules/
# │   ├── always/
# │   ├── manual/
# │   └── requested/
# └── .gitignore
```

### 2. 开始使用

直接与 CodeBuddy AI 对话,AI 会自动遵守项目规则:

```
你: "请帮我实现一个新的Tokenizer"
AI: [自动读取 tokenizer_integration.md 规则]
    [遵循编码规范]
    [使用标准工作流]
```

## 📋 常见使用场景

### 场景1: 添加新功能

```
你: "给TokenizerManager添加批量编码功能"

AI会:
1. ✅ 阅读相关代码 (read_file)
2. ✅ 搜索依赖 (search_content)
3. ✅ 创建TODO (如果复杂)
4. ✅ 精确修改 (replace_in_file)
5. ✅ 验证语法 (read_lints)
6. ✅ 提示运行测试
```

### 场景2: 修复Bug

```
你: "TokenizerManager::encode在处理空字符串时崩溃"

AI会:
1. ✅ 读取相关代码
2. ✅ 定位问题
3. ✅ 精确修复
4. ✅ 添加边界检查
5. ✅ 建议添加测试用例
```

### 场景3: 性能优化

```
你: "优化Tokenizer的encode性能"

AI会:
1. ✅ 自动加载 performance_optimization.md 规则
2. ✅ 分析性能瓶颈
3. ✅ 应用优化策略 (缓存、并行等)
4. ✅ 添加性能监控
5. ✅ 建议进行性能测试
```

### 场景4: 代码重构

```
你: "重构TokenizerManager,将生成逻辑提取到单独的类"

AI会:
1. ✅ 自动加载 refactoring_guide.md 规则
2. ✅ 制定重构计划 (TODO)
3. ✅ 分步骤执行
4. ✅ 每步后验证测试
5. ✅ 保持接口兼容
```

## 🚫 AI 不会做的事 (受规则约束)

### 文件操作

- ❌ 不会重写整个文件 (使用 write_to_file 覆盖现有文件)
- ❌ 不会删除 .codebuddy 目录
- ❌ 不会创建临时脚本 (如 test_*.py, benchmark_*.py)
- ❌ 不会生成超过 800 行的单个文件

### Git 操作

- ❌ 不会修改 git config
- ❌ 不会 force push
- ❌ 不会 hard reset
- ❌ 不会主动 commit (除非你明确要求)

### 代码质量

- ❌ 不会使用裸指针 (使用智能指针)
- ❌ 不会使用全局变量
- ❌ 不会破坏模块依赖规则
- ❌ 不会添加 emoji (除非你要求,或在日志中)

## ✅ AI 会自动做的事

### 工作流程

1. **并行工具调用**
   - 同时读取多个文件
   - 同时搜索多个模式
   - 提高效率

2. **精确修改**
   - 使用 replace_in_file
   - 保留原始格式
   - 避免重写整个文件

3. **验证检查**
   - 修改后运行 read_lints
   - 检查编译错误
   - 提示运行测试

4. **任务管理**
   - 复杂任务创建 TODO
   - 实时更新进度
   - 完成后标记

### 代码规范

1. **命名约定**
   - 类名: `PascalCase`
   - 函数名: `camelCase`
   - 成员变量: `camelCase_` (后缀下划线)

2. **文件组织**
   - 头文件: `include/cllm/模块名/`
   - 实现文件: `src/模块名/`
   - 测试文件: `tests/test_模块名.cpp`

3. **依赖管理**
   - 遵循模块依赖规则
   - 添加必要的 #include
   - 使用条件编译保护

## 🎯 最佳实践建议

### 1. 明确你的需求

```
❌ 不好: "优化代码"
✅ 好的: "优化TokenizerManager::encode的性能,目标提升3倍"

❌ 不好: "修改Tokenizer"
✅ 好的: "给HFTokenizer添加批量编码功能,支持并行处理"
```

### 2. 提供上下文

```
✅ "Qwen3模型加载失败,日志显示tokenizer.json找不到,请修复"
✅ "参考HFTokenizer的实现,给NativeTokenizer添加类似的缓存功能"
✅ "按照docs/analysis/README_TOKENIZER_MIGRATION.md的方案执行"
```

### 3. 分步骤进行

```
✅ "第一步: 实现基础接口"
✅ "第二步: 添加条件编译"
✅ "第三步: 更新CMakeLists.txt"
```

### 4. 要求验证

```
✅ "实现后请运行read_lints检查语法"
✅ "修改后请确保测试通过"
✅ "添加性能监控代码"
```

## 🔍 查看 AI 的决策过程

AI 会在响应中解释它的行为:

```
AI: "我将按以下步骤执行:
     1. 读取 hf_tokenizer.h 和 hf_tokenizer.cpp
     2. 搜索相关的调用点
     3. 创建 TODO 跟踪任务
     4. 实现新功能
     5. 验证语法错误"
```

如果 AI 的决策不符合预期,你可以:
- 提供更多上下文
- 明确你的要求
- 指出AI的错误

## 📊 规则优先级

当规则冲突时,优先级:

1. **用户明确要求** (最高)
2. **核心约束规则** (00_core_constraints.md)
3. **架构规则** (01_architecture_rules.md)
4. **工作流规则** (02_workflow_standards.md)
5. **专项规则** (manual/requested/)

## 🐛 常见问题

### Q1: AI 重复犯同样的错误

**解决**:
1. 检查规则是否清晰
2. 明确告诉 AI "不要..."
3. 更新规则文件
4. 使用更强的约束词 (❌ 禁止、必须、不要)

### Q2: AI 不遵守规则

**解决**:
1. 确认规则文件在 `.codebuddy/rules/always/` 中
2. 检查规则格式是否正确
3. 重启对话 (让AI重新加载规则)
4. 在对话中明确提醒 AI

### Q3: AI 执行效率低

**解决**:
1. 明确要求 "使用并行工具调用"
2. 提供具体的文件路径
3. 限制搜索范围
4. 分解大任务为小任务

### Q4: AI 不理解项目结构

**解决**:
1. 更新 `project.yaml` 配置
2. 在规则中添加更多项目上下文
3. 引用具体的设计文档
4. 提供代码示例

## 📝 反馈和改进

如果发现:
- AI 重复犯错
- 规则不清晰
- 缺少某个场景的规则
- 规则之间有冲突

请:
1. 记录问题
2. 更新规则文件
3. 测试验证
4. 提交到版本控制

## 📚 学习资源

- **核心约束**: `.codebuddy/rules/always/00_core_constraints.md`
- **架构规则**: `.codebuddy/rules/always/01_architecture_rules.md`
- **工作流程**: `.codebuddy/rules/always/02_workflow_standards.md`
- **项目文档**: `docs/` 目录

## 🎓 进阶技巧

### 1. 引用规则

```
你: "按照 performance_optimization 规则优化这段代码"
AI: [自动加载对应规则]
```

### 2. 临时覆盖规则

```
你: "这次特殊情况,需要重写整个文件 (我已备份)"
AI: [遵循明确指示]
```

### 3. 批量操作

```
你: "批量重构所有Tokenizer实现类,统一接口"
AI: [创建TODO, 分步骤执行]
```

### 4. 验证规则效果

```
你: "请解释你为什么使用replace_in_file而不是write_to_file"
AI: "根据核心约束规则,禁止重写整个文件..."
```

---

**提示**: 本指南会随着项目发展持续更新,建议定期查阅。

**最后更新**: 2026-01-11  
**版本**: v1.0
