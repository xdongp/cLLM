# CodeBuddy 使用指南

**AI 辅助开发助手使用说明** 🤖

---

## 什么是 CodeBuddy？

CodeBuddy 是 cLLM 项目的 AI 编程助手约束系统，通过规则文件规范 AI 的代码生成行为，确保:

- ✅ 遵守项目编码规范
- ✅ 符合架构设计原则
- ✅ 生成高质量代码
- ✅ 避免常见错误

---

## 📂 规则文件位置

CodeBuddy 规则位于 `.codebuddy/rules/` 目录:

```
.codebuddy/
├── rules/
│   ├── always/          # 自动生效的核心规则
│   │   ├── 00_core_constraints.md
│   │   ├── 01_architecture_rules.md
│   │   └── 02_workflow_standards.md
│   ├── manual/          # 手动触发的专项规则
│   │   ├── performance_optimization.md
│   │   └── refactoring_guide.md
│   └── requested/       # 按需加载的规则
│       └── tokenizer_integration.md
├── USAGE_GUIDE.md       # 详细使用指南
└── QUICK_REFERENCE.md   # 快速参考
```

---

## 🎯 规则类型

### 1. Always 规则 (自动生效)

AI 每次对话都会自动加载这些规则:

| 规则文件 | 内容 | 作用 |
|---------|------|------|
| `00_core_constraints.md` | 核心约束、编码规范 | 最高优先级规则 |
| `01_architecture_rules.md` | 架构设计、模块依赖 | 确保架构一致性 |
| `02_workflow_standards.md` | 工作流程、工具使用 | 规范开发流程 |

### 2. Manual 规则 (手动触发)

需要手动提及才会加载:

| 规则文件 | 触发关键词 | 适用场景 |
|---------|-----------|---------|
| `performance_optimization.md` | "优化"、"性能" | 性能优化任务 |
| `refactoring_guide.md` | "重构"、"优化代码" | 代码重构任务 |

### 3. Requested 规则 (按需加载)

AI 根据需要自动调用:

| 规则文件 | 适用场景 |
|---------|---------|
| `tokenizer_integration.md` | Tokenizer 集成开发 |

---

## 💡 常见对话示例

### 示例1: 添加新功能

**用户**:
```
给 TokenizerManager 添加批量编码功能
```

**AI 会**:
1. ✅ 读取 `Tokenizer模块设计.md`
2. ✅ 遵守 `00_core_constraints.md` 的命名规范
3. ✅ 遵守 `01_architecture_rules.md` 的模块依赖
4. ✅ 使用 `replace_in_file` 精确修改（不重写整个文件）
5. ✅ 运行 `read_lints` 检查语法错误

### 示例2: 性能优化

**用户**:
```
优化 Tokenizer::encode 性能，目标提升 3 倍
```

**AI 会**:
1. ✅ 自动加载 `manual/performance_optimization.md`
2. ✅ 应用缓存、并行等优化策略
3. ✅ 使用 `Timer` 类进行性能测试
4. ✅ 生成性能对比报告

### 示例3: 代码重构

**用户**:
```
重构 HFTokenizer，提取加载逻辑到单独的类
```

**AI 会**:
1. ✅ 自动加载 `manual/refactoring_guide.md`
2. ✅ 遵守 SOLID 原则
3. ✅ 保持接口不变
4. ✅ 添加单元测试

### 示例4: 修复 Bug

**用户**:
```
Qwen3 模型加载失败，tokenizer.json 找不到，请修复
```

**AI 会**:
1. ✅ 读取相关代码
2. ✅ 搜索相关配置
3. ✅ 分析错误原因
4. ✅ 精确修复（不改动无关代码）
5. ✅ 验证修复结果

---

## 🛡️ AI 遵守的核心约束

### 禁止事项

AI **绝对不会**:

- ❌ 删除 `.codebuddy/` 目录
- ❌ 重写整个文件（使用 `replace_in_file` 精确替换）
- ❌ 创建临时脚本文件
- ❌ 使用裸指针（使用智能指针）
- ❌ 使用全局变量
- ❌ Git 危险操作 (`force push`, `hard reset`)
- ❌ 破坏模块依赖规则

### 必须遵守

AI **必须遵守**:

- ✅ C++17 标准
- ✅ 命名规范 (PascalCase/camelCase/snake_case)
- ✅ 文件组织 (`include/cllm/`, `src/`)
- ✅ 模块依赖规则
- ✅ 5 阶段工作流（理解→分析→规划→执行→验证）
- ✅ 并行工具调用（提升效率）

---

## 📖 详细文档

### 完整指南

详细的 CodeBuddy 使用说明请查看:

- [CodeBuddy 完整使用指南](../../.codebuddy/USAGE_GUIDE.md)
- [CodeBuddy 快速参考](../../.codebuddy/QUICK_REFERENCE.md)
- [CodeBuddy 设置完成报告](../../.codebuddy/SETUP_COMPLETE.md)

### 规则文档

查看 AI 遵守的详细规则:

- [核心约束规则](../../.codebuddy/rules/always/00_core_constraints.md)
- [架构设计规则](../../.codebuddy/rules/always/01_architecture_rules.md)
- [工作流程规范](../../.codebuddy/rules/always/02_workflow_standards.md)

---

## 🎯 使用技巧

### 1. 明确需求

❌ **模糊**: "优化代码"  
✅ **明确**: "优化 TokenizerManager::encode 性能，目标提升 3 倍"

### 2. 提供上下文

❌ **无上下文**: "修复 Bug"  
✅ **有上下文**: "Qwen3 加载失败，tokenizer.json 找不到，请修复"

### 3. 分步骤执行

✅ **推荐**:
```
第一步: 实现 ITokenizer 接口
第二步: 添加条件编译
第三步: 更新 CMakeLists.txt
```

### 4. 利用专项规则

需要性能优化时，明确提及:
```
使用性能优化规则，优化 encode 函数
```

---

## 🔍 验证 AI 行为

### 测试场景1: 遵守命名规范

**测试**:
```
创建一个新的 Tokenizer 类
```

**验证**: AI 应该使用:
- 类名: `PascalCase` (如 `HFTokenizer`)
- 函数名: `camelCase` (如 `encodeText`)
- 文件名: `snake_case` (如 `hf_tokenizer.h`)

### 测试场景2: 精确修改

**测试**:
```
给 TokenizerManager 添加一个方法
```

**验证**: AI 应该:
- ✅ 先 `read_file` 读取文件
- ✅ 使用 `replace_in_file` 精确插入
- ❌ 不应该重写整个文件

### 测试场景3: 模块依赖

**测试**:
```
让 Tokenizer 模块依赖 ModelExecutor
```

**验证**: AI 应该:
- ❌ 拒绝此操作
- ✅ 说明违反了模块依赖规则
- ✅ 提供正确的依赖方向

---

## 📊 与传统开发对比

| 维度 | 传统开发 | 使用 CodeBuddy |
|------|---------|---------------|
| **规范遵守** | 手动检查 | AI 自动遵守 |
| **代码风格** | Code Review 发现 | 生成时即符合 |
| **架构一致性** | 容易偏离 | 强制约束 |
| **文档查阅** | 手动查找 | AI 自动参考 |
| **错误预防** | 事后修复 | 事前预防 |
| **效率提升** | - | 3-5倍 |

---

## 🚨 注意事项

### 1. 规则冲突

如果人类文档与 AI 规则冲突:

- ✅ AI 规则优先（`.codebuddy/rules/`）
- 📝 人类文档仅供参考（`docs/specifications/`）

### 2. 规则更新

修改 AI 行为时:

```bash
# 编辑规则文件
vim .codebuddy/rules/always/00_core_constraints.md

# 提交到 Git
git add .codebuddy/
git commit -m "docs: 更新 AI 约束规则"
```

### 3. 规则测试

更新规则后，测试 AI 行为:

```
请按照最新规则，给 TokenizerManager 添加批量编码功能
```

---

## 📞 问题排查

### Q1: AI 没有遵守规则？

**检查**:
1. 规则文件是否在 `always/` 目录？
2. 规则描述是否清晰、可执行？
3. 是否有冲突的规则？

### Q2: AI 行为异常？

**排查**:
1. 查看 AI 是否读取了规则文件
2. 检查规则文件语法是否正确
3. 尝试更明确的需求描述

### Q3: 想添加新规则？

**步骤**:
1. 确定规则类型（always/manual/requested）
2. 创建 Markdown 文件
3. 使用清晰的 ✅/❌ 格式
4. 测试规则是否生效

---

## 🎉 总结

CodeBuddy 让 AI 开发:

- ✅ 更规范（自动遵守编码规范）
- ✅ 更高效（3-5倍效率提升）
- ✅ 更安全（预防常见错误）
- ✅ 更一致（架构设计一致）

**开始使用 CodeBuddy，享受高效开发！** 🚀

---

**更新日期**: 2026-01-11  
**维护者**: cLLM Core Team  
**详细文档**: [.codebuddy/USAGE_GUIDE.md](../../.codebuddy/USAGE_GUIDE.md)
