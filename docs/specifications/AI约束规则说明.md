# AI 约束规则说明

**人类文档与AI规则的关系说明** 🤖

---

## 📊 双轨文档体系

cLLM 项目采用 **人类文档** 与 **AI规则** 分离的体系:

| 体系 | 位置 | 适用对象 | 特点 |
|------|------|---------|------|
| **人类文档** | `docs/specifications/` | 👥 团队成员 | 精简、友好、易读 |
| **AI规则** | `.codebuddy/rules/` | 🤖 AI助手 | 详细、可执行、可验证 |

---

## 🎯 规则位置

### AI 约束规则目录

```
.codebuddy/
├── rules/
│   ├── always/          # 自动生效的核心规则
│   │   ├── 00_core_constraints.md      ⭐ 核心约束
│   │   ├── 01_architecture_rules.md    ⭐ 架构规则
│   │   └── 02_workflow_standards.md    ⭐ 工作流程
│   ├── manual/          # 手动触发的专项规则
│   │   ├── performance_optimization.md # 性能优化
│   │   └── refactoring_guide.md        # 代码重构
│   └── requested/       # 按需加载的规则
│       └── tokenizer_integration.md    # Tokenizer集成
├── USAGE_GUIDE.md       # 完整使用指南
└── QUICK_REFERENCE.md   # 快速参考
```

---

## 📝 规则内容

### 核心约束 (00_core_constraints.md)

包含:
- ✅ 详细的命名规范
- ✅ 文件组织规范
- ✅ 编码标准
- ✅ 禁止事项清单
- ✅ 错误预防检查清单

### 架构规则 (01_architecture_rules.md)

包含:
- ✅ 模块依赖规则
- ✅ 接口设计原则
- ✅ 设计模式应用
- ✅ 目录结构规范

### 工作流程 (02_workflow_standards.md)

包含:
- ✅ 5阶段标准流程
- ✅ 工具使用规范
- ✅ 并行执行策略
- ✅ 验证检查步骤

---

## 🔄 与人类文档的关系

### 规则优先级

```
AI开发时:
  .codebuddy/rules/ (详细规则) > docs/specifications/ (精简文档)

人类阅读时:
  docs/specifications/ (友好文档) 优先查看
```

### 内容同步

- **核心规范**: `.codebuddy/rules/` 是权威来源
- **人类文档**: `docs/specifications/` 是精简摘要
- **更新策略**: 先更新 AI 规则,再同步人类文档

---

## 📖 如何使用

### 人类开发者

1. **日常开发**: 查看 [C++编程规范_团队版.md](./C++编程规范_团队版.md)
2. **详细参考**: 查看 [完整版规范](./C++编程规范_完整版.md)
3. **AI行为**: 了解 [CodeBuddy使用指南](../guides/CodeBuddy使用指南.md)

### AI 开发时

AI 会自动:
1. ✅ 加载 `.codebuddy/rules/always/` 中的所有规则
2. ✅ 根据关键词加载 `manual/` 规则
3. ✅ 按需调用 `requested/` 规则
4. ✅ 严格遵守所有约束

---

## 🔧 如何更新规则

### 更新流程

```bash
# 1. 编辑 AI 规则
vim .codebuddy/rules/always/00_core_constraints.md

# 2. 测试 AI 行为
# 与 AI 对话,验证规则生效

# 3. 同步人类文档 (可选)
vim docs/specifications/C++编程规范_团队版.md

# 4. 提交
git add .codebuddy/ docs/specifications/
git commit -m "docs: 更新编码规范"
```

### 注意事项

- ✅ 使用清晰的 ✅/❌ 格式
- ✅ 规则要具体、可执行
- ✅ 避免模糊的描述
- ✅ 提供代码示例

---

## 📚 延伸阅读

- [CodeBuddy 使用指南](../guides/CodeBuddy使用指南.md)
- [CodeBuddy 完整文档](../../.codebuddy/USAGE_GUIDE.md)
- [规则快速参考](../../.codebuddy/QUICK_REFERENCE.md)

---

**版本**: v1.0  
**更新日期**: 2026-01-11  
**维护者**: cLLM Core Team
