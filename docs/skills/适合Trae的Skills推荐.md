# 适合 Trae AI 助手的 Skills 推荐

本文档总结了从 [antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills) 仓库中适合 Trae AI 助手使用的 skills。

## 核心推荐 Skills

### 1. code-reviewer ⭐⭐⭐

**描述**: 精英代码审查专家，专门用于现代 AI 驱动的代码分析、安全漏洞、性能优化和生产可靠性。

**何时使用**:
- 代码审查任务或工作流
- 需要指导、最佳实践或代码审查检查清单

**能力**:
- AI 驱动的代码分析（集成 Trag、Bito、Codiga、GitHub Copilot）
- 现代静态分析工具（SonarQube、CodeQL、Semgrep）
- 安全代码审查（OWASP Top 10、输入验证、认证和授权）
- 性能和可扩展性分析（数据库查询优化、内存泄漏、缓存策略）
- 配置和基础设施审查（生产配置、Kubernetes 清单、CI/CD 管道）
- 现代开发实践（TDD、BDD、功能标志、蓝绿部署）

**为什么适合**:
- Trae AI 助手经常需要审查代码、检查最佳实践
- 需要确保代码质量、安全性和可维护性
- 需要提供可操作的反馈和代码示例

---

### 2. cpp-pro ⭐⭐⭐

**描述**: 现代 C++ 专家，处理 RAII、智能指针、STL 算法、模板元编程、移动语义和性能优化。

**何时使用**:
- C++ 专业任务或工作流
- 需要指导、最佳实践或 C++ 专业检查清单

**重点领域**:
- 现代 C++ (C++11/14/17/20/23) 特性
- RAII 和智能指针（unique_ptr、shared_ptr）
- 模板元编程和概念
- 移动语义和完美转发
- STL 算法和容器
- 并发（std::thread 和 atomics）
- 异常安全保证

**为什么适合**:
- cLLM 项目主要是 C++ 代码
- 需要编写符合现代 C++ 标准的代码
- 需要处理内存管理、并发和性能优化

---

### 3. debugger ⭐⭐⭐

**描述**: 调试专家，专门用于错误、测试失败和意外行为的根本原因分析。

**何时使用**:
- 调试器任务或工作流
- 需要指导、最佳实践或调试器检查清单

**调试过程**:
- 分析错误消息和日志
- 检查最近的代码更改
- 形成和测试假设
- 添加战略性调试日志
- 检查变量状态

**为什么适合**:
- Trae AI 助手经常需要调试代码
- 需要分析错误、定位问题根本原因
- 需要提供具体的修复方案和预防建议

---

### 4. llm-app-patterns ⭐⭐⭐

**描述**: 构建生产就绪的 LLM 应用程序模式，涵盖 RAG 管道、Agent 架构、Prompt IDE 和 LLMOps 监控。

**何时使用**:
- 设计 LLM 驱动的应用程序
- 实现 RAG（检索增强生成）
- 构建 AI Agent
- 设置 LLMOps 监控
- 在 Agent 架构之间进行选择

**核心模式**:
1. **RAG 管道架构**: 文档摄取、嵌入和存储、检索策略、上下文生成
2. **Agent 架构**: ReAct 模式、函数调用、计划-执行、多 Agent 协作
3. **Prompt IDE 模式**: 提示模板、版本控制和 A/B 测试、提示链
4. **LLMOps 和可观察性**: 指标跟踪、日志和追踪、评估框架

**为什么适合**:
- cLLM 是一个 LLM 应用程序
- 需要了解 LLM 应用的最佳实践
- 需要实现 RAG、Agent 等高级功能

---

### 5. rag-engineer ⭐⭐

**描述**: RAG 系统架构师，精通嵌入模型、向量数据库、分块策略和检索优化。

**何时使用**:
- 构建 RAG 系统
- 向量搜索
- 嵌入
- 语义搜索
- 文档检索

**能力**:
- 向量嵌入和相似性搜索
- 文档分块和预处理
- 检索管道设计
- 语义搜索实现
- 上下文窗口优化
- 混合搜索（关键字 + 语义）

**为什么适合**:
- cLLM 未来可能会添加 RAG 功能
- 需要了解向量数据库和嵌入模型
- 需要优化检索质量和上下文使用

---

### 6. architecture ⭐⭐

**描述**: 架构决策框架，涵盖需求分析、权衡评估、ADR 文档。

**何时使用**:
- 制定架构决策
- 分析系统设计

**核心原则**:
- "简单性是终极的复杂性"
- 从简单开始
- 只在证明必要时才添加复杂性
- 你总是可以稍后添加模式
- 移除复杂性比添加它难得多

**为什么适合**:
- Trae AI 助手经常需要做出架构决策
- 需要权衡不同的方案
- 需要记录决策理由

---

### 7. clean-code ⭐⭐⭐

**描述**: 实用编码标准 - 简洁、直接、不过度工程、不必要的注释。

**何时使用**:
- 编写代码时
- 重构代码时
- 审查代码时

**核心原则**:
- **SRP** - 单一职责 - 每个函数/类只做一件事
- **DRY** - 不要重复自己 - 提取重复项，重用
- **KISS** - 保持简单 - 有效的最简单解决方案
- **YAGNI** - 你不会需要它 - 不要构建未使用的功能
- **Boy Scout** - 比你发现时更整洁地离开代码

**为什么适合**:
- Trae AI 助手经常需要编写高质量、可维护的代码
- 需要遵循编码最佳实践
- 需要避免过度工程化

---

### 8. evaluation ⭐⭐

**描述**: 构建 Agent 系统的评估框架。

**何时使用**:
- 系统测试 Agent 性能
- 验证上下文工程选择
- 衡量随时间的改进
- 在部署前捕获回归
- 为 Agent 管道构建质量门

**评估方法论**:
- **LLM-as-Judge**: LLM 基于评估，可扩展到大型测试集
- **人类评估**: 人类评估捕获自动化遗漏的内容
- **最终状态评估**: 对于改变持久状态的 Agent

**为什么适合**:
- Trae AI 助手经常需要评估代码质量和系统性能
- 需要建立评估框架
- 需要跟踪指标随时间的变化

---

## 次要推荐 Skills

### 9. cost-optimization

**描述**: 通过资源调整、标记策略、预留实例和支出分析来优化云成本。

**何时使用**:
- 减少云支出
- 调整资源大小
- 实施成本治理
- 优化多云成本
- 满足预算约束

**为什么适合**:
- 虽然 cLLM 目前主要是本地运行，但未来可能会涉及云部署
- 需要了解云成本优化策略

---

### 10. prompt-engineer

**描述**: 提示工程专家，专门用于优化 LLM 提示以获得更好的结果。

**何时使用**:
- 优化 LLM 提示
- 设计提示模板
- 测试提示变体
- 分析提示性能

**为什么适合**:
- Trae AI 助手经常需要与 LLM 交互
- 需要优化提示以获得更好的结果
- 需要设计可重用的提示模板

---

### 11. testing-patterns

**描述**: 测试驱动开发和测试最佳实践专家。

**何时使用**:
- 编写测试
- 设计测试
- 修复测试
- QA 工作流

**为什么适合**:
- Trae AI 助手经常需要编写和维护测试
- 需要确保代码质量和可靠性

---

### 12. api-patterns

**描述**: API 设计模式和最佳实践专家。

**何时使用**:
- 设计 API
- 实现 REST/GraphQL API
- API 版本控制
- API 文档

**为什么适合**:
- cLLM 有 HTTP 服务器和 API 端点
- 需要设计符合最佳实践的 API

---

## 使用建议

### 安装 Skills

Skills 已安装在 `.agent/skills/` 目录下。

### 调用 Skills

在对话中自然地使用：

```
"Use @code-reviewer to review this code"
"Use @cpp-pro to write modern C++ code"
"Use @debugger to analyze this error"
"Use @llm-app-patterns to design RAG system"
```

### 组合使用

多个 skills 可以组合使用以解决复杂问题：

```
"Use @architecture to design the system, then @cpp-pro to implement it, and @code-reviewer to review the code"
```

---

## 总结

| Skill | 优先级 | 适用场景 |
|-------|---------|---------|
| code-reviewer | 高 | 代码审查、质量保证 |
| cpp-pro | 高 | C++ 开发、现代 C++ 特性 |
| debugger | 高 | 调试、错误分析 |
| llm-app-patterns | 高 | LLM 应用设计、RAG、Agent |
| rag-engineer | 中 | RAG 系统、向量搜索 |
| architecture | 中 | 架构决策、系统设计 |
| clean-code | 高 | 编码标准、代码质量 |
| evaluation | 中 | 评估框架、性能测试 |
| cost-optimization | 低 | 云成本优化 |
| prompt-engineer | 中 | 提示工程、LLM 交互 |
| testing-patterns | 中 | 测试驱动开发 |
| api-patterns | 中 | API 设计 |

---

## 参考资源

- [antigravity-awesome-skills 仓库](https://github.com/sickn33/antigravity-awesome-skills)
- [完整技能目录](https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/CATALOG.md)
- [使用指南](https://github.com/sickn33/antigravity-awesome-skills/blob/main/docs/GETTING_STARTED.md)

---

## 更新记录

- 2026-02-05: 初始版本，基于 antigravity-awesome-skills v4.0.0
