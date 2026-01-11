# 🚀 CodeBuddy 快速参考

## 📋 核心规则速查

### ❌ 绝对禁止

```
❌ 删除 .codebuddy/ 目录
❌ 重写整个文件 (use replace_in_file)
❌ 创建临时脚本 (test_*.py, benchmark_*.py)
❌ 使用裸指针 (use unique_ptr/shared_ptr)
❌ git push --force
❌ git reset --hard
❌ 修改 git config
```

### ✅ 必须遵守

```
✅ 使用 replace_in_file 精确修改
✅ 修改前 read_file 读取完整内容
✅ 修改后 read_lints 检查语法
✅ 并行执行工具调用
✅ 复杂任务创建 TODO
✅ 保留原始缩进格式
```

---

## 🔧 常用命令模板

### 新增功能

```
你: "给 [ClassName] 添加 [功能描述]"

AI 会:
1. read_file 读取相关文件
2. search_content 搜索依赖
3. 创建 TODO (如果复杂)
4. replace_in_file 精确修改
5. read_lints 验证语法
```

### Bug修复

```
你: "[ClassName]::[method] 在 [场景] 时崩溃,请修复"

AI 会:
1. 读取相关代码
2. 定位问题
3. 精确修复
4. 添加边界检查
5. 建议添加测试
```

### 性能优化

```
你: "优化 [功能] 的性能"

AI 会:
1. 自动加载 performance_optimization.md
2. 分析瓶颈
3. 应用优化策略
4. 添加性能监控
```

### 代码重构

```
你: "重构 [ClassName],提取 [逻辑] 到单独的类"

AI 会:
1. 自动加载 refactoring_guide.md
2. 制定重构计划
3. 分步骤执行
4. 每步后验证测试
```

---

## 📁 目录结构速查

```
.codebuddy/
├── rules/
│   ├── always/              # 自动生效
│   │   ├── 00_core_constraints.md
│   │   ├── 01_architecture_rules.md
│   │   └── 02_workflow_standards.md
│   ├── manual/              # 手动触发
│   │   ├── performance_optimization.md  (优化/性能)
│   │   └── refactoring_guide.md        (重构/解耦)
│   └── requested/           # 按需加载
│       └── tokenizer_integration.md
├── project.yaml             # 项目配置
├── README.md               # 配置说明
└── USAGE_GUIDE.md          # 使用指南
```

---

## 🎯 编码规范速查

### 命名规范

```cpp
// 类名: PascalCase
class TokenizerManager {};

// 函数名: camelCase
bool loadTokenizer(const std::string& path);

// 变量名: camelCase
int maxTokens = 100;

// 成员变量: camelCase_
std::unique_ptr<ITokenizer> tokenizer_;

// 常量: kPascalCase
const int kMaxBatchSize = 32;
```

### 文件组织

```
include/cllm/tokenizer/hf_tokenizer.h
src/tokenizer/hf_tokenizer.cpp
tests/test_hf_tokenizer.cpp
```

### 必备头文件

```cpp
#include "cllm/common/logger.h"      // 日志
#include <nlohmann/json.hpp>         // JSON
#include <yaml-cpp/yaml.h>           // YAML
#include <memory>                    // 智能指针
```

---

## 🏗️ 模块依赖速查

```
允许的依赖方向 (上层→下层):

HTTP → TokenizerManager → ModelExecutor → Backend
         ↓                     ↓
      Request              KVCache
                              ↓
                        Infrastructure

❌ 禁止: Infrastructure → 上层
❌ 禁止: Backend → TokenizerManager
❌ 禁止: 任何循环依赖
```

---

## 🔍 工具使用速查

### read_file

```python
# ✅ 并行读取
read_file("file1.h")
read_file("file2.h")
read_file("file3.h")
```

### replace_in_file

```python
# ✅ 精确替换
replace_in_file(
    "file.cpp",
    old_str="    int oldCode() {\n        return 0;\n    }",
    new_str="    int newCode() {\n        return calculate();\n    }"
)
```

### search_content

```python
# ✅ 正则搜索
search_content("class\\s+Tokenizer", ".h,.cpp")
search_content("include.*<tokenizers", ".h,.cpp")
```

### execute_command

```bash
# ✅ 安全命令
execute_command("mkdir -p dir", requires_approval=false)

# ⚠️  危险命令
execute_command("rm -rf dir/", requires_approval=true)
```

---

## 📊 TODO管理速查

### 创建

```python
todo_write(
    merge=false,
    todos='[
        {"id":"1","status":"in_progress","content":"任务1"},
        {"id":"2","status":"pending","content":"任务2"}
    ]'
)
```

### 更新

```python
todo_write(
    merge=true,
    todos='[
        {"id":"1","status":"completed","content":"任务1"},
        {"id":"2","status":"in_progress","content":"任务2"}
    ]'
)
```

---

## ⚡ 性能优化速查

### CPU优化

```cpp
// ✅ 避免拷贝
void process(const std::vector<int>& data);

// ✅ 预分配
std::vector<int> tokens;
tokens.reserve(size);

// ✅ 并行处理
BS::thread_pool pool;
pool.parallelize_loop(0, n, [&](int i, int j) { /*...*/ });
```

### 内存优化

```cpp
// ✅ 对象池
ObjectPool<Tensor> pool;
auto tensor = pool.acquire();

// ✅ 智能指针
std::unique_ptr<T> ptr;  // 优先
std::shared_ptr<T> ptr;  // 需要共享时
```

---

## 🚨 错误预防速查

### 修改前

```
[ ] read_file 读取目标文件?
[ ] old_str 完全匹配?
[ ] 检查 #include?
[ ] 检查命名空间?
[ ] 条件编译宏?
```

### 修改后

```
[ ] read_lints 检查?
[ ] 编译通过?
[ ] 测试通过?
[ ] TODO 更新?
```

---

## 📚 文档快速链接

| 需求 | 查看 |
|------|------|
| 快速上手 | `.codebuddy/USAGE_GUIDE.md` |
| 禁止事项 | `.codebuddy/rules/always/00_*.md` |
| 架构设计 | `.codebuddy/rules/always/01_*.md` |
| 工作流程 | `.codebuddy/rules/always/02_*.md` |
| 性能优化 | `.codebuddy/rules/manual/performance_*.md` |
| 代码重构 | `.codebuddy/rules/manual/refactoring_*.md` |

---

## 💡 最佳实践

```
✅ 明确需求: "优化 encode 性能,目标提升3倍"
✅ 提供上下文: "参考 HFTokenizer 实现"
✅ 分步骤: "第一步: ..., 第二步: ..."
✅ 要求验证: "实现后运行 read_lints"
```

---

**快速查阅** | **v1.0** | **2026-01-11**
