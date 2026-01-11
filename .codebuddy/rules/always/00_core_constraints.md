# 🎯 cLLM 核心约束规则

> **优先级**: CRITICAL | 所有AI操作必须遵守本规则

---

## 📌 项目基本信息

- **项目名称**: cLLM (C++ Large Language Model Inference Engine)
- **语言标准**: C++17
- **命名空间**: `cllm`
- **构建系统**: CMake 3.15+
- **编译器**: GCC 9+ / Clang 10+

---

## 🚫 绝对禁止事项

### 目录与文件

- ❌ **禁止删除 `.codebuddy/` 目录及其内容**
- ❌ **禁止删除 `docs/` 目录中的设计文档**
- ❌ **禁止重写整个文件** (必须使用 `replace_in_file` 精确修改)
- ❌ **禁止创建临时脚本文件** (如 `benchmark_*.py`, `test_*.py`)
- ❌ **禁止生成超过 800 行的单个文件**
- ❌ **禁止添加 emoji** (除非用户明确要求，或在注释/日志中使用规范emoji)

### Git 操作

- ❌ **禁止修改 git config**
- ❌ **禁止执行 `git push --force`**
- ❌ **禁止执行 `git reset --hard`**
- ❌ **禁止跳过钩子** (`--no-verify`, `--no-gpg-sign`)
- ⚠️  **禁止主动提交代码** (除非用户明确要求)

### 代码质量

- ❌ **禁止使用裸指针** (使用 `std::unique_ptr` / `std::shared_ptr`)
- ❌ **禁止使用全局变量** (使用单例模式或依赖注入)
- ❌ **禁止在头文件中实现大段代码** (除模板外)
- ❌ **禁止循环依赖**
- ❌ **禁止使用 `using namespace std;`** (头文件中)

---

## ✅ 必须遵守的工作流程

### 1. 代码修改前置检查

```markdown
每次修改代码前必须:
1. ✅ 使用 `read_file` 读取目标文件完整内容
2. ✅ 使用 `search_content` 搜索相关依赖
3. ✅ 检查是否需要同步修改配套文件 (.h ↔ .cpp)
4. ✅ 规划修改范围 (使用 `todo_write`)
```

### 2. 代码修改执行

```markdown
1. ✅ 使用 `replace_in_file` 进行精确替换
   - old_str 必须完全匹配 (包括空白符)
   - 保留原始缩进和格式
   - 一次替换不超过 100 行

2. ✅ 大规模修改分批执行
   - 相邻 20 行内的修改可合并
   - 超过 20 行的修改分多次调用

3. ✅ 修改后立即验证
   - 运行 `read_lints` 检查语法错误
   - 检查编译通过
```

### 3. 文件操作规范

```markdown
创建新文件:
- ✅ 必须有充分理由 (新增模块/功能)
- ✅ 遵循项目目录结构
- ✅ 同步更新 CMakeLists.txt

删除文件:
- ⚠️  必须确认不被其他模块依赖
- ⚠️  提示用户确认
- ✅ 同步更新 CMakeLists.txt
```

---

## 📁 项目目录结构

```
cLLM/
├── .codebuddy/              # CodeBuddy配置 (🔒 禁止修改)
│   ├── rules/               # 约束规则
│   ├── memory/              # 记忆存储
│   └── context/             # 上下文缓存
│
├── include/cllm/            # 公共头文件
│   ├── common/              # 通用工具
│   ├── tokenizer/           # 分词器接口
│   ├── model/               # 模型推理
│   ├── kv_cache/            # KV缓存
│   ├── scheduler/           # 调度器
│   ├── http/                # HTTP服务
│   └── ...
│
├── src/                     # 实现文件 (与include对应)
│   ├── common/
│   ├── tokenizer/
│   ├── CTokenizer/          # C++原生分词器
│   ├── model/
│   └── ...
│
├── tests/                   # 单元测试
├── examples/                # 示例代码
├── docs/                    # 设计文档 (🔒 重要)
│   ├── analysis/            # 分析报告
│   ├── modules/             # 模块设计
│   └── implementation/      # 实施报告
│
├── config/                  # 配置文件
├── scripts/                 # 工具脚本
├── third_party/             # 第三方库
└── CMakeLists.txt           # CMake配置
```

---

## 🔧 编译与依赖

### 核心依赖库

| 库 | 用途 | 头文件包含 |
|----|------|-----------|
| **spdlog** | 日志系统 | `#include <spdlog/spdlog.h>` |
| **nlohmann/json** | JSON解析 | `#include <nlohmann/json.hpp>` |
| **yaml-cpp** | YAML配置 | `#include <yaml-cpp/yaml.h>` |
| **LibTorch** | 模型推理 | `#include <torch/torch.h>` |
| **BS::thread_pool** | 线程池 | `#include <BS_thread_pool.hpp>` |
| **SentencePiece** | 分词 | `#include <sentencepiece_processor.h>` |
| **tokenizers-cpp** | HF分词 | `#include <tokenizers_cpp.h>` (条件编译) |

### 条件编译宏

```cpp
// HuggingFace Tokenizer支持
#ifdef USE_TOKENIZERS_CPP
  // 使用tokenizers-cpp实现
#else
  // 回退到NativeTokenizer
#endif

// Kylin后端支持
#ifdef USE_KYLIN_BACKEND
  // 使用Kylin加速
#endif
```

---

## 📝 代码风格规范

### 命名约定

```cpp
// 1. 文件命名: snake_case
// 头文件: hf_tokenizer.h
// 实现文件: hf_tokenizer.cpp

// 2. 类名: PascalCase
class HFTokenizer;
class TokenizerManager;

// 3. 函数名: camelCase
bool loadTokenizer(const std::string& path);
std::vector<int> encodeText(const std::string& text);

// 4. 变量名: camelCase + 类型后缀
std::unique_ptr<ITokenizer> tokenizer_;  // 成员变量后缀 _
int maxTokens;                           // 局部变量无后缀
const int kMaxBatchSize = 32;            // 常量前缀 k

// 5. 命名空间: 全小写
namespace cllm {
namespace detail {
}
}
```

### 头文件格式

```cpp
#pragma once

#include <cllm/path/to/dependency.h>  // 项目头文件
#include <vector>                     // 标准库
#include <memory>
#include <nlohmann/json.hpp>          // 第三方库

namespace cllm {

/**
 * @brief 类简要说明
 * 
 * 详细说明（可选）
 */
class MyClass {
public:
    /**
     * @brief 构造函数
     * @param param 参数说明
     */
    explicit MyClass(int param);
    
    ~MyClass();
    
    // 接口方法
    bool doSomething();
    
private:
    // 辅助方法
    void helperMethod();
    
    // 成员变量
    int value_;
    std::unique_ptr<Dependency> dependency_;
};

} // namespace cllm
```

---

## 🛠️ 日志规范

### 日志宏使用

```cpp
#include "cllm/common/logger.h"

// 使用项目定义的宏 (推荐)
CLLM_INFO("Tokenizer loaded successfully");
CLLM_WARN("Token count exceeds limit: %d", tokenCount);
CLLM_ERROR("Failed to load model: %s", path.c_str());
CLLM_DEBUG("Cache hit rate: %.2f%%", hitRate);

// 或使用spdlog (次选)
spdlog::info("Message");
spdlog::warn("Warning");
```

### 日志Emoji规范 (可选)

```cpp
CLLM_INFO("✅ Initialization complete");
CLLM_WARN("⚠️  Memory usage high: %d MB", memUsage);
CLLM_ERROR("❌ Failed to connect to server");
CLLM_INFO("🔸 Using HFTokenizer");
CLLM_INFO("🚀 Starting generation...");
```

---

## 🧪 测试规范

### 测试文件命名

```
tests/
├── test_tokenizer.cpp           # 单元测试
├── test_kv_cache.cpp
├── integration_test.cpp         # 集成测试
└── benchmark_*.cpp              # 性能测试
```

### 测试用例编写

```cpp
#include <gtest/gtest.h>
#include "cllm/tokenizer/hf_tokenizer.h"

TEST(HFTokenizerTest, EncodeDecodeRoundtrip) {
    cllm::HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load("path/to/model"));
    
    std::string text = "Hello, world!";
    auto ids = tokenizer.encode(text, true);
    auto decoded = tokenizer.decode(ids, true);
    
    EXPECT_EQ(text, decoded);
}
```

---

## ⚡ 性能优化原则

### 1. 避免不必要的拷贝

```cpp
// ❌ 错误: 值传递大对象
void processTokens(std::vector<int> tokens);

// ✅ 正确: 引用传递
void processTokens(const std::vector<int>& tokens);

// ✅ 正确: 移动语义
void setTokens(std::vector<int>&& tokens) {
    tokens_ = std::move(tokens);
}
```

### 2. 预分配内存

```cpp
std::vector<int> tokens;
tokens.reserve(estimatedSize);  // ✅ 避免多次realloc
```

### 3. 使用并行处理

```cpp
#include <BS_thread_pool.hpp>

BS::thread_pool pool(numThreads);
pool.parallelize_loop(0, batchSize, 
    [&](int start, int end) {
        // 并行处理
    }
);
pool.wait();
```

---

## 📊 错误处理

### 异常使用规范

```cpp
// ✅ 使用异常传递致命错误
if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
}

// ✅ 使用bool返回值表示操作成功/失败
bool loadModel(const std::string& path) {
    try {
        // 加载逻辑
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("Load failed: %s", e.what());
        return false;
    }
}

// ✅ 使用std::optional表示可选结果
std::optional<Token> getToken(int id) {
    if (id < 0 || id >= vocabSize_) {
        return std::nullopt;
    }
    return tokens_[id];
}
```

---

## 🔍 代码审查检查清单

每次修改后自检:

- [ ] 是否添加了必要的 `#include`?
- [ ] 命名空间是否正确?
- [ ] 条件编译宏是否完整?
- [ ] 是否有内存泄漏风险?
- [ ] 是否有线程安全问题?
- [ ] 日志输出是否充分?
- [ ] 错误处理是否完善?
- [ ] 是否添加了必要的注释?
- [ ] 是否通过 `read_lints` 检查?
- [ ] 是否需要更新文档?

---

## 📚 参考文档

修改代码前必须阅读的文档:

1. **核心设计**: `docs/cLLM详细设计.md`
2. **编码规范**: `docs/C++编程规范.md`
3. **模块设计**: `docs/modules/` 目录
4. **实施报告**: `docs/analysis/` 目录
5. **构建指南**: `docs/工程编译设计.md`

---

**最后更新**: 2026-01-11  
**版本**: v1.0  
**维护者**: cLLM Core Team
