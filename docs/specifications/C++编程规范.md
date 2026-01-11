# C++ 编程规范

**cLLM 项目编程规范 - 团队快速参考** 📋

---

## 📖 关于本文档

本文档是 cLLM 项目的 C++ 编程规范，供团队成员日常开发参考。

### 文档层次

| 文档 | 用途 | 适合场景 |
|------|------|---------|
| **本文档** | 快速参考 | 日常开发、Code Review |
| [C++编程规范参考手册](./C++编程规范参考手册.md) | 详细规范 | 深入学习、疑难问题 |
| [AI约束规则](../../.codebuddy/rules/) | AI开发规则 | AI 自动遵守 |

> **提示**: 需要详细规范时，请查看 [C++编程规范参考手册](./C++编程规范参考手册.md)

---

## 🎯 核心规范速查

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| **类名** | PascalCase | `TokenizerManager`, `HFTokenizer` |
| **函数名** | camelCase | `encodeText()`, `getTokenId()` |
| **变量名** | camelCase | `maxLength`, `tokenId` |
| **成员变量** | camelCase + `_` 后缀 | `tokenizer_`, `maxLength_` |
| **常量** | UPPER_CASE | `MAX_LENGTH`, `DEFAULT_SIZE` |
| **命名空间** | 小写 | `cllm`, `cllm::tokenizer` |
| **文件名** | snake_case | `hf_tokenizer.h`, `tokenizer.cpp` |

### 文件组织

```
cpp/cLLM/
├── include/cllm/       # 公共头文件
│   ├── tokenizer/      # 分词器模块
│   ├── scheduler/      # 调度器模块
│   ├── model/          # 模型执行器
│   └── ...
├── src/                # 实现文件
│   ├── tokenizer/
│   ├── scheduler/
│   └── ...
└── tests/              # 单元测试
    ├── test_tokenizer.cpp
    └── ...
```

---

## 📝 代码示例

### 1. 头文件模板

```cpp
#pragma once

#include <vector>
#include <string>
#include <memory>

namespace cllm {

/**
 * @brief 分词器接口
 * 
 * 负责文本编码和解码
 */
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    
    /**
     * @brief 编码文本为 Token IDs
     * @param text 输入文本
     * @return Token ID 列表
     */
    std::vector<int> encode(const std::string& text);
    
    /**
     * @brief 解码 Token IDs 为文本
     * @param ids Token ID 列表
     * @return 解码后的文本
     */
    std::string decode(const std::vector<int>& ids);
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cllm
```

### 2. 实现文件模板

```cpp
#include "cllm/tokenizer/tokenizer.h"

#include <spdlog/spdlog.h>

namespace cllm {

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {
    spdlog::info("Tokenizer initialized");
}

Tokenizer::~Tokenizer() = default;

std::vector<int> Tokenizer::encode(const std::string& text) {
    if (text.empty()) {
        spdlog::warn("Empty text provided to encode()");
        return {};
    }
    
    return impl_->encode(text);
}

std::string Tokenizer::decode(const std::vector<int>& ids) {
    if (ids.empty()) {
        return "";
    }
    
    return impl_->decode(ids);
}

}  // namespace cllm
```

### 3. 智能指针使用

```cpp
// ✅ 推荐：使用智能指针
std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>();
std::shared_ptr<Model> model = std::make_shared<Model>();

// ✅ 传递智能指针
void processData(const std::shared_ptr<Model>& model) {
    model->forward(data);
}

// ❌ 避免：裸指针（除非必要）
Tokenizer* tokenizer = new Tokenizer();  // 不推荐
delete tokenizer;  // 容易忘记
```

---

## 🔧 关键原则

### RAII (Resource Acquisition Is Initialization)

```cpp
class FileHandler {
public:
    FileHandler(const std::string& path) {
        file_.open(path);
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open file");
        }
    }
    
    ~FileHandler() {
        if (file_.is_open()) {
            file_.close();  // 自动释放资源
        }
    }
    
    // 禁止拷贝
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;
    
    // 允许移动
    FileHandler(FileHandler&&) = default;
    FileHandler& operator=(FileHandler&&) = default;
    
private:
    std::ifstream file_;
};

// 使用
{
    FileHandler handler("data.txt");
    // 使用文件...
}  // 自动关闭文件
```

### 错误处理

```cpp
// 方式1: 使用异常（推荐用于构造函数和严重错误）
void loadModel(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file not found: " + path);
    }
    // 加载模型...
}

// 方式2: 返回布尔值（推荐用于可恢复的错误）
bool tryLoadModel(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        spdlog::error("Model file not found: {}", path);
        return false;
    }
    // 加载模型...
    return true;
}

// 方式3: 返回 std::optional（推荐用于可能失败的查询）
std::optional<TokenInfo> findToken(int tokenId) {
    auto it = tokenMap_.find(tokenId);
    if (it == tokenMap_.end()) {
        return std::nullopt;
    }
    return it->second;
}
```

### const 正确性

```cpp
class Cache {
public:
    // const 成员函数：不修改对象状态
    size_t getSize() const { return size_; }
    bool isEmpty() const { return size_ == 0; }
    
    // 非 const 成员函数：可能修改对象状态
    void clear() { size_ = 0; }
    
    // const 引用参数：避免拷贝，不修改参数
    void add(const std::string& key, const Data& value) {
        cache_[key] = value;
        ++size_;
    }
    
private:
    size_t size_ = 0;
    std::unordered_map<std::string, Data> cache_;
};
```

---

## ⚡ 性能最佳实践

### 1. 避免不必要的拷贝

```cpp
// ✅ 使用 const 引用
void processTokens(const std::vector<int>& tokens) {
    for (const auto& token : tokens) {
        // 处理...
    }
}

// ❌ 按值传递（会拷贝）
void processTokens(std::vector<int> tokens) {  // 不推荐
    // ...
}

// ✅ 使用移动语义
std::vector<int> createTokens() {
    std::vector<int> tokens = {1, 2, 3};
    return tokens;  // 自动移动（C++17）
}
```

### 2. 预留容量

```cpp
// ✅ 预留容量
std::vector<int> tokens;
tokens.reserve(1000);  // 避免多次重新分配
for (int i = 0; i < 1000; ++i) {
    tokens.push_back(i);
}

// ✅ 使用 emplace_back
std::vector<Token> tokenList;
tokenList.emplace_back(id, text);  // 原地构造
```

### 3. 字符串优化

```cpp
// ✅ 使用 string_view（只读）
void printText(std::string_view text) {
    std::cout << text << std::endl;
}

// ✅ 拼接字符串使用 +=
std::string result;
result.reserve(totalSize);  // 预留空间
for (const auto& part : parts) {
    result += part;
}
```

---

## 🧪 测试规范

### 单元测试模板

```cpp
#include <gtest/gtest.h>
#include "cllm/tokenizer/tokenizer.h"

namespace cllm {
namespace test {

class TokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = std::make_unique<Tokenizer>();
    }
    
    void TearDown() override {
        tokenizer_.reset();
    }
    
    std::unique_ptr<Tokenizer> tokenizer_;
};

TEST_F(TokenizerTest, EncodeBasicText) {
    std::string text = "Hello, world!";
    auto tokens = tokenizer_->encode(text);
    
    EXPECT_FALSE(tokens.empty());
    EXPECT_GT(tokens.size(), 0);
}

TEST_F(TokenizerTest, DecodeTokens) {
    std::vector<int> tokens = {1, 2, 3};
    std::string text = tokenizer_->decode(tokens);
    
    EXPECT_FALSE(text.empty());
}

}  // namespace test
}  // namespace cllm
```

---

## 📚 延伸阅读

### 详细规范
- [C++编程规范参考手册](./C++编程规范参考手册.md) - 完整编程规范
- [AI约束规则说明](./AI约束规则说明.md) - AI 规则体系说明

### AI 规则（自动生效）
- [核心约束](../../.codebuddy/rules/always/00_core_constraints.md)
- [架构规则](../../.codebuddy/rules/always/01_architecture_rules.md)
- [工作流程](../../.codebuddy/rules/always/02_workflow_standards.md)

### 开发指南
- [快速开始](../guides/快速开始.md)
- [CodeBuddy使用指南](../guides/CodeBuddy使用指南.md)
- [开发环境搭建](../guides/开发环境搭建.md)

---

## 🔍 常见问题

### Q1: 什么时候使用 unique_ptr vs shared_ptr?

**A**: 
- **unique_ptr**: 独占所有权（推荐默认使用）
- **shared_ptr**: 共享所有权（需要多个对象持有时）

```cpp
// unique_ptr: 资源独占
std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>();

// shared_ptr: 资源共享
std::shared_ptr<Model> model = std::make_shared<Model>();
cache->setModel(model);     // Cache 持有引用
executor->setModel(model);  // Executor 也持有引用
```

### Q2: 成员变量为什么要加 _ 后缀?

**A**: 
- 区分成员变量和局部变量
- 避免命名冲突
- 提高代码可读性

```cpp
class Example {
public:
    void setValue(int value) {
        value_ = value;  // 清晰区分成员变量和参数
    }
    
private:
    int value_;  // 成员变量
};
```

### Q3: 什么时候使用异常 vs 返回值?

**A**:
- **异常**: 构造函数失败、不可恢复的错误
- **返回值**: 可恢复的错误、正常的失败情况

```cpp
// 异常: 构造失败
Model::Model(const std::string& path) {
    if (!load(path)) {
        throw std::runtime_error("Failed to load model");
    }
}

// 返回值: 可恢复
bool Model::reload(const std::string& path) {
    if (!load(path)) {
        return false;  // 可以重试
    }
    return true;
}
```

---

## ✅ Code Review 检查清单

开发完成后，请自查以下项目：

- [ ] 命名符合规范（类名、函数名、变量名）
- [ ] 使用智能指针管理资源
- [ ] RAII 原则正确应用
- [ ] const 正确性
- [ ] 避免不必要的拷贝
- [ ] 错误处理得当
- [ ] 代码有适当注释
- [ ] 单元测试覆盖
- [ ] 无编译警告
- [ ] 通过 clang-tidy 检查

---

**版本**: v3.0  
**更新日期**: 2026-01-11  
**维护者**: cLLM Core Team  
**反馈**: 如有问题或建议，请提交 Issue
