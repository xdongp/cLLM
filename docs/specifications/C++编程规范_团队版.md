# C++编程规范 (团队版)

**精简版编程规范供团队成员参考** 📋

> **注意**: 本文档是给团队成员阅读的精简版。  
> AI开发时使用的详细约束规则请查看：[.codebuddy/rules/](../../.codebuddy/rules/)

---

## 🎯 核心规范摘要

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| **类名** | PascalCase | `TokenizerManager`, `HFTokenizer` |
| **函数名** | camelCase | `encodeText()`, `getTokenId()` |
| **变量名** | camelCase | `maxLength`, `tokenId` |
| **成员变量** | camelCase + 后缀`_` | `tokenizer_`, `maxLength_` |
| **常量** | UPPER_CASE | `MAX_LENGTH`, `DEFAULT_SIZE` |
| **文件名** | snake_case | `hf_tokenizer.h`, `tokenizer.cpp` |

### 目录结构

```
cpp/cLLM/
├── include/cllm/     # 公共头文件
│   ├── tokenizer/
│   ├── scheduler/
│   └── ...
├── src/              # 实现文件
│   ├── tokenizer/
│   ├── scheduler/
│   └── ...
└── tests/            # 单元测试
    ├── test_tokenizer.cpp
    └── ...
```

---

## 📝 编码规范

### 1. 头文件

```cpp
#pragma once

#include <vector>
#include <string>

namespace cllm {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    
    // 编码文本
    std::vector<int> encode(const std::string& text);
    
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace cllm
```

### 2. 实现文件

```cpp
#include "cllm/tokenizer/tokenizer.h"

namespace cllm {

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {
    CLLM_INFO("Tokenizer initialized");
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    // 实现
    return impl_->encode(text);
}

}  // namespace cllm
```

### 3. 智能指针

```cpp
// ✅ 使用智能指针
std::unique_ptr<Tokenizer> tokenizer = std::make_unique<Tokenizer>();
std::shared_ptr<Model> model = std::make_shared<Model>();

// ❌ 避免裸指针
Tokenizer* tokenizer = new Tokenizer();  // 不推荐
```

---

## 🔧 关键原则

### RAII 原则

```cpp
class Resource {
public:
    Resource() { /* 获取资源 */ }
    ~Resource() { /* 释放资源 */ }
    
    // 禁止拷贝
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
};
```

### 错误处理

```cpp
// 使用异常
if (!file.is_open()) {
    throw std::runtime_error("Failed to open file");
}

// 或返回 bool
bool loadModel(const std::string& path) {
    if (!exists(path)) {
        CLLM_ERROR("Model file not found: {}", path);
        return false;
    }
    return true;
}
```

---

## 📚 完整规范

详细的编程规范请查阅:

1. **AI约束规则** (最权威)
   - [核心约束](../../.codebuddy/rules/always/00_core_constraints.md)
   - [架构规则](../../.codebuddy/rules/always/01_architecture_rules.md)

2. **完整文档** (参考)
   - [C++编程规范_完整版](./C++编程规范_完整版.md)
   - [生成代码规范_完整版](./生成代码规范_完整版.md)

---

**版本**: v2.0 (精简版)  
**更新日期**: 2026-01-11  
**维护者**: cLLM Core Team
