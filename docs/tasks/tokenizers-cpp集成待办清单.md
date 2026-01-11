# tokenizers-cpp 集成待办清单

## 📋 文档概述

**创建日期**: 2026-01-11  
**状态**: 部分完成，需继续集成  
**优先级**: 高  
**预计工作量**: 3-5 天

---

## ✅ 已完成的工作

### 1. 代码准备（100%）
- [x] HFTokenizer 头文件定义 (`include/cllm/tokenizer/hf_tokenizer.h`)
- [x] HFTokenizer 基础实现 (`src/tokenizer/hf_tokenizer.cpp`)
- [x] 单元测试代码 (`tests/test_hf_tokenizer.cpp` - 17个测试)
- [x] 示例代码 (`examples/hf_tokenizer_example.cpp` - 5个示例)
- [x] CMakeLists.txt 配置支持
- [x] 安装脚本 (`scripts/install_tokenizers_cpp.sh`)

### 2. 库准备（100%）
- [x] 克隆 tokenizers-cpp 到 `third_party/`
- [x] 初始化子模块（msgpack, sentencepiece）
- [x] 编译 tokenizers-cpp（生成 `libtokenizers_c.a` 和 `libtokenizers_cpp.a`）

### 3. 文档（100%）
- [x] Tokenizers库安装指南
- [x] tokenizers-cpp集成分析
- [x] tokenizers-cpp集成完成报告
- [x] tokenizers-cpp集成验证指南
- [x] tokenizers-cpp集成执行总结

---

## 🚧 未完成的任务

### 任务分类

```
├── 核心任务（必须完成）
│   ├── Task-1: API 适配修复
│   ├── Task-2: 编译和链接配置
│   └── Task-3: 测试验证
│
├── 功能增强（重要）
│   ├── Task-4: 特殊Token处理
│   ├── Task-5: 批量处理支持
│   └── Task-6: 性能优化
│
└── 文档和维护（可选）
    ├── Task-7: API文档更新
    ├── Task-8: 故障排查指南
    └── Task-9: CI/CD 集成
```

---

## 📌 核心任务（必须完成）

### Task-1: API 适配修复 🔴 高优先级

**问题描述**:  
当前 `HFTokenizer` 实现使用的 API 与 `tokenizers-cpp` 实际 API 不匹配：
- 使用了不存在的 `FromFile()` 方法
- `Encode()` 和 `Decode()` 参数不匹配
- 缺少文件读取逻辑

**需要修改的文件**:
- `src/tokenizer/hf_tokenizer.cpp`

**具体任务**:

#### Task-1.1: 修复 load() 方法
```cpp
// 现有代码（错误）:
tokenizer_ = tokenizers::Tokenizer::FromFile(tokenizerJsonPath);

// 需要改为:
// 1. 读取 tokenizer.json 文件内容
std::ifstream f(tokenizerJsonPath);
std::string json_blob((std::istreambuf_iterator<char>(f)), 
                      std::istreambuf_iterator<char>());

// 2. 使用 FromBlobJSON 创建
tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 编译通过

---

#### Task-1.2: 修复 encode() 方法
```cpp
// 现有代码（错误）:
auto encoding = tokenizer_->Encode(text, addSpecialTokens);

// 需要改为:
auto encoding = tokenizer_->Encode(text);

// 注意: tokenizers-cpp 不支持 addSpecialTokens 参数
// 需要手动处理特殊Token
```

**难度**: ⭐️⭐️  
**预计时间**: 20 分钟  
**验证方式**: 编译通过

---

#### Task-1.3: 修复 decode() 方法
```cpp
// 现有代码（错误）:
std::string text = tokenizer_->Decode(tokenIds, skipSpecialTokens);

// 需要改为:
std::string text = tokenizer_->Decode(tokenIds);

// 注意: tokenizers-cpp 不支持 skipSpecialTokens 参数
// 需要手动过滤特殊Token
```

**难度**: ⭐️⭐️  
**预计时间**: 20 分钟  
**验证方式**: 编译通过

---

#### Task-1.4: 修复 tokenize() 方法
```cpp
// 现有代码（错误）:
auto encoding = tokenizer_->Encode(text, false);

// 需要改为:
auto encoding = tokenizer_->Encode(text);
```

**难度**: ⭐️  
**预计时间**: 10 分钟  
**验证方式**: 编译通过

---

#### Task-1.5: 更新类型定义
```cpp
// 检查所有使用 uint32_t 的地方，改为 int32_t
// tokenizers-cpp 使用 int32_t 而不是 uint32_t

// 示例:
std::vector<uint32_t> tokenIds;  // 错误
std::vector<int32_t> tokenIds;   // 正确
```

**难度**: ⭐️  
**预计时间**: 10 分钟  
**验证方式**: 编译通过

---

### Task-2: 编译和链接配置 🔴 高优先级

**问题描述**:  
虽然 CMake 能找到库，但可能存在链接问题

**具体任务**:

#### Task-2.1: 验证库链接
```bash
# 确保以下库都被正确链接:
- libtokenizers_cpp.a  (C++ 包装层)
- libtokenizers_c.a    (Rust 核心库)
- libsentencepiece.a   (SentencePiece 依赖)

# 检查命令:
cd build
make test_hf_tokenizer VERBOSE=1 | grep "tokenizers"
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 编译链接成功

---

#### Task-2.2: 解决潜在的符号冲突
```bash
# Rust 库可能需要额外的系统库
# macOS 可能需要:
- Security.framework
- Foundation.framework

# 更新 CMakeLists.txt:
if(APPLE)
    target_link_libraries(cllm_core
        ${TOKENIZERS_LIBRARIES}
        "-framework Security"
        "-framework Foundation"
    )
endif()
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 链接成功，无 undefined symbols 错误

---

#### Task-2.3: 添加 Rust 标准库依赖
```bash
# tokenizers-cpp 依赖 Rust，可能需要链接:
- pthread
- dl (Linux)
- resolv (macOS)

# 检查是否需要添加到 CMakeLists.txt
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 链接成功

---

### Task-3: 测试验证 🟡 中优先级

**具体任务**:

#### Task-3.1: 编译测试程序
```bash
cd build
cmake .. -DUSE_TOKENIZERS_CPP=ON
make test_hf_tokenizer -j8
```

**难度**: ⭐️  
**预计时间**: 10 分钟（假设 Task-1 和 Task-2 完成）  
**验证方式**: 编译成功，生成 `bin/test_hf_tokenizer`

---

#### Task-3.2: 运行基本测试（不需要模型）
```bash
cd build
./bin/test_hf_tokenizer --gtest_filter="HFTokenizerBasicTest.*"

# 预期: 8个基本测试通过
```

**难度**: ⭐️⭐️  
**预计时间**: 20 分钟  
**验证方式**: 所有基本测试通过

---

#### Task-3.3: 准备测试模型
```bash
# 下载一个 HuggingFace 模型（包含 tokenizer.json）
# 推荐: Qwen/Qwen2-7B-Instruct 或 meta-llama/Llama-2-7b-hf

# 设置环境变量:
export CLLM_TEST_MODEL_PATH=/path/to/model
```

**难度**: ⭐️  
**预计时间**: 30 分钟（取决于下载速度）  
**验证方式**: 模型目录包含 `tokenizer.json` 和 `config.json`

---

#### Task-3.4: 运行集成测试
```bash
cd build
export CLLM_TEST_MODEL_PATH=/path/to/model
./bin/test_hf_tokenizer --gtest_filter="HFTokenizerIntegrationTest.*"

# 预期: 6个集成测试通过
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 所有集成测试通过

---

#### Task-3.5: 运行示例程序
```bash
cd build
./bin/hf_tokenizer_example /path/to/model

# 预期: 5个示例正常运行，输出正确
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 示例运行无错误，输出合理

---

## 🎯 功能增强（重要）

### Task-4: 特殊Token处理 🟡 中优先级

**问题描述**:  
`tokenizers-cpp` API 不支持 `addSpecialTokens` 和 `skipSpecialTokens` 参数，需要手动实现

**具体任务**:

#### Task-4.1: 实现 addSpecialTokens 功能
```cpp
std::vector<int> HFTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    auto ids = tokenizer_->Encode(text);
    
    if (addSpecialTokens) {
        // 在开头添加 BOS token
        if (bosId_ != -1) {
            ids.insert(ids.begin(), bosId_);
        }
        // 在结尾添加 EOS token
        if (eosId_ != -1) {
            ids.push_back(eosId_);
        }
    }
    
    return ids;
}
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 测试用例验证特殊Token正确添加

---

#### Task-4.2: 实现 skipSpecialTokens 功能
```cpp
std::string HFTokenizer::decode(const std::vector<int>& ids, bool skipSpecialTokens) {
    std::vector<int32_t> tokenIds;
    
    for (int id : ids) {
        // 如果需要跳过特殊Token
        if (skipSpecialTokens && isSpecialToken(id)) {
            continue;
        }
        tokenIds.push_back(static_cast<int32_t>(id));
    }
    
    return tokenizer_->Decode(tokenIds);
}
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 测试用例验证特殊Token正确过滤

---

#### Task-4.3: 更新测试用例
```cpp
// 更新现有测试用例，验证特殊Token处理
TEST_F(HFTokenizerIntegrationTest, SpecialTokensHandling) {
    // 测试 addSpecialTokens = true
    auto ids_with = tokenizer_->encode(text, true);
    EXPECT_EQ(ids_with.front(), tokenizer_->getBosId());
    EXPECT_EQ(ids_with.back(), tokenizer_->getEosId());
    
    // 测试 addSpecialTokens = false
    auto ids_without = tokenizer_->encode(text, false);
    EXPECT_NE(ids_without.front(), tokenizer_->getBosId());
}
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 测试通过

---

### Task-5: 批量处理支持 🟢 低优先级

**具体任务**:

#### Task-5.1: 实现批量编码
```cpp
std::vector<std::vector<int>> HFTokenizer::encodeBatch(
    const std::vector<std::string>& texts,
    bool addSpecialTokens) {
    
    std::vector<std::vector<int>> results;
    results.reserve(texts.size());
    
    for (const auto& text : texts) {
        results.push_back(encode(text, addSpecialTokens));
    }
    
    return results;
}
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 测试批量处理正确性

---

#### Task-5.2: 实现批量解码
```cpp
std::vector<std::string> HFTokenizer::decodeBatch(
    const std::vector<std::vector<int>>& batch_ids,
    bool skipSpecialTokens) {
    
    std::vector<std::string> results;
    results.reserve(batch_ids.size());
    
    for (const auto& ids : batch_ids) {
        results.push_back(decode(ids, skipSpecialTokens));
    }
    
    return results;
}
```

**难度**: ⭐️⭐️  
**预计时间**: 30 分钟  
**验证方式**: 测试批量处理正确性

---

### Task-6: 性能优化 🟢 低优先级

**具体任务**:

#### Task-6.1: 添加缓存机制
```cpp
// 缓存常用文本的编码结果
class HFTokenizer {
private:
    std::unordered_map<std::string, std::vector<int>> encodeCache_;
    size_t maxCacheSize_ = 10000;
};
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 2 小时  
**验证方式**: 性能测试，缓存命中率 > 50%

---

#### Task-6.2: 并行批量处理
```cpp
// 使用线程池并行处理批量请求
std::vector<std::vector<int>> HFTokenizer::encodeBatchParallel(
    const std::vector<std::string>& texts) {
    // 使用 BS_thread_pool.hpp
}
```

**难度**: ⭐️⭐️⭐️⭐️  
**预计时间**: 3 小时  
**验证方式**: 性能测试，加速比 > 2x

---

## 📚 文档和维护（可选）

### Task-7: API文档更新 🟢 低优先级

#### Task-7.1: 更新头文件注释
```cpp
// 为所有公共方法添加详细的 Doxygen 注释
/**
 * @brief 编码文本为Token IDs
 * @param text 输入文本
 * @param addSpecialTokens 是否添加特殊Token（BOS/EOS）
 * @return Token IDs 向量
 * @note tokenizers-cpp 不原生支持特殊Token参数，由本类手动处理
 */
std::vector<int> encode(const std::string& text, bool addSpecialTokens = true);
```

**难度**: ⭐️⭐️  
**预计时间**: 1 小时  
**验证方式**: 文档生成成功

---

#### Task-7.2: 创建 API 参考文档
```markdown
# HFTokenizer API 参考

## 类方法

### encode()
- 功能: 将文本编码为Token IDs
- 参数: ...
- 返回值: ...
- 示例: ...
```

**难度**: ⭐️⭐️  
**预计时间**: 2 小时  
**验证方式**: 文档完整清晰

---

### Task-8: 故障排查指南 🟢 低优先级

#### Task-8.1: 收集常见问题
```markdown
# HFTokenizer 故障排查

## 问题1: 编译错误 - undefined reference to `tokenizers::Tokenizer::FromFile`
解决方案: ...

## 问题2: 运行时错误 - Failed to load tokenizer
解决方案: ...
```

**难度**: ⭐️⭐️  
**预计时间**: 1 小时  
**依赖**: 完成 Task-1 到 Task-3

---

#### Task-8.2: 创建调试检查清单
```markdown
## HFTokenizer 调试检查清单

编译阶段:
- [ ] tokenizers-cpp 已正确安装
- [ ] CMake 找到了 tokenizers-cpp 库
- [ ] 链接了所有必需的库

运行阶段:
- [ ] tokenizer.json 文件存在
- [ ] 模型路径正确
- [ ] 配置文件可读
```

**难度**: ⭐️  
**预计时间**: 30 分钟

---

### Task-9: CI/CD 集成 🟢 低优先级

#### Task-9.1: 添加 GitHub Actions 工作流
```yaml
name: Build and Test HFTokenizer

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Rust
        run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      - name: Install tokenizers-cpp
        run: ./scripts/install_tokenizers_cpp.sh
      - name: Build cLLM
        run: |
          mkdir build && cd build
          cmake .. -DUSE_TOKENIZERS_CPP=ON
          make -j$(nproc)
      - name: Run tests
        run: cd build && ./bin/test_hf_tokenizer
```

**难度**: ⭐️⭐️⭐️  
**预计时间**: 2 小时  
**验证方式**: GitHub Actions 运行成功

---

## 📊 任务优先级矩阵

| 任务ID | 任务名称 | 优先级 | 难度 | 预计时间 | 依赖 |
|--------|----------|--------|------|----------|------|
| **Task-1.1** | 修复 load() 方法 | 🔴 高 | ⭐️⭐️ | 30分钟 | 无 |
| **Task-1.2** | 修复 encode() 方法 | 🔴 高 | ⭐️⭐️ | 20分钟 | Task-1.1 |
| **Task-1.3** | 修复 decode() 方法 | 🔴 高 | ⭐️⭐️ | 20分钟 | Task-1.1 |
| **Task-1.4** | 修复 tokenize() 方法 | 🔴 高 | ⭐️ | 10分钟 | Task-1.1 |
| **Task-1.5** | 更新类型定义 | 🔴 高 | ⭐️ | 10分钟 | 无 |
| **Task-2.1** | 验证库链接 | 🔴 高 | ⭐️⭐️⭐️ | 1小时 | Task-1.* |
| **Task-2.2** | 解决符号冲突 | 🔴 高 | ⭐️⭐️⭐️ | 1小时 | Task-2.1 |
| **Task-2.3** | 添加Rust依赖 | 🔴 高 | ⭐️⭐️ | 30分钟 | Task-2.1 |
| **Task-3.1** | 编译测试程序 | 🟡 中 | ⭐️ | 10分钟 | Task-1.*, Task-2.* |
| **Task-3.2** | 运行基本测试 | 🟡 中 | ⭐️⭐️ | 20分钟 | Task-3.1 |
| **Task-3.3** | 准备测试模型 | 🟡 中 | ⭐️ | 30分钟 | 无 |
| **Task-3.4** | 运行集成测试 | 🟡 中 | ⭐️⭐️⭐️ | 1小时 | Task-3.1, Task-3.3 |
| **Task-3.5** | 运行示例程序 | 🟡 中 | ⭐️⭐️ | 30分钟 | Task-3.1, Task-3.3 |
| **Task-4.1** | 实现 addSpecialTokens | 🟡 中 | ⭐️⭐️⭐️ | 1小时 | Task-3.* |
| **Task-4.2** | 实现 skipSpecialTokens | 🟡 中 | ⭐️⭐️⭐️ | 1小时 | Task-3.* |
| **Task-4.3** | 更新测试用例 | 🟡 中 | ⭐️⭐️ | 30分钟 | Task-4.1, Task-4.2 |
| **Task-5.1** | 实现批量编码 | 🟢 低 | ⭐️⭐️ | 30分钟 | Task-3.* |
| **Task-5.2** | 实现批量解码 | 🟢 低 | ⭐️⭐️ | 30分钟 | Task-3.* |
| **Task-6.1** | 添加缓存机制 | 🟢 低 | ⭐️⭐️⭐️ | 2小时 | Task-3.* |
| **Task-6.2** | 并行批量处理 | 🟢 低 | ⭐️⭐️⭐️⭐️ | 3小时 | Task-5.* |
| **Task-7.1** | 更新头文件注释 | 🟢 低 | ⭐️⭐️ | 1小时 | Task-1.* |
| **Task-7.2** | 创建API参考文档 | 🟢 低 | ⭐️⭐️ | 2小时 | Task-7.1 |
| **Task-8.1** | 收集常见问题 | 🟢 低 | ⭐️⭐️ | 1小时 | Task-3.* |
| **Task-8.2** | 创建调试检查清单 | 🟢 低 | ⭐️ | 30分钟 | Task-8.1 |
| **Task-9.1** | 添加CI/CD | 🟢 低 | ⭐️⭐️⭐️ | 2小时 | Task-3.* |

---

## 🎯 推荐执行顺序

### 阶段1: 核心修复（必须完成）⏱️ 预计 4 小时

```
Day 1 上午:
1. Task-1.1 → Task-1.2 → Task-1.3 → Task-1.4 → Task-1.5
   (API 适配修复，约 1.5 小时)

Day 1 下午:
2. Task-2.1 → Task-2.2 → Task-2.3
   (编译和链接配置，约 2.5 小时)
```

### 阶段2: 测试验证（必须完成）⏱️ 预计 3 小时

```
Day 2 上午:
3. Task-3.1 → Task-3.2
   (编译和基本测试，约 30 分钟)

4. Task-3.3 (并行进行，下载模型)

Day 2 下午:
5. Task-3.4 → Task-3.5
   (集成测试和示例，约 1.5 小时)
```

### 阶段3: 功能增强（重要）⏱️ 预计 5 小时

```
Day 3:
6. Task-4.1 → Task-4.2 → Task-4.3
   (特殊Token处理，约 2.5 小时)

7. Task-5.1 → Task-5.2
   (批量处理，约 1 小时)

8. Task-6.1 (可选)
   (缓存优化，约 2 小时)
```

### 阶段4: 文档和维护（可选）⏱️ 预计 7 小时

```
Day 4-5:
9. Task-7.1 → Task-7.2
   (API 文档，约 3 小时)

10. Task-8.1 → Task-8.2
    (故障排查，约 1.5 小时)

11. Task-9.1
    (CI/CD，约 2 小时)
```

---

## 🔧 Agent 分工建议

### Agent-1: 核心开发者（C++ 专家）
**负责任务**:
- Task-1.* (API 适配修复)
- Task-2.* (编译和链接配置)
- Task-4.* (特殊Token处理)
- Task-5.* (批量处理)

**技能要求**:
- 熟悉 C++17
- 了解 CMake
- 有 Rust FFI 经验更佳

---

### Agent-2: 测试工程师
**负责任务**:
- Task-3.* (测试验证)
- Task-4.3 (测试用例更新)
- Task-8.* (故障排查指南)

**技能要求**:
- 熟悉 Google Test
- 了解模型文件格式
- 有测试经验

---

### Agent-3: 性能优化工程师
**负责任务**:
- Task-6.* (性能优化)

**技能要求**:
- 熟悉多线程编程
- 了解缓存设计
- 有性能分析经验

---

### Agent-4: 文档工程师
**负责任务**:
- Task-7.* (API 文档)
- Task-8.* (故障排查)

**技能要求**:
- 技术写作能力
- 了解 Markdown 和 Doxygen
- 有 API 文档经验

---

### Agent-5: DevOps 工程师
**负责任务**:
- Task-9.* (CI/CD 集成)

**技能要求**:
- 熟悉 GitHub Actions
- 了解 Docker
- 有 CI/CD 经验

---

## 📝 进度跟踪模板

### 任务进度表

| 任务ID | 负责Agent | 状态 | 开始时间 | 完成时间 | 备注 |
|--------|-----------|------|----------|----------|------|
| Task-1.1 | Agent-1 | ⏳ 待开始 | - | - | - |
| Task-1.2 | Agent-1 | ⏳ 待开始 | - | - | - |
| ... | ... | ... | ... | ... | ... |

### 状态说明
- ⏳ 待开始
- 🏃 进行中
- ✅ 已完成
- ❌ 已阻塞
- ⚠️ 需要帮助

---

## 🚨 已知问题和风险

### 问题1: tokenizers-cpp API 限制
**描述**: tokenizers-cpp 的 API 比较简单，不支持特殊Token参数  
**影响**: 需要手动处理特殊Token  
**风险等级**: 🟡 中  
**缓解措施**: Task-4 专门处理这个问题

### 问题2: Rust 库链接复杂
**描述**: tokenizers-cpp 依赖 Rust，链接可能复杂  
**影响**: 可能出现 undefined symbols 错误  
**风险等级**: 🔴 高  
**缓解措施**: Task-2 专门处理链接问题

### 问题3: 模型兼容性
**描述**: 不同模型的 tokenizer.json 格式可能略有差异  
**影响**: 可能无法加载某些模型  
**风险等级**: 🟡 中  
**缓解措施**: 多模型测试（Task-3.4）

---

## 📞 联系和协作

### 问题反馈
- 遇到问题时，在对应 Task 中添加注释
- 使用 GitHub Issues 跟踪 Bug
- 重要问题及时沟通

### 代码审查
- 每个 Task 完成后提交 PR
- 至少一个其他 Agent 审查
- 通过所有测试后合并

### 文档更新
- 完成任务后更新本文档
- 标记任务状态
- 记录遇到的问题和解决方案

---

## 📚 参考资源

### 官方文档
- [tokenizers-cpp GitHub](https://github.com/mlc-ai/tokenizers-cpp)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
- [Rust FFI 指南](https://doc.rust-lang.org/nomicon/ffi.html)

### 项目内文档
- `docs/guides/Tokenizers库安装指南.md`
- `docs/guides/tokenizers-cpp集成验证指南.md`
- `docs/guides/tokenizers-cpp集成执行总结.md`

### 相关代码
- `include/cllm/tokenizer/hf_tokenizer.h`
- `src/tokenizer/hf_tokenizer.cpp`
- `tests/test_hf_tokenizer.cpp`
- `examples/hf_tokenizer_example.cpp`

---

**文档版本**: v1.0  
**最后更新**: 2026-01-11  
**维护者**: AI Assistant  
**状态**: ✅ 完整，可用于分工执行
