# CTokenizer 测试方案

## 1. 测试目标
验证CTokenizer模块的正确性、稳定性和性能，包括：
- 基本编解码功能
- 特殊字符处理
- 多语言支持
- 性能指标

## 2. 测试环境
- 硬件：标准开发环境
- 软件：
  - macOS Sequoia
  - C++17
  - SentencePiece 0.1.95

## 3. 测试用例设计

### 3.1 单元测试
```cpp
TEST(CTokenizerTest, LoadModel) {
    CTokenizer tokenizer;
    EXPECT_TRUE(tokenizer.load("models/qwen.model"));
}

TEST(CTokenizerTest, EncodeDecode) {
    CTokenizer tokenizer;
    tokenizer.load("models/qwen.model");
    
    std::string text = "Hello world";
    auto ids = tokenizer.encode(text);
    auto decoded = tokenizer.decode(ids);
    
    EXPECT_EQ(text, decoded);
}
```

### 3.2 边界测试
```cpp
TEST(CTokenizerTest, EmptyString) {
    CTokenizer tokenizer;
    tokenizer.load("models/qwen.model");
    
    auto ids = tokenizer.encode("");
    EXPECT_TRUE(ids.empty());
    
    auto text = tokenizer.decode({});
    EXPECT_TRUE(text.empty());
}

TEST(CTokenizerTest, SpecialCharacters) {
    CTokenizer tokenizer;
    tokenizer.load("models/qwen.model");
    
    std::string text = "Hello\nWorld\t!@#$%^&*()";
    auto ids = tokenizer.encode(text);
    auto decoded = tokenizer.decode(ids);
    
    EXPECT_EQ(text, decoded);
}
```

### 3.3 多语言测试
```cpp
TEST(CTokenizerTest, Multilingual) {
    CTokenizer tokenizer;
    tokenizer.load("models/qwen.model");
    
    std::vector<std::string> texts = {
        "你好世界",
        "こんにちは世界",
        "안녕하세요 세계",
        "Привет мир"
    };
    
    for (const auto& text : texts) {
        auto ids = tokenizer.encode(text);
        auto decoded = tokenizer.decode(ids);
        EXPECT_EQ(text, decoded);
    }
}
```

### 3.4 性能测试
```cpp
TEST(CTokenizerTest, Performance) {
    CTokenizer tokenizer;
    tokenizer.load("models/qwen.model");
    
    std::string longText(10000, 'A'); // 10KB文本
    
    auto start = std::chrono::high_resolution_clock::now();
    auto ids = tokenizer.encode(longText);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 100); // 编码时间应小于100ms
    
    start = std::chrono::high_resolution_clock::now();
    auto decoded = tokenizer.decode(ids);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 100); // 解码时间应小于100ms
}
```

## 4. 测试执行流程
1. 准备测试模型文件 `models/qwen.model`
2. 编译测试程序
3. 执行测试用例
4. 生成测试报告

## 5. 预期结果
所有测试用例应通过，性能指标满足要求：
- 单次编解码延迟 < 100ms (10KB文本)
- 内存占用 < 50MB
- 支持并发请求

## 6. 测试报告模板
```markdown
# CTokenizer 测试报告
- 测试时间: [日期]
- 测试环境: [环境信息]
- 通过率: [x/y]
- 性能指标:
  - 平均编码延迟: [x]ms
  - 平均解码延迟: [y]ms
- 问题列表:
  - [问题描述1]
  - [问题描述2]
```

## 7. 后续改进
根据测试结果优化：
- 内存管理
- 并发性能
- 异常处理