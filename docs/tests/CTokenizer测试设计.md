# CTokenizer测试设计

## 1. 测试目标

### 1.1 核心功能验证
- 验证文本编码功能（text → token IDs）
- 验证文本解码功能（token IDs → text）
- 验证特殊Token处理（BOS/EOS/PAD/UNK等）
- 验证模型类型自动检测功能
- 验证FIM（Fill-in-the-Middle）处理（针对Qwen模型）

### 1.2 性能指标验证
- 编码速度 ≥ 50MB/s
- 内存占用 ≤ 50MB
- 模型加载时间 ≤ 100ms
- 支持并发访问

### 1.3 兼容性验证
- 支持Qwen系列模型（Qwen、Qwen2等）
- 支持DeepSeek系列模型（DeepSeek-LLM、DeepSeek-Coder、DeepSeek3等）
- 支持Llama系列模型
- 向后兼容现有SentencePiece模型

## 2. 测试策略

### 2.1 测试层级
- **单元测试**：验证各个组件的独立功能
- **集成测试**：验证组件间的协作
- **性能测试**：验证性能指标是否达标
- **压力测试**：验证在极端条件下的稳定性
- **回归测试**：防止引入新的bug

### 2.2 测试方法
- **黑盒测试**：验证接口功能
- **白盒测试**：验证内部逻辑实现
- **边界测试**：验证边界条件处理
- **异常测试**：验证错误处理机制

## 3. 单元测试设计

### 3.1 CTokenizer接口测试

#### 3.1.1 基础功能测试
```cpp
TEST(CTokenizerTest, EncodeDecodeBasic) {
    // 测试基本的编码解码功能
    std::unique_ptr<CTokenizer> tokenizer = std::make_unique<SentencePieceTokenizer>(ModelType::QWEN);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    std::string text = "Hello, world!";
    auto tokens = tokenizer->encode(text);
    ASSERT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer->decode(tokens);
    EXPECT_EQ(decoded, text);
}

TEST(CTokenizerTest, VocabOperations) {
    // 测试词汇表操作
    std::unique_ptr<CTokenizer> tokenizer = std::make_unique<SentencePieceTokenizer>(ModelType::QWEN);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    int vocabSize = tokenizer->getVocabSize();
    EXPECT_GT(vocabSize, 0);
    
    // 测试ID到Token的转换
    std::string token = tokenizer->idToToken(100); // 假设ID 100存在
    EXPECT_FALSE(token.empty());
    
    // 测试Token到ID的转换
    llama_token id = tokenizer->tokenToId(token);
    EXPECT_EQ(token, tokenizer->idToToken(id));
}
```

#### 3.1.2 特殊Token处理测试
```cpp
TEST(CTokenizerTest, SpecialTokens) {
    std::unique_ptr<CTokenizer> tokenizer = std::make_unique<SentencePieceTokenizer>(ModelType::QWEN);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    // 测试特殊Token
    llama_token bosId = tokenizer->getBosId();
    llama_token eosId = tokenizer->getEosId();
    llama_token padId = tokenizer->getPadId();
    
    EXPECT_GT(bosId, 0);
    EXPECT_GT(eosId, 0);
    EXPECT_GE(padId, 0); // padId可能为-1（未设置）
    
    // 测试带特殊Token的编码
    std::string text = "Hello";
    auto idsWithoutSpecial = tokenizer->encode(text, false);
    auto idsWithSpecial = tokenizer->encode(text, true);
    
    // 带特殊Token的序列应该更长
    EXPECT_GE(idsWithSpecial.size(), idsWithoutSpecial.size());
}
```

#### 3.1.3 边界条件测试
```cpp
TEST(CTokenizerTest, BoundaryConditions) {
    std::unique_ptr<CTokenizer> tokenizer = std::make_unique<SentencePieceTokenizer>(ModelType::QWEN);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    // 空字符串测试
    auto emptyIds = tokenizer->encode("");
    EXPECT_TRUE(emptyIds.empty() || emptyIds.size() == 2); // 可能包含BOS/EOS
    
    // 单字符测试
    auto singleCharIds = tokenizer->encode("A");
    EXPECT_FALSE(singleCharIds.empty());
    
    std::string singleDecoded = tokenizer->decode(singleCharIds);
    EXPECT_EQ(singleDecoded, "A");
    
    // 特殊字符测试
    std::string specialText = "Hello, 世界! 🌍";
    auto specialIds = tokenizer->encode(specialText);
    ASSERT_FALSE(specialIds.empty());
    
    std::string specialDecoded = tokenizer->decode(specialIds);
    EXPECT_EQ(specialDecoded, specialText);
}
```

### 3.2 QwenTokenizer测试

#### 3.2.1 FIM处理测试
```cpp
TEST(QwenTokenizerTest, FimProcessing) {
    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load("test_models/qwen/tokenizer.model"));
    
    // 测试FIM处理
    std::string text = "<|fim_pre|>def hello():<|fim_suf|>    return 'world'<|fim_end|>";
    auto ids = tokenizer.encode(text);
    ASSERT_FALSE(ids.empty());
    
    std::string decoded = tokenizer.decode(ids);
    EXPECT_EQ(decoded, text);
}

TEST(QwenTokenizerTest, FimDetection) {
    QwenTokenizer tokenizer;
    
    // 测试FIM标记检测
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_begin|>test<|fim_end|>"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("test `` code ``"));
    EXPECT_FALSE(tokenizer.needsFimProcessing("regular text"));
}
```

#### 3.2.2 Qwen特定功能测试
```cpp
TEST(QwenTokenizerTest, QwenSpecificFeatures) {
    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load("test_models/qwen/tokenizer.model"));
    
    // 测试Qwen特有的预处理
    std::string code = "def function():\n    pass";
    auto tokens = tokenizer.encode(code);
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode(tokens);
    EXPECT_EQ(decoded, code);
}
```

### 3.3 DeepSeekTokenizer测试

#### 3.3.1 预处理测试
```cpp
TEST(DeepSeekTokenizerTest, Preprocessing) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    ASSERT_TRUE(tokenizer.load("test_models/deepseek-coder/tokenizer.model"));
    
    std::string code = "class MyClass:\n    def method(self):\n        return True";
    auto tokens = tokenizer.encode(code);
    ASSERT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode(tokens);
    EXPECT_EQ(decoded, code);
}

TEST(DeepSeekTokenizerTest, ModelTypeSpecific) {
    // 测试不同DeepSeek模型类型的处理
    {
        DeepSeekTokenizer llmTokenizer(ModelType::DEEPSEEK_LLM);
        ASSERT_TRUE(llmTokenizer.load("test_models/deepseek-llm/tokenizer.model"));
        std::string text = "Hello world";
        auto tokens = llmTokenizer.encode(text);
        EXPECT_FALSE(tokens.empty());
    }
    
    {
        DeepSeekTokenizer coderTokenizer(ModelType::DEEPSEEK_CODER);
        ASSERT_TRUE(coderTokenizer.load("test_models/deepseek-coder/tokenizer.model"));
        std::string code = "def hello(): pass";
        auto tokens = coderTokenizer.encode(code);
        EXPECT_FALSE(tokens.empty());
    }
}
```

### 3.4 ModelDetector测试

#### 3.4.1 模型类型检测测试
```cpp
TEST(ModelDetectorTest, AutoDetection) {
    // 测试模型自动检测功能
    ModelType type = ModelDetector::detectModelType("test_models/qwen/config.json");
    EXPECT_EQ(type, ModelType::QWEN);
    
    type = ModelDetector::detectModelType("test_models/deepseek-coder/config.json");
    EXPECT_EQ(type, ModelType::DEEPSEEK_CODER);
    
    type = ModelDetector::detectModelType("test_models/deepseek-llm/config.json");
    EXPECT_EQ(type, ModelType::DEEPSEEK_LLM);
}

TEST(ModelDetectorTest, InvalidConfig) {
    // 测试无效配置文件的处理
    ModelType type = ModelDetector::detectModelType("nonexistent/config.json");
    EXPECT_EQ(type, ModelType::SPM); // 应该返回默认类型
}
```

### 3.5 TokenizerManager测试

#### 3.5.1 分词器获取测试
```cpp
TEST(TokenizerManagerTest, GetTokenizer) {
    TokenizerManager manager;
    
    auto qwenTokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(qwenTokenizer, nullptr);
    EXPECT_EQ(qwenTokenizer->getModelType(), ModelType::QWEN);
    
    auto deepseekTokenizer = manager.getTokenizer("deepseek-coder");
    ASSERT_NE(deepseekTokenizer, nullptr);
    EXPECT_EQ(deepseekTokenizer->getModelType(), ModelType::DEEPSEEK_CODER);
    
    auto llamaTokenizer = manager.getTokenizer("llama");
    ASSERT_NE(llamaTokenizer, nullptr);
    EXPECT_EQ(llamaTokenizer->getModelType(), ModelType::LLAMA);
}

TEST(TokenizerManagerTest, CacheBehavior) {
    // 测试分词器缓存行为
    TokenizerManager manager;
    
    auto tokenizer1 = manager.getTokenizer("qwen");
    auto tokenizer2 = manager.getTokenizer("qwen");
    
    // 应该返回同一个实例（如果是单例实现）
    EXPECT_EQ(tokenizer1, tokenizer2);
}
```

## 4. 集成测试设计

### 4.1 端到端测试
```cpp
TEST(IntegrationTest, EndToEnd) {
    // 模拟完整的工作流
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::string input = "This is a test sentence for end-to-end validation.";
    auto tokens = tokenizer->encode(input);
    ASSERT_FALSE(tokens.empty());
    
    std::string output = tokenizer->decode(tokens);
    EXPECT_EQ(input, output);
    
    // 验证词汇表大小的一致性
    int vocabSize = tokenizer->getVocabSize();
    EXPECT_GT(vocabSize, 1000); // 合理的词汇表大小
}

TEST(IntegrationTest, MultiModelSupport) {
    TokenizerManager manager;
    
    // 测试不同模型类型的分词器
    std::vector<std::string> modelTypes = {"qwen", "deepseek-llm", "deepseek-coder", "llama"};
    
    for (const auto& modelType : modelTypes) {
        auto tokenizer = manager.getTokenizer(modelType);
        ASSERT_NE(tokenizer, nullptr) << "Failed to get tokenizer for " << modelType;
        
        std::string testText = "Test text for " + modelType;
        auto tokens = tokenizer->encode(testText);
        ASSERT_FALSE(tokens.empty()) << "Encoding failed for " << modelType;
        
        std::string decoded = tokenizer->decode(tokens);
        EXPECT_EQ(decoded, testText) << "Decoding mismatch for " << modelType;
    }
}
```

### 4.2 批处理测试
```cpp
TEST(BatchTokenizerTest, BatchEncodeDecode) {
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::vector<std::string> texts = {
        "Hello, world!",
        "This is a test sentence.",
        "Another test with numbers: 12345",
        "Mixed content: Hello 世界 🌍"
    };
    
    // 批量编码测试
    for (const auto& text : texts) {
        auto tokens = tokenizer->encode(text);
        ASSERT_FALSE(tokens.empty());
        
        std::string decoded = tokenizer->decode(tokens);
        EXPECT_EQ(decoded, text);
    }
}
```

## 5. 性能测试设计

### 5.1 编码速度测试
```cpp
TEST(PerformanceTest, EncodeSpeed) {
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    std::string longText;
    for (int i = 0; i < 1000; ++i) {
        longText += "This is a test sentence for performance evaluation. ";
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto tokens = tokenizer->encode(longText);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 1000); // 应该在1秒内完成
    
    EXPECT_FALSE(tokens.empty());
    
    // 计算编码速度 (字符/秒)
    double speed = (double)longText.length() / (duration.count() / 1000.0);
    EXPECT_GT(speed, 50000); // 至少50KB/s
}

TEST(PerformanceTest, DecodeSpeed) {
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    std::string text = "Performance test text. ";
    auto tokens = tokenizer->encode(text);
    
    // 重复多次以获得更好的测量结果
    std::vector<std::vector<llama_token>> batchTokens;
    for (int i = 0; i < 1000; ++i) {
        batchTokens.push_back(tokens);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& tokenList : batchTokens) {
        std::string decoded = tokenizer->decode(tokenList);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 1000); // 应该在1秒内完成
}

TEST(PerformanceTest, MemoryUsage) {
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    // 检查初始内存使用
    size_t initialMemory = getCurrentMemoryUsage();
    
    // 执行多次编码/解码操作
    for (int i = 0; i < 10000; ++i) {
        std::string text = "Test " + std::to_string(i);
        auto tokens = tokenizer->encode(text);
        std::string decoded = tokenizer->decode(tokens);
    }
    
    size_t finalMemory = getCurrentMemoryUsage();
    
    // 内存增长不应超过阈值（例如10MB）
    EXPECT_LT(finalMemory - initialMemory, 10 * 1024 * 1024);
}
```

## 6. 验证测试设计

### 6.1 精度验证
```cpp
TEST(ValidationTest, CrossPlatformConsistency) {
    // 验证在不同平台上产生的结果一致性
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    std::vector<std::string> testCases = {
        "Hello, world!",
        "测试中文分词",
        "Test with numbers: 123456789",
        "Mixed: Hello 世界 🌍 emoji",
        "Special chars: !@#$%^&*()",
        "Long text with multiple sentences. This is sentence two. And this is three."
    };
    
    for (const auto& testCase : testCases) {
        auto tokens = tokenizer->encode(testCase);
        std::string decoded = tokenizer->decode(tokens);
        
        EXPECT_EQ(testCase, decoded) << "Mismatch for test case: " << testCase;
        
        // 验证词汇表大小的一致性
        int vocabSize = tokenizer->getVocabSize();
        EXPECT_GT(vocabSize, 0);
    }
}

TEST(ValidationTest, ModelSpecificFeatures) {
    // 验证特定模型的特征
    {
        // Qwen模型特有功能测试
        QwenTokenizer qwenTokenizer;
        ASSERT_TRUE(qwenTokenizer.load("test_models/qwen/tokenizer.model"));
        
        // 测试Qwen特有的FIM处理
        std::string code = "def function():\n    pass";
        auto tokens = qwenTokenizer.encode(code);
        EXPECT_FALSE(tokens.empty());
        
        std::string decoded = qwenTokenizer.decode(tokens);
        EXPECT_EQ(decoded, code);
    }
    
    {
        // DeepSeek模型特有功能测试
        DeepSeekTokenizer deepseekTokenizer(ModelType::DEEPSEEK_CODER);
        ASSERT_TRUE(deepseekTokenizer.load("test_models/deepseek-coder/tokenizer.model"));
        
        // 测试DeepSeek特定的预处理
        std::string code = "class MyClass:\n    def method(self):\n        return True";
        auto tokens = deepseekTokenizer.encode(code);
        EXPECT_FALSE(tokens.empty());
        
        std::string decoded = deepseekTokenizer.decode(tokens);
        EXPECT_EQ(decoded, code);
    }
}
```

### 6.2 回归测试
```cpp
TEST(RegressionTest, KnownIssues) {
    // 针对已知问题的回归测试
    TokenizerManager manager;
    auto tokenizer = manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    ASSERT_TRUE(tokenizer->load("test_models/qwen/tokenizer.model"));
    
    // 测试可能导致问题的特定输入
    std::vector<std::string> problematicInputs = {
        "", // 空字符串
        " ", // 单空格
        "\n", // 单换行
        "\t", // 单制表符
        "\r\n", // Windows换行
        std::string(1000, 'A'), // 长重复字符串
        "A" + std::string(1000, 'B') + "C", // 长中间字符串
        "!@#$%^&*()_+-=[]{}|;:,.<>?", // 所有特殊字符
        "αβγδεζηθικλμνξοπρστυφχψω", // 希腊字母
        "あいうえおかきくけこ", // 日文平假名
        "한국어 테스트", // 韩文
        "العربية", // 阿拉伯文
        "Русский", // 俄文
        " 🌍 ✨ 🚀 " // 表情符号
    };
    
    for (const auto& input : problematicInputs) {
        try {
            auto tokens = tokenizer->encode(input);
            std::string decoded = tokenizer->decode(tokens);
            
            // 对于大多数输入，编码后再解码应该得到相同的结果
            EXPECT_EQ(decoded, input) << "Regression detected for input: " << input;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Exception thrown for input '" << input << "': " << e.what();
        }
    }
}
```

## 7. 测试数据准备

### 7.1 测试模型文件
- `test_models/qwen/` - Qwen模型测试文件
- `test_models/deepseek-coder/` - DeepSeek Coder模型测试文件
- `test_models/deepseek-llm/` - DeepSeek LLM模型测试文件
- `test_models/llama/` - Llama模型测试文件

### 7.2 测试配置文件
- `tokenizer.model` - SentencePiece模型文件
- `config.json` - 模型配置文件
- `tokenizer.json` - 分词器配置文件

## 8. 测试执行策略

### 8.1 测试覆盖率
- 功能覆盖率: 确保所有功能都经过测试
- 代码覆盖率: 目标达到85%以上
- 数据覆盖率: 涵盖各种输入类型和边界条件

### 8.2 自动化测试
```bash
# 完整测试套件执行
./bin/ctokenizer_tests --gtest_filter=* --verbose

# 性能测试
./bin/ctokenizer_benchmark --model=qwen --text=performance_test.txt --iterations=1000

# 生成覆盖率报告
gcovr --html --html-details -o coverage.html
```

### 8.3 测试环境
- Linux/Windows/macOS 多平台支持
- 不同模型格式测试
- 多线程并发测试

## 9. 预期结果

### 9.1 功能验证
- 所有编码解码功能正常工作
- 特殊Token处理正确
- 模型类型检测准确
- FIM处理功能正常

### 9.2 性能验证
- 编码速度满足 ≥ 50MB/s 要求
- 内存占用满足 ≤ 50MB 要求
- 模型加载时间满足 ≤ 100ms 要求

### 9.3 兼容性验证
- 支持所有目标模型类型
- 与现有系统兼容
- 向后兼容性保持

## 10. 风险与缓解

### 10.1 潜在风险
- 模型文件加载失败
- 内存泄漏
- 并发访问问题
- 性能不达标

### 10.2 缓解措施
- 完善的错误处理机制
- 严格的内存管理
- 线程安全测试
- 性能监控和优化