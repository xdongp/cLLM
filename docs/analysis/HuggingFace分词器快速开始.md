# HuggingFace Tokenizer快速上手指南

> **目标读者**: cLLM开发者  
> **预计时间**: 30分钟完成环境配置和第一个示例  
> **前置条件**: macOS/Linux + CMake + C++17

---

## 🚀 快速开始 (5分钟)

### Step 1: 安装tokenizers-cpp

#### macOS
```bash
# 安装Rust (如果没有)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 编译安装tokenizers-cpp
git clone https://github.com/mlc-ai/tokenizers-cpp /tmp/tokenizers-cpp
cd /tmp/tokenizers-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/homebrew
make -j8
sudo make install

# 验证安装
ls /opt/homebrew/include/tokenizers_cpp.h
ls /opt/homebrew/lib/libtokenizers_cpp.dylib
```

#### Linux (Ubuntu/Debian)
```bash
# 安装依赖
sudo apt-get update
sudo apt-get install -y cargo rustc cmake g++ git

# 编译安装
git clone https://github.com/mlc-ai/tokenizers-cpp /tmp/tokenizers-cpp
cd /tmp/tokenizers-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install

# 验证
pkg-config --cflags --libs tokenizers_cpp
```

### Step 2: 编译cLLM (启用HF支持)

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
mkdir -p build && cd build

# ✅ 启用tokenizers-cpp支持
cmake .. -DUSE_TOKENIZERS_CPP=ON

# 编译
make -j8

# 验证HF Tokenizer已启用
grep "USE_TOKENIZERS_CPP" CMakeCache.txt
# 应输出: USE_TOKENIZERS_CPP:BOOL=ON
```

### Step 3: 测试运行

```bash
# 运行HTTP Server测试 (使用HF Tokenizer)
cd build
./bin/test_http_server_direct

# 预期输出:
# [INFO] Detected HuggingFace format (tokenizer.json)
# [INFO] HFTokenizer loaded successfully
# [INFO] Vocab size: 151936, BOS: 151643, EOS: 151645
# [PASS] All tests passed!
```

---

## 📝 代码示例

### 示例1: 基础使用 (自动检测格式)

```cpp
#include "cllm/tokenizer/base_tokenizer.h"

int main() {
    // ✅ 自动检测tokenizer.json → 使用HFTokenizer
    auto tokenizer = cllm::TokenizerFactory::create(
        "/path/to/Qwen3-0.6B",
        cllm::TokenizerFactory::Backend::AUTO
    );
    
    // 编码
    std::string text = "Hello, world!";
    auto ids = tokenizer->encode(text, true);
    
    // 输出Token IDs
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
    
    // 解码
    std::string decoded = tokenizer->decode(ids, true);
    std::cout << "Decoded: \"" << decoded << "\"" << std::endl;
    
    return 0;
}
```

**编译运行**:
```bash
g++ -std=c++17 example.cpp -o example \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lcllm_core -ltokenizers_cpp

./example
# 输出:
# Token IDs: [151643, 9707, 11, 1879, 0, 151645]
# Decoded: "Hello, world!"
```

### 示例2: 强制使用HF Tokenizer

```cpp
#include "cllm/tokenizer/hf_tokenizer.h"

int main() {
    // ✅ 强制使用HuggingFace格式
    auto tokenizer = cllm::TokenizerFactory::create(
        "/path/to/model",
        cllm::TokenizerFactory::Backend::HUGGINGFACE  // 显式指定
    );
    
    // 获取词表信息
    std::cout << "Vocab size: " << tokenizer->getVocabSize() << std::endl;
    std::cout << "BOS ID: " << tokenizer->getBosId() << std::endl;
    std::cout << "EOS ID: " << tokenizer->getEosId() << std::endl;
    
    // Token ID → 字符串
    std::cout << "BOS Token: " << tokenizer->idToToken(tokenizer->getBosId()) << std::endl;
    
    return 0;
}
```

### 示例3: Chat Template支持

```cpp
#include "cllm/tokenizer/hf_tokenizer.h"

int main() {
    auto hfTokenizer = dynamic_cast<cllm::HFTokenizer*>(
        cllm::TokenizerFactory::create(
            "/path/to/Qwen3-0.6B",
            cllm::TokenizerFactory::Backend::HUGGINGFACE
        ).get()
    );
    
    // 构造聊天消息
    std::vector<cllm::ChatMessage> messages = {
        {"system", "You are a helpful assistant"},
        {"user", "What is the capital of France?"}
    };
    
    // 应用Chat Template并编码
    auto ids = hfTokenizer->applyChatTemplate(messages, true);
    
    std::cout << "Tokenized chat: " << ids.size() << " tokens" << std::endl;
    
    return 0;
}
```

### 示例4: 批量处理

```cpp
#include "cllm/tokenizer/base_tokenizer.h"
#include <vector>
#include <chrono>

int main() {
    auto tokenizer = cllm::TokenizerFactory::create("/path/to/model");
    
    // 准备批量文本
    std::vector<std::string> texts = {
        "Hello, world!",
        "How are you today?",
        "This is a test sentence.",
        "Machine learning is amazing!"
    };
    
    // 批量编码 (并行处理)
    auto start = std::chrono::high_resolution_clock::now();
    auto results = tokenizer->batchEncode(texts, true);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Encoded " << texts.size() << " texts in " 
              << duration.count() << " ms" << std::endl;
    
    // 批量解码
    auto decodedTexts = tokenizer->batchDecode(results, true);
    
    for (size_t i = 0; i < texts.size(); ++i) {
        std::cout << "Original: \"" << texts[i] << "\"" << std::endl;
        std::cout << "Decoded:  \"" << decodedTexts[i] << "\"" << std::endl;
        std::cout << "Tokens:   " << results[i].size() << std::endl << std::endl;
    }
    
    return 0;
}
```

### 示例5: 增量解码 (流式生成)

```cpp
#include "cllm/tokenizer/hf_tokenizer.h"
#include <iostream>

int main() {
    auto hfTokenizer = dynamic_cast<cllm::HFTokenizer*>(
        cllm::TokenizerFactory::create("/path/to/model").get()
    );
    
    // 创建增量解码器
    auto decoder = hfTokenizer->createIncrementalDecoder();
    
    // 模拟流式生成
    std::vector<cllm::token_id_t> generatedTokens = {
        151643,  // BOS
        9707,    // "Hello"
        11,      // ","
        1879,    // " world"
        0,       // "!"
        151645   // EOS
    };
    
    std::cout << "Streaming output: ";
    for (auto tokenId : generatedTokens) {
        std::string chunk = decoder->add(tokenId);
        if (!chunk.empty()) {
            std::cout << chunk << std::flush;  // 实时输出
        }
    }
    
    // 完成解码
    std::cout << decoder->finish() << std::endl;
    
    return 0;
}
```

---

## 🔧 配置选项

### CMake选项

```cmake
# 启用HuggingFace tokenizers支持
option(USE_TOKENIZERS_CPP "Use tokenizers-cpp" ON)

# 强制使用SentencePiece (应急回滚)
option(FORCE_SENTENCEPIECE "Force SentencePiece backend" OFF)
```

### 运行时配置

#### 环境变量
```bash
# 强制使用特定backend
export CLLM_TOKENIZER_BACKEND=huggingface  # 或 sentencepiece

# 调试模式
export CLLM_LOG_LEVEL=DEBUG
```

#### 配置文件 (config/tokenizer.yaml)
```yaml
tokenizer:
  # backend选择: auto | huggingface | sentencepiece | native
  backend: auto
  
  # 模型路径
  model_path: /path/to/model
  
  # 缓存配置
  cache:
    enabled: true
    max_size: 10000  # LRU缓存大小
    
  # 性能配置
  performance:
    enable_metrics: true
    batch_size: 32
```

---

## 🐛 故障排查

### 问题1: tokenizers-cpp找不到

**错误**:
```
CMake Error: Could not find tokenizers_cpp
```

**解决**:
```bash
# 检查安装位置
ls /opt/homebrew/include/tokenizers_cpp.h
ls /usr/local/include/tokenizers_cpp.h

# 如果没有,重新安装
cd /tmp/tokenizers-cpp
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/homebrew
make -j8 && sudo make install

# 清理CMake缓存
cd /path/to/cLLM/build
rm CMakeCache.txt
cmake .. -DUSE_TOKENIZERS_CPP=ON
```

### 问题2: tokenizer.json找不到

**错误**:
```
[ERROR] tokenizer.json not found: /path/to/model/tokenizer.json
```

**解决**:
```bash
# 检查文件是否存在
ls -la /path/to/model/tokenizer.json

# 如果不存在,下载正确的模型
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /path/to/model

# 或从本地模型复制
cp /path/to/correct/tokenizer.json /path/to/model/
```

### 问题3: 编码结果为空

**症状**:
```cpp
auto ids = tokenizer->encode("Hello");
// ids.size() == 0  (错误)
```

**诊断**:
```cpp
// 检查tokenizer是否加载成功
if (tokenizer->getVocabSize() == 0) {
    std::cerr << "Tokenizer not loaded properly!" << std::endl;
}

// 检查特殊Token
std::cout << "BOS: " << tokenizer->getBosId() << std::endl;
std::cout << "EOS: " << tokenizer->getEosId() << std::endl;
```

**解决**:
```cpp
// 确保正确加载
auto tokenizer = cllm::TokenizerFactory::create(modelPath);
if (!tokenizer) {
    throw std::runtime_error("Failed to create tokenizer");
}

// 验证功能
std::string testText = "test";
auto ids = tokenizer->encode(testText, false);  // 不添加特殊Token
if (ids.empty()) {
    std::cerr << "Encode failed!" << std::endl;
}
```

### 问题4: 性能不达预期

**症状**: 编码速度 < 50 MB/s

**优化措施**:
```cpp
// 1. 启用缓存
tokenizer->enableCache(true);

// 2. 使用批处理
std::vector<std::string> texts = {...};
auto results = tokenizer->batchEncode(texts, true);  // 并行处理

// 3. 预热tokenizer
tokenizer->encode("warmup", true);  // 第一次调用较慢

// 4. 检查编译优化
// CMakeLists.txt中确保:
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
```

---

## 📊 性能基准

### 预期性能指标

| 操作 | 文本长度 | 预期速度 | 实测速度 |
|------|---------|---------|---------|
| 编码 (英文) | 100字符 | >100 MB/s | _待测_ |
| 编码 (中文) | 100字符 | >80 MB/s | _待测_ |
| 解码 | 100 tokens | >50 MB/s | _待测_ |
| 批处理 (x64) | 100字符/个 | >200 MB/s | _待测_ |

### 基准测试命令

```bash
cd build

# 运行基准测试
./bin/benchmark_tokenizers --benchmark_filter=HFTokenizer

# 输出示例:
# BM_HFTokenizer_Encode            1000000 ns/op
# BM_HFTokenizer_Decode            2000000 ns/op
# BM_HFTokenizer_BatchEncode/64     500000 ns/op
```

---

## 🔄 从SentencePiece迁移

### 迁移清单

#### 代码层面

**旧代码** (SentencePiece):
```cpp
#include "cllm/tokenizer/tokenizer.h"

auto tokenizer = std::make_unique<cllm::Tokenizer>(modelPath);
tokenizer->loadModel(modelPath + "/tokenizer.model");
auto ids = tokenizer->encode(text, true);
```

**新代码** (统一接口):
```cpp
#include "cllm/tokenizer/base_tokenizer.h"

// ✅ 自动选择最佳backend (HF或SentencePiece)
auto tokenizer = cllm::TokenizerFactory::create(modelPath);
auto ids = tokenizer->encode(text, true);
```

#### 模型文件

**检查模型目录**:
```bash
ls /path/to/model/
# 如果有tokenizer.json → 自动使用HF
# 如果有tokenizer.model → 自动使用SentencePiece
```

**转换工具** (如果需要):
```python
# 从SentencePiece转换为HF格式
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/old/model")
tokenizer.save_pretrained("/path/to/new/model")
# 会生成tokenizer.json
```

#### 配置文件

**旧配置**:
```yaml
tokenizer:
  type: sentencepiece
  model_file: tokenizer.model
```

**新配置**:
```yaml
tokenizer:
  backend: auto  # 自动检测
  model_path: /path/to/model
```

---

## 📚 API参考

### BaseTokenizer核心接口

```cpp
class BaseTokenizer {
public:
    // 加载模型
    virtual bool load(const std::string& modelPath) = 0;
    
    // 编码/解码
    virtual TokenSequence encode(const std::string& text, bool addSpecialTokens = true) = 0;
    virtual std::string decode(const TokenSequence& ids, bool skipSpecialTokens = true) = 0;
    
    // 批处理
    virtual std::vector<TokenSequence> batchEncode(
        const std::vector<std::string>& texts, 
        bool addSpecialTokens = true
    );
    
    virtual std::vector<std::string> batchDecode(
        const std::vector<TokenSequence>& sequences,
        bool skipSpecialTokens = true
    );
    
    // 信息查询
    virtual int getVocabSize() const = 0;
    virtual token_id_t getBosId() const;
    virtual token_id_t getEosId() const;
    virtual token_id_t getPadId() const;
    virtual token_id_t getUnkId() const;
    
    // Token转换
    virtual std::string idToToken(token_id_t id) const = 0;
    virtual token_id_t tokenToId(const std::string& token) const = 0;
    
    // 工具方法
    virtual bool isSpecialToken(token_id_t id) const;
    virtual std::vector<std::string> tokenize(const std::string& text);
};
```

### TokenizerFactory工厂方法

```cpp
class TokenizerFactory {
public:
    enum class Backend {
        AUTO,           // 自动检测 (推荐)
        HUGGINGFACE,    // 强制HF
        SENTENCEPIECE,  // 强制SentencePiece
        NATIVE          // 自研实现
    };
    
    // 创建tokenizer
    static std::unique_ptr<BaseTokenizer> create(
        const std::string& modelPath,
        Backend backend = Backend::AUTO,
        ModelType modelType = ModelType::AUTO
    );
};
```

### HFTokenizer扩展功能

```cpp
class HFTokenizer : public BaseTokenizer {
public:
    // Chat Template
    TokenSequence applyChatTemplate(
        const std::vector<ChatMessage>& messages,
        bool addGenerationPrompt = false
    );
    
    // 增量解码
    class IncrementalDecoder {
    public:
        std::string add(token_id_t tokenId);
        std::string finish();
        void reset();
    };
    
    std::unique_ptr<IncrementalDecoder> createIncrementalDecoder();
    
    // 分词 (返回字符串列表)
    std::vector<std::string> tokenize(const std::string& text);
    
    // 特殊Token判断
    bool isSpecialToken(token_id_t tokenId) const;
};
```

---

## 🎓 进阶主题

### 自定义预处理

```cpp
class MyTokenizer : public cllm::HFTokenizer {
public:
    TokenSequence encode(const std::string& text, bool addSpecialTokens) override {
        // 自定义预处理
        std::string processed = preprocessText(text);
        
        // 调用父类方法
        return HFTokenizer::encode(processed, addSpecialTokens);
    }
    
private:
    std::string preprocessText(const std::string& text) {
        // 例: 移除特殊字符
        std::string result = text;
        // ... 自定义逻辑
        return result;
    }
};
```

### 性能监控

```cpp
auto tokenizer = cllm::TokenizerFactory::create(modelPath);

// 编码多次
for (int i = 0; i < 1000; ++i) {
    tokenizer->encode("test text", true);
}

// 获取性能指标
auto metrics = tokenizer->getMetrics();
std::cout << "Avg encode time: " << metrics.avgEncodeTime() << " ms" << std::endl;
std::cout << "Total encodes: " << metrics.encodeCount << std::endl;
```

### 多Tokenizer共存

```cpp
// 同时使用HF和SentencePiece
auto hfTokenizer = cllm::TokenizerFactory::create(
    "/path/to/qwen",
    cllm::TokenizerFactory::Backend::HUGGINGFACE
);

auto spTokenizer = cllm::TokenizerFactory::create(
    "/path/to/llama",
    cllm::TokenizerFactory::Backend::SENTENCEPIECE
);

// 根据任务选择
auto ids1 = hfTokenizer->encode("Modern model text", true);
auto ids2 = spTokenizer->encode("Legacy model text", true);
```

---

## 💡 最佳实践

### ✅ 推荐做法

1. **使用AUTO backend**
   ```cpp
   // ✅ 推荐: 自动检测
   auto tokenizer = TokenizerFactory::create(modelPath);
   ```

2. **启用缓存**
   ```cpp
   // 对于重复文本,启用缓存可提升10倍性能
   tokenizer->enableCache(true);
   ```

3. **批量处理**
   ```cpp
   // ✅ 推荐: 批量处理 (并行)
   auto results = tokenizer->batchEncode(texts, true);
   
   // ❌ 避免: 逐个处理
   for (const auto& text : texts) {
       auto ids = tokenizer->encode(text, true);  // 串行,慢
   }
   ```

4. **异常处理**
   ```cpp
   try {
       auto tokenizer = TokenizerFactory::create(modelPath);
       auto ids = tokenizer->encode(text, true);
   } catch (const std::exception& e) {
       std::cerr << "Error: " << e.what() << std::endl;
       // 回退或错误处理
   }
   ```

### ❌ 避免的做法

1. **硬编码backend**
   ```cpp
   // ❌ 避免: 除非有特殊需求
   auto tokenizer = TokenizerFactory::create(
       modelPath, 
       TokenizerFactory::Backend::SENTENCEPIECE  // 限制兼容性
   );
   ```

2. **频繁创建tokenizer**
   ```cpp
   // ❌ 避免: 每次都创建新实例
   for (int i = 0; i < 1000; ++i) {
       auto tokenizer = TokenizerFactory::create(modelPath);  // 很慢!
       tokenizer->encode(texts[i], true);
   }
   
   // ✅ 推荐: 复用实例
   auto tokenizer = TokenizerFactory::create(modelPath);
   for (int i = 0; i < 1000; ++i) {
       tokenizer->encode(texts[i], true);
   }
   ```

3. **忽略错误处理**
   ```cpp
   // ❌ 避免: 不检查返回值
   auto ids = tokenizer->encode(text, true);
   // 如果ids为空怎么办?
   
   // ✅ 推荐: 检查结果
   auto ids = tokenizer->encode(text, true);
   if (ids.empty()) {
       std::cerr << "Encode failed!" << std::endl;
       return;
   }
   ```

---

## 🆘 获取帮助

### 文档
- 完整技术方案: [hf_tokenizer_migration_strategy.md](./hf_tokenizer_migration_strategy.md)
- 执行摘要: [tokenizer_migration_executive_summary.md](./tokenizer_migration_executive_summary.md)
- API文档: [待生成]

### 社区支持
- GitHub Issues: [提交问题]
- 邮件列表: team@cllm-project.org
- Slack频道: #tokenizer-support

### 常见问题
- FAQ: [docs/FAQ.md](../FAQ.md)
- 故障排查: 见上文"故障排查"章节

---

## ✅ 总结

恭喜!你已经掌握了HuggingFace Tokenizer的基础使用:

1. ✅ 安装tokenizers-cpp依赖
2. ✅ 编译cLLM (启用HF支持)
3. ✅ 运行第一个示例
4. ✅ 理解核心API
5. ✅ 了解最佳实践

**下一步**:
- 尝试加载自己的模型
- 集成到HTTP Server
- 性能优化与调优
- 贡献代码与反馈

**Happy Tokenizing! 🎉**

---

**文档版本**: v1.0  
**最后更新**: 2026-01-11  
**维护者**: cLLM Core Team
