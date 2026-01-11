# 🔌 Tokenizer集成专项规则

> **使用场景**: 集成新的Tokenizer实现或修改现有Tokenizer

---

## 📋 集成检查清单

### 1. 实现ITokenizer接口

```cpp
class NewTokenizer : public ITokenizer {
public:
    // ✅ 必须实现的接口
    bool load(const std::string& modelPath) override;
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override;
    std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) override;
    
    int getVocabSize() const override;
    std::string idToToken(int id) const override;
    int tokenToId(const std::string& token) const override;
    
    int getBosId() const override;
    int getEosId() const override;
    int getPadId() const override;
    int getUnkId() const override;
    
    ModelType getModelType() const override;
};
```

### 2. 条件编译保护

```cpp
// include/cllm/tokenizer/new_tokenizer.h
#pragma once

#include "i_tokenizer.h"

#ifdef USE_NEW_TOKENIZER_LIB
#include <new_tokenizer_lib.h>
#endif

namespace cllm {

class NewTokenizer : public ITokenizer {
public:
    NewTokenizer(ModelType modelType = ModelType::AUTO);
    ~NewTokenizer() override;
    
    bool load(const std::string& modelPath) override;
    // ...

private:
#ifdef USE_NEW_TOKENIZER_LIB
    std::unique_ptr<new_tokenizer::Tokenizer> impl_;
#else
    // 回退实现或抛出异常
#endif
};

} // namespace cllm
```

### 3. 更新CMakeLists.txt

```cmake
# 添加编译选项
option(USE_NEW_TOKENIZER_LIB "Use new tokenizer library" OFF)

if(USE_NEW_TOKENIZER_LIB)
    message(STATUS "✅ Enabling new tokenizer support")
    
    # 查找库
    find_path(NEW_TOKENIZER_INCLUDE_DIR 
        NAMES new_tokenizer.h
        PATHS /usr/local/include /opt/homebrew/include
    )
    
    find_library(NEW_TOKENIZER_LIBRARY 
        NAMES new_tokenizer
        PATHS /usr/local/lib /opt/homebrew/lib
    )
    
    if(NEW_TOKENIZER_INCLUDE_DIR AND NEW_TOKENIZER_LIBRARY)
        message(STATUS "   Include: ${NEW_TOKENIZER_INCLUDE_DIR}")
        message(STATUS "   Library: ${NEW_TOKENIZER_LIBRARY}")
        
        add_compile_definitions(USE_NEW_TOKENIZER_LIB)
        include_directories(${NEW_TOKENIZER_INCLUDE_DIR})
        
        set(NEW_TOKENIZER_LIBRARIES ${NEW_TOKENIZER_LIBRARY})
    else()
        message(WARNING "⚠️  new tokenizer not found")
        set(USE_NEW_TOKENIZER_LIB OFF)
    endif()
endif()

# 添加到链接库
target_link_libraries(cllm_core
    # ... 其他库 ...
    ${NEW_TOKENIZER_LIBRARIES}
)
```

### 4. 更新TokenizerManager检测逻辑

```cpp
// src/tokenizer/manager.cpp

namespace {
    bool hasNewTokenizerFormat(const std::string& modelPath) {
        namespace fs = std::filesystem;
        // 检测特定文件
        return fs::exists(fs::path(modelPath) / "new_tokenizer.json");
    }
}

TokenizerManager::TokenizerManager(...) {
    switch(impl) {
        case TokenizerImpl::AUTO:
            // ✅ 添加到自动检测逻辑
            if (hasNewTokenizerFormat(modelPath)) {
                CLLM_INFO("✅ Detected new tokenizer format");
                tokenizer_ = new NewTokenizer(modelType);
                
            } else if (hasTokenizerJson(modelPath)) {
                CLLM_INFO("✅ Detected HuggingFace format");
                tokenizer_ = new HFTokenizer(modelType);
                
            } else if (hasTokenizerModel(modelPath)) {
                CLLM_INFO("✅ Detected SentencePiece format");
                tokenizer_ = new NativeTokenizer(modelType);
                
            } else {
                CLLM_WARN("⚠️  Unknown format, fallback to Native");
                tokenizer_ = new NativeTokenizer(modelType);
            }
            break;
    }
}
```

### 5. 编写单元测试

```cpp
// tests/test_new_tokenizer.cpp
#include <gtest/gtest.h>
#include "cllm/tokenizer/new_tokenizer.h"

class NewTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        tokenizer_ = std::make_unique<cllm::NewTokenizer>();
        ASSERT_TRUE(tokenizer_->load("path/to/test/model"));
    }
    
    std::unique_ptr<cllm::NewTokenizer> tokenizer_;
};

TEST_F(NewTokenizerTest, EncodeDecodeRoundtrip) {
    std::string text = "Hello, world!";
    auto ids = tokenizer_->encode(text, true);
    auto decoded = tokenizer_->decode(ids, true);
    EXPECT_EQ(text, decoded);
}

TEST_F(NewTokenizerTest, SpecialTokens) {
    EXPECT_GE(tokenizer_->getBosId(), 0);
    EXPECT_GE(tokenizer_->getEosId(), 0);
    EXPECT_GE(tokenizer_->getVocabSize(), 1000);
}

TEST_F(NewTokenizerTest, EmptyText) {
    auto ids = tokenizer_->encode("", false);
    EXPECT_TRUE(ids.empty());
}

TEST_F(NewTokenizerTest, LongText) {
    std::string longText(10000, 'a');
    auto ids = tokenizer_->encode(longText, false);
    EXPECT_GT(ids.size(), 0);
}
```

---

## 🔧 实现最佳实践

### 1. 错误处理

```cpp
bool NewTokenizer::load(const std::string& modelPath) {
#ifdef USE_NEW_TOKENIZER_LIB
    try {
        impl_ = new_tokenizer::Tokenizer::FromFile(modelPath);
        if (!impl_) {
            CLLM_ERROR("Failed to load tokenizer: %s", modelPath.c_str());
            return false;
        }
        
        CLLM_INFO("✅ NewTokenizer loaded: %s", modelPath.c_str());
        return true;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Exception loading tokenizer: %s", e.what());
        return false;
    }
#else
    CLLM_ERROR("NewTokenizer requires USE_NEW_TOKENIZER_LIB=ON");
    return false;
#endif
}
```

### 2. 特殊Token处理

```cpp
void NewTokenizer::loadConfig(const std::string& modelPath) {
    namespace fs = std::filesystem;
    
    std::string configPath = (fs::path(modelPath) / "config.json").string();
    if (!fs::exists(configPath)) return;
    
    std::ifstream f(configPath);
    if (!f.is_open()) return;
    
    try {
        auto config = nlohmann::json::parse(f);
        
        // 读取特殊Token IDs
        if (config.contains("bos_token_id")) {
            bosId_ = config["bos_token_id"].get<int>();
        }
        if (config.contains("eos_token_id")) {
            eosId_ = config["eos_token_id"].get<int>();
        }
        if (config.contains("pad_token_id") && !config["pad_token_id"].is_null()) {
            padId_ = config["pad_token_id"].get<int>();
        }
        if (config.contains("unk_token_id")) {
            unkId_ = config["unk_token_id"].get<int>();
        }
        
        CLLM_INFO("Loaded special tokens: BOS=%d, EOS=%d, PAD=%d, UNK=%d",
                  bosId_, eosId_, padId_, unkId_);
        
    } catch (const std::exception& e) {
        CLLM_WARN("Failed to parse config: %s", e.what());
    }
}
```

### 3. 性能优化

```cpp
// 缓存编码结果
class CachedNewTokenizer : public NewTokenizer {
    std::unordered_map<std::string, std::vector<int>> cache_;
    size_t maxCacheSize_ = 10000;
    
public:
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        // 查缓存
        auto it = cache_.find(text);
        if (it != cache_.end()) {
            return it->second;
        }
        
        // 编码
        auto result = NewTokenizer::encode(text, addSpecialTokens);
        
        // 缓存
        if (cache_.size() < maxCacheSize_) {
            cache_[text] = result;
        }
        
        return result;
    }
};
```

---

## 📊 集成验证步骤

### 1. 编译验证

```bash
cd build && rm -rf *
cmake .. -DUSE_NEW_TOKENIZER_LIB=ON
make -j8

# 检查是否成功链接
ldd bin/cllm_server | grep new_tokenizer
```

### 2. 单元测试

```bash
./bin/test_new_tokenizer
```

### 3. 集成测试

```bash
# 使用测试模型
./bin/test_tokenizer_manager --model=/path/to/test/model

# 检查日志
cat logs/cllm.log | grep "NewTokenizer"
```

### 4. 性能测试

```bash
# 对比不同Tokenizer性能
./bin/benchmark_tokenizer --impl=new
./bin/benchmark_tokenizer --impl=hf
./bin/benchmark_tokenizer --impl=native
```

---

## 🚨 常见问题

### 问题1: 链接失败

```markdown
错误: undefined reference to 'new_tokenizer::Tokenizer::FromFile'

解决:
1. 检查库是否安装: ls /usr/local/lib | grep new_tokenizer
2. 检查CMake是否找到库: cmake .. -DUSE_NEW_TOKENIZER_LIB=ON 查看输出
3. 检查链接顺序: target_link_libraries 中添加库
```

### 问题2: 头文件找不到

```markdown
错误: fatal error: new_tokenizer.h: No such file or directory

解决:
1. 检查头文件路径: find /usr/local/include -name "new_tokenizer.h"
2. 添加include路径: include_directories(...)
3. 检查条件编译: #ifdef USE_NEW_TOKENIZER_LIB
```

### 问题3: 运行时找不到库

```markdown
错误: error while loading shared libraries: libnew_tokenizer.so

解决:
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# 或安装到系统路径
sudo make install
sudo ldconfig  # Linux
```

---

## 📚 参考实现

查看现有Tokenizer实现:

- **HFTokenizer**: `src/tokenizer/hf_tokenizer.cpp`
- **NativeTokenizer**: `src/tokenizer/native_tokenizer.cpp`
- **UnifiedTokenizer**: `src/tokenizer/unified_tokenizer.cpp`

---

**最后更新**: 2026-01-11
