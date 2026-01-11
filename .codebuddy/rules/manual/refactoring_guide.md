# 🔄 代码重构指南

> **触发条件**: 用户提到"重构"、"优化结构"、"解耦"时使用本规则

---

## 🎯 重构原则

### 核心原则

1. **小步快跑** - 每次重构保持小范围
2. **保持测试绿色** - 每步后测试通过
3. **接口兼容** - 尽量不破坏现有接口
4. **可回滚** - 每步都可独立回滚

---

## 📋 重构检查清单

### 开始前

- [ ] 是否有完整的测试覆盖?
- [ ] 是否理解现有设计?
- [ ] 是否制定了重构计划?
- [ ] 是否有性能基准?

### 执行中

- [ ] 每步修改是否足够小?
- [ ] 每步后是否运行测试?
- [ ] 是否保持编译通过?
- [ ] 是否记录了变更?

### 完成后

- [ ] 所有测试是否通过?
- [ ] 性能是否保持/提升?
- [ ] 文档是否更新?
- [ ] 代码是否更清晰?

---

## 🛠️ 常见重构模式

### 1. 提取函数 (Extract Function)

**场景**: 函数过长,职责不清

```cpp
// ❌ 重构前: 长函数
void processRequest(const Request& req) {
    // 验证输入 (20行)
    if (req.text.empty()) return;
    if (req.maxTokens <= 0) return;
    // ...
    
    // 编码 (30行)
    std::vector<int> ids;
    // ...
    
    // 推理 (40行)
    torch::Tensor output;
    // ...
    
    // 解码 (20行)
    std::string result;
    // ...
}

// ✅ 重构后: 小函数
void processRequest(const Request& req) {
    if (!validateRequest(req)) return;
    
    auto ids = encodeText(req.text);
    auto output = runInference(ids, req.maxTokens);
    auto result = decodeOutput(output);
    
    sendResponse(result);
}

private:
    bool validateRequest(const Request& req);
    std::vector<int> encodeText(const std::string& text);
    torch::Tensor runInference(const std::vector<int>& ids, int maxTokens);
    std::string decodeOutput(const torch::Tensor& output);
```

### 2. 提取类 (Extract Class)

**场景**: 类职责过多

```cpp
// ❌ 重构前: 上帝类
class TokenizerManager {
    // Tokenizer相关
    ITokenizer* tokenizer_;
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    
    // 生成相关
    std::string generate(const std::string& prompt, int maxTokens);
    std::vector<GenerationResponse> generateStream(...);
    
    // 统计相关
    TokenizerStats stats_;
    void updateStats(...);
    TokenizerStats getStats();
    
    // 缓存相关
    std::unordered_map<std::string, std::vector<int>> cache_;
    void cacheResult(...);
};

// ✅ 重构后: 职责分离
class TokenizerManager {
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<Generator> generator_;      // 提取生成逻辑
    std::unique_ptr<StatsCollector> stats_;     // 提取统计逻辑
    std::unique_ptr<TokenCache> cache_;         // 提取缓存逻辑
    
public:
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& ids);
    
    std::string generate(const std::string& prompt, int maxTokens) {
        return generator_->generate(prompt, maxTokens);
    }
};

class Generator {
    // 专注于生成逻辑
};

class StatsCollector {
    // 专注于统计收集
};

class TokenCache {
    // 专注于缓存管理
};
```

### 3. 引入接口 (Extract Interface)

**场景**: 需要多种实现,缺乏抽象

```cpp
// ❌ 重构前: 具体类耦合
class ModelExecutor {
    LibTorchBackend* backend_;  // 直接依赖具体实现
    
public:
    torch::Tensor forward(...) {
        return backend_->forward(...);
    }
};

// ✅ 重构后: 依赖抽象
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    virtual torch::Tensor forward(...) = 0;
};

class LibTorchBackend : public IInferenceBackend {
    torch::Tensor forward(...) override;
};

class KylinBackend : public IInferenceBackend {
    torch::Tensor forward(...) override;
};

class ModelExecutor {
    std::unique_ptr<IInferenceBackend> backend_;  // 依赖接口
    
public:
    torch::Tensor forward(...) {
        return backend_->forward(...);
    }
};
```

### 4. 用组合替代继承 (Replace Inheritance with Composition)

**场景**: 继承层次过深,不灵活

```cpp
// ❌ 重构前: 深层继承
class BaseTokenizer { /* ... */ };
class CachedTokenizer : public BaseTokenizer { /* ... */ };
class StatefulCachedTokenizer : public CachedTokenizer { /* ... */ };

// ✅ 重构后: 组合
class Tokenizer {
    std::unique_ptr<ITokenizer> impl_;
    std::unique_ptr<TokenCache> cache_;    // 可选组件
    std::unique_ptr<StateManager> state_;  // 可选组件
    
public:
    std::vector<int> encode(const std::string& text) {
        // 先查缓存
        if (cache_) {
            if (auto cached = cache_->get(text)) {
                return *cached;
            }
        }
        
        // 执行编码
        auto result = impl_->encode(text);
        
        // 更新状态
        if (state_) {
            state_->update(result);
        }
        
        // 缓存结果
        if (cache_) {
            cache_->put(text, result);
        }
        
        return result;
    }
};
```

### 5. 引入参数对象 (Introduce Parameter Object)

**场景**: 参数过多

```cpp
// ❌ 重构前: 参数过多
std::string generate(
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP,
    float topK,
    float repetitionPenalty,
    int numBeams,
    bool doSample,
    int seed
);

// ✅ 重构后: 参数对象
struct GenerationConfig {
    std::string prompt;
    int maxTokens = 100;
    float temperature = 1.0f;
    float topP = 0.9f;
    float topK = 50.0f;
    float repetitionPenalty = 1.0f;
    int numBeams = 1;
    bool doSample = true;
    int seed = -1;
};

std::string generate(const GenerationConfig& config);
```

---

## 🔧 重构步骤示例

### 示例: 重构TokenizerManager

**目标**: 将生成逻辑提取到单独的Generator类

#### Step 1: 创建新接口

```cpp
// 新建 include/cllm/tokenizer/generator.h
namespace cllm {

class Generator {
public:
    Generator(ITokenizer* tokenizer, ModelExecutor* executor);
    
    std::string generate(const std::string& prompt, int maxTokens);
    std::vector<GenerationResponse> generateStream(...);
    
private:
    ITokenizer* tokenizer_;
    ModelExecutor* executor_;
};

} // namespace cllm
```

#### Step 2: 实现新类

```cpp
// 新建 src/tokenizer/generator.cpp
#include "cllm/tokenizer/generator.h"

namespace cllm {

Generator::Generator(ITokenizer* tokenizer, ModelExecutor* executor)
    : tokenizer_(tokenizer), executor_(executor) {}

std::string Generator::generate(const std::string& prompt, int maxTokens) {
    // 从TokenizerManager移植逻辑
    auto inputIds = tokenizer_->encode(prompt, true);
    auto outputIds = executor_->generate(inputIds, maxTokens);
    return tokenizer_->decode(outputIds, true);
}

// ... 其他方法实现 ...

} // namespace cllm
```

#### Step 3: 更新TokenizerManager使用新类

```cpp
// 修改 include/cllm/tokenizer/manager.h
class TokenizerManager {
public:
    // 保持接口兼容
    std::string generate(const std::string& prompt, int maxTokens) {
        return generator_->generate(prompt, maxTokens);
    }
    
private:
    std::unique_ptr<Generator> generator_;  // 新增
};

// 修改 src/tokenizer/manager.cpp
TokenizerManager::TokenizerManager(...) {
    // ...
    generator_ = std::make_unique<Generator>(tokenizer_, modelExecutor_);
}
```

#### Step 4: 运行测试验证

```bash
cd build
cmake .. && make -j8
./bin/test_tokenizer
./bin/test_generator  # 新增测试
```

#### Step 5: 清理旧代码 (可选)

```cpp
// 如果不需要保持兼容,可以移除TokenizerManager中的generate实现
// 让用户直接使用Generator
```

---

## 🚨 重构陷阱

### 陷阱1: 一次改太多

```markdown
❌ 错误:
1. 提取接口
2. 修改所有实现类
3. 更新所有调用点
4. 重命名
5. 添加新功能
↓
结果: 测试大面积失败,难以定位问题

✅ 正确:
1. 提取接口
   → 测试 ✅
2. 修改一个实现类
   → 测试 ✅
3. 修改另一个实现类
   → 测试 ✅
...
```

### 陷阱2: 破坏接口

```cpp
// ❌ 错误: 改变公共接口
class Tokenizer {
public:
    // 旧接口: std::vector<int> encode(const std::string&)
    // 新接口: Encoding encode(const std::string&)  // 破坏兼容性!
};

// ✅ 正确: 保持兼容或提供过渡
class Tokenizer {
public:
    // 保留旧接口
    std::vector<int> encode(const std::string& text) {
        return encodeV2(text).ids;
    }
    
    // 添加新接口
    Encoding encodeV2(const std::string& text);
};
```

### 陷阱3: 没有测试覆盖

```markdown
❌ 错误:
重构前没有测试 → 重构后不知道是否正确

✅ 正确:
1. 先补充测试
2. 确保测试通过
3. 开始重构
4. 每步后验证测试仍通过
```

### 陷阱4: 过度设计

```cpp
// ❌ 错误: 过度抽象
class ITokenizerFactory {
    virtual std::unique_ptr<ITokenizer> create() = 0;
};

class AbstractTokenizerFactoryBuilder {
    virtual ITokenizerFactory* build() = 0;
};

class TokenizerFactoryBuilderProvider {
    // 为了2个Tokenizer实现创建4层抽象...
};

// ✅ 正确: 简单直接
class TokenizerFactory {
    static std::unique_ptr<ITokenizer> create(const std::string& path);
};
```

---

## 📊 重构后验证

### 功能验证

```bash
# 1. 单元测试
./bin/test_tokenizer
./bin/test_model_executor
./bin/test_integration

# 2. 端到端测试
./bin/test_http_server_direct

# 3. 回归测试
python scripts/regression_test.py
```

### 性能验证

```bash
# 对比重构前后性能
./bin/benchmark_before > before.txt
./bin/benchmark_after > after.txt
diff before.txt after.txt
```

### 内存验证

```bash
# 检查内存泄漏
valgrind --leak-check=full ./bin/cllm_server
```

---

## 📚 重构模式参考

### 《重构:改善既有代码的设计》

- Extract Function (提取函数)
- Extract Class (提取类)
- Extract Interface (提取接口)
- Move Function (移动函数)
- Inline Function (内联函数)
- Replace Conditional with Polymorphism (用多态替换条件)
- Replace Type Code with Subclasses (用子类替换类型码)

### SOLID原则

- **S**ingle Responsibility (单一职责)
- **O**pen/Closed (开闭原则)
- **L**iskov Substitution (里氏替换)
- **I**nterface Segregation (接口隔离)
- **D**ependency Inversion (依赖倒置)

---

## 🎯 重构优先级

### 高优先级

1. **消除重复代码** - 最有价值
2. **简化复杂函数** - 提高可维护性
3. **解耦模块** - 降低依赖

### 中优先级

4. **统一命名** - 提高可读性
5. **优化性能** - 在瓶颈处
6. **补充文档** - 关键接口

### 低优先级

7. **美化格式** - 自动化工具处理
8. **重构测试** - 不影响功能
9. **优化注释** - 锦上添花

---

**最后更新**: 2026-01-11
