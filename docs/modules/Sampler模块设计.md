# Sampler模块详细设计

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](../../C++编程规范.md)：定义编码风格、命名规范等
- [生成代码规范.md](../生成代码规范.md)：定义代码生成流程、设计文档一致性要求、优化同步机制等

## 0. 要生成的文件

### 0.1 头文件（include/cllm/）

根据[C++编程规范.md](../../C++编程规范.md)的命名规范，本模块需要生成以下头文件：

| 文件名 | 对应类/结构体 | 说明 |
|--------|--------------|------|
| `sampler.h` | `Sampler`, `SamplerConfig`, `SamplerStats` | 采样器主类、配置和统计信息 |

**注意**：Sampler模块的文件直接放在 `include/cllm/` 目录下，没有单独的sampler子目录。

### 0.2 源文件（src/sampler/）

| 文件名 | 对应头文件 | 说明 |
|--------|-----------|------|
| `sampler.cpp` | `sampler.h` | Sampler类的实现 |

### 0.3 测试文件（tests/）

| 文件名 | 测试目标 | 说明 |
|--------|---------|------|
| `test_sampler.cpp` | Sampler | 采样器模块的单元测试 |

### 0.4 文件命名规范说明

- **头文件名**：使用小写字母+下划线，与类名对应（大驼峰转小写下划线）
- **源文件名**：与对应头文件名保持一致
- **目录结构**：头文件位于 `include/cllm/`，源文件位于 `src/sampler/`
- **特殊性**：Sampler模块文件较少，直接放在cllm目录下
- **一致性原则**：所有文件命名遵循[C++编程规范.md](../../C++编程规范.md)第1.1节

## 1. 模块概述

### 1.1 模块职责
Sampler模块负责从模型输出的logits中进行token采样，提供多种采样策略，包括贪婪采样、Top-K采样、Top-P采样、温度采样等，并支持批量采样优化。

### 1.2 核心功能
- 贪婪采样：选择概率最高的token
- Top-K采样：从概率最高的K个token中采样
- Top-P采样：从累积概率达到P的token集合中采样
- 温度采样：通过温度参数控制采样的随机性
- 批量采样：优化处理多个序列的采样
- 性能优化：根据参数自动选择最优采样策略
- 性能统计：记录采样策略的使用情况
- 优化策略配置：支持不同的优化策略（速度、质量、平衡）

### 1.3 设计原则
- 简单性：避免复杂的C++语法，使用RAII包装器
- 高性能：使用SIMD指令加速计算
- 可扩展性：支持自定义采样策略
- 可维护性：清晰的代码结构和命名规范
- 灵活性：支持多种采样参数组合

### 1.4 技术选型
- 数学库：Eigen（线性代数计算）
- 随机数生成：C++11 <random>
- SIMD优化：AVX2/AVX-512指令集
- 线程安全：std::mutex和std::atomic

## 2. 类设计

### 2.1 SamplerConfig类

```cpp
class SamplerConfig {
public:
    SamplerConfig();
    ~SamplerConfig();
    
    void setTemperature(float temperature);
    void setTopK(int topK);
    void setTopP(float topP);
    void setGreedyThreshold(float threshold);
    void setFastTopKThreshold(int threshold);
    void setVocabLimit(int limit);
    
    float getTemperature() const;
    int getTopK() const;
    float getTopP() const;
    float getGreedyThreshold() const;
    int getFastTopKThreshold() const;
    int getVocabLimit() const;
    
    void loadFromJson(const nlohmann::json& json);
    nlohmann::json toJson() const;
    
private:
    float temperature_;
    int topK_;
    float topP_;
    float greedyThreshold_;
    int fastTopKThreshold_;
    int vocabLimit_;
};
```

### 2.2 SamplerStats类

```cpp
class SamplerStats {
public:
    SamplerStats();
    ~SamplerStats();
    
    void incrementTotalSamples();
    void incrementGreedySamples();
    void incrementFastTopKSamples();
    void incrementTemperatureSamples();
    void incrementTopPSamples();
    void incrementStandardSamples();
    
    long long getTotalSamples() const;
    long long getGreedySamples() const;
    long long getFastTopKSamples() const;
    long long getTemperatureSamples() const;
    long long getTopPSamples() const;
    long long getStandardSamples() const;
    
    float getGreedyPercentage() const;
    float getFastTopKPercentage() const;
    float getTemperaturePercentage() const;
    float getTopPPercentage() const;
    float getStandardPercentage() const;
    
    void reset();
    
    nlohmann::json toJson() const;
    
private:
    long long totalSamples_;
    long long greedySamples_;
    long long fastTopKSamples_;
    long long temperatureSamples_;
    long long topPSamples_;
    long long standardSamples_;
};
```

### 2.3 Sampler类

```cpp
class Sampler {
public:
    explicit Sampler(const SamplerConfig& config = SamplerConfig());
    ~Sampler();
    
    int sample(const FloatArray& logits);
    int sample(
        const FloatArray& logits,
        float temperature,
        int topK,
        float topP
    );
    
    std::vector<int> sampleBatch(
        const std::vector<FloatArray>& logitsBatch,
        const std::vector<float>& temperatures,
        const std::vector<int>& topKs,
        const std::vector<float>& topPs
    );
    
    void setConfig(const SamplerConfig& config);
    SamplerConfig getConfig() const;
    
    SamplerStats getStats() const;
    void resetStats();
    
    void setOptimizationStrategy(const std::string& strategy);
    std::string getOptimizationStrategy() const;
    
private:
    int sampleGreedy(const FloatArray& logits);
    int sampleTopK(const FloatArray& logits, int k);
    int sampleTopP(const FloatArray& logits, float topP, float temperature);
    int sampleTemperature(const FloatArray& logits, float temperature);
    int sampleStandard(const FloatArray& logits);
    
    int sampleTopKFast(const FloatArray& logits, int k);
    int sampleGreedyFast(const FloatArray& logits);
    
    std::vector<int> sampleBatchUniform(
        const std::vector<FloatArray>& logitsBatch,
        float temperature,
        int topK,
        float topP
    );
    
    void applyOptimizationStrategy(const std::string& strategy);
    
    SamplerConfig config_;
    SamplerStats stats_;
    
    std::string optimizationStrategy_;
    
    std::mt19937 randomGenerator_;
    std::mutex statsMutex_;
};
```

### 2.4 SamplingOptimizer类

```cpp
class SamplingOptimizer {
public:
    SamplingOptimizer();
    ~SamplingOptimizer();
    
    void applyStrategy(Sampler* sampler, const std::string& strategy);
    
    void registerStrategy(
        const std::string& name,
        const SamplerConfig& config
    );
    
    bool hasStrategy(const std::string& name) const;
    std::vector<std::string> getAvailableStrategies() const;
    
private:
    std::map<std::string, SamplerConfig> strategies_;
};
```

### 2.5 TopKSampler类

```cpp
class TopKSampler {
public:
    explicit TopKSampler(int k);
    ~TopKSampler();
    
    int sample(const FloatArray& logits);
    
    void setK(int k);
    int getK() const;
    
private:
    int k_;
    std::mt19937 randomGenerator_;
};
```

### 2.6 TopPSampler类

```cpp
class TopPSampler {
public:
    explicit TopPSampler(float topP, float temperature = 1.0f);
    ~TopPSampler();
    
    int sample(const FloatArray& logits);
    
    void setTopP(float topP);
    void setTemperature(float temperature);
    
    float getTopP() const;
    float getTemperature() const;
    
private:
    float topP_;
    float temperature_;
    std::mt19937 randomGenerator_;
};
```

### 2.7 TemperatureSampler类

```cpp
class TemperatureSampler {
public:
    explicit TemperatureSampler(float temperature = 1.0f);
    ~TemperatureSampler();
    
    int sample(const FloatArray& logits);
    
    void setTemperature(float temperature);
    float getTemperature() const;
    
private:
    float temperature_;
    std::mt19937 randomGenerator_;
};
```

### 2.8 GreedySampler类

```cpp
class GreedySampler {
public:
    GreedySampler();
    ~GreedySampler();
    
    int sample(const FloatArray& logits);
};
```

## 3. 接口设计

### 3.1 ISampler接口

```cpp
class ISampler {
public:
    virtual ~ISampler() {}
    
    virtual int sample(const FloatArray& logits) = 0;
    virtual int sample(
        const FloatArray& logits,
        float temperature,
        int topK,
        float topP
    ) = 0;
    
    virtual std::vector<int> sampleBatch(
        const std::vector<FloatArray>& logitsBatch,
        const std::vector<float>& temperatures,
        const std::vector<int>& topKs,
        const std::vector<float>& topPs
    ) = 0;
};
```

### 3.2 IOptimizedSampler接口

```cpp
class IOptimizedSampler {
public:
    virtual ~IOptimizedSampler() {}
    
    virtual void setOptimizationStrategy(const std::string& strategy) = 0;
    virtual std::string getOptimizationStrategy() const = 0;
    
    virtual SamplerStats getStats() const = 0;
    virtual void resetStats() = 0;
};
```

## 4. 算法设计

### 4.1 贪婪采样算法

```
算法：sampleGreedy
输入：logits (浮点数组)
输出：采样的token ID

1. 遍历logits数组
2. 找到最大值的索引
3. 返回该索引
```

### 4.2 Top-K采样算法

```
算法：sampleTopK
输入：logits (浮点数组), k (整数)
输出：采样的token ID

1. 找到logits中最大的k个值及其索引
2. 对这k个值计算softmax概率
3. 根据概率进行采样
4. 返回采样的索引
```

### 4.3 Top-P采样算法

```
算法：sampleTopP
输入：logits (浮点数组), topP (浮点数), temperature (浮点数)
输出：采样的token ID

1. 如果temperature != 1.0，对logits除以temperature
2. 计算softmax概率
3. 按概率降序排序
4. 计算累积概率
5. 找到累积概率超过topP的索引
6. 创建掩码，保留累积概率 <= topP的token
7. 确保至少保留一个token
8. 重新归一化概率
9. 根据归一化后的概率采样
10. 返回采样的索引
```

### 4.4 温度采样算法

```
算法：sampleTemperature
输入：logits (浮点数组), temperature (浮点数)
输出：采样的token ID

1. 对logits除以temperature
2. 生成Gumbel噪声：-log(-log(U))，其中U是均匀分布随机数
3. 将Gumbel噪声加到缩放后的logits上
4. 找到最大值的索引
5. 返回该索引
```

### 4.5 批量采样算法

```
算法：sampleBatch
输入：logitsBatch (logits数组列表), temperatures (浮点数列表), topKs (整数列表), topPs (浮点数列表)
输出：采样结果列表

1. 检查所有参数是否相同
2. 如果相同，使用sampleBatchUniform进行批量采样
3. 如果不同，按参数分组
4. 对每组使用相同的采样参数
5. 将结果放回对应位置
6. 返回采样结果列表
```

### 4.6 统一参数批量采样算法

```
算法：sampleBatchUniform
输入：logitsBatch (logits数组列表), temperature (浮点数), topK (整数), topP (浮点数)
输出：采样结果列表

1. 根据参数选择采样方法
2. 对每个logits应用相同的采样方法
3. 返回采样结果列表
```

## 5. 并发设计

### 5.1 并发模型
Sampler模块采用线程安全的并发模型：
- 每个采样操作独立处理
- 使用互斥锁保护统计数据
- 使用线程安全的随机数生成器
- 使用原子操作统计采样计数

### 5.2 线程模型
```
主线程
  ├── 采样线程1（处理序列1）
  ├── 采样线程2（处理序列2）
  ├── 采样线程3（处理序列3）
  └── 采样线程N（处理序列N）
```

### 5.3 锁策略
- **SamplerStats**：使用std::mutex保护统计数据的读写
- **Sampler**：无锁，采样操作是只读的
- **随机数生成器**：每个线程独立，无锁

### 5.4 并发安全
- 每个采样操作独立处理，不共享状态
- 使用RAII包装器管理资源生命周期
- 使用线程安全的随机数生成器
- 使用原子操作统计采样计数

### 5.5 性能优化
- 使用SIMD指令加速logits计算
- 使用无锁数据结构减少锁竞争
- 使用线程亲和性提高缓存命中率
- 使用批量采样减少函数调用开销

## 6. 内存管理

### 6.1 内存分配策略
- 使用mimalloc作为全局内存分配器
- 使用RAII包装器管理Sampler对象
- 使用预分配的缓冲区减少动态分配

### 6.2 RAII包装器
```cpp
class SamplerWrapper {
public:
    explicit SamplerWrapper(Sampler* sampler);
    ~SamplerWrapper();
    
    Sampler* get() const;
    Sampler* release();
    
private:
    Sampler* sampler_;
};
```

### 6.3 内存优化
- 使用浮点数组视图避免数据拷贝
- 使用移动语义减少对象拷贝
- 预分配排序缓冲区减少动态分配
- 使用内存池管理临时对象

### 6.4 内存监控
- 记录采样操作的内存使用
- 统计内存分配和释放次数
- 监控内存泄漏
- 设置内存使用上限

## 7. 错误处理

### 7.1 错误类型
```cpp
enum SamplerError {
    SAMPLER_OK = 0,
    SAMPLER_INVALID_LOGITS = 1,
    SAMPLER_INVALID_TEMPERATURE = 2,
    SAMPLER_INVALID_TOP_K = 3,
    SAMPLER_INVALID_TOP_P = 4,
    SAMPLER_INVALID_STRATEGY = 5
};

class SamplerException {
public:
    SamplerException(int code, const std::string& message);
    
    int getCode() const;
    std::string getMessage() const;
    
private:
    int code_;
    std::string message_;
};
```

### 7.2 错误处理策略
- 无效logits：返回SAMPLER_INVALID_LOGITS错误
- 无效温度：返回SAMPLER_INVALID_TEMPERATURE错误
- 无效Top-K：返回SAMPLER_INVALID_TOP_K错误
- 无效Top-P：返回SAMPLER_INVALID_TOP_P错误
- 无效策略：返回SAMPLER_INVALID_STRATEGY错误
- 所有错误都记录到日志

### 7.3 错误响应格式
```json
{
  "error": {
    "code": 2,
    "message": "Invalid temperature value"
  }
}
```

### 7.4 异常传播
- 使用try-catch捕获异常
- 将异常转换为SamplerException
- 记录异常堆栈到日志
- 不泄露敏感信息

## 8. 性能优化

### 8.1 SIMD优化
- 使用AVX2指令加速logits计算
- 使用AVX-512指令加速softmax计算
- 使用向量化操作加速排序
- 使用SIMD指令加速随机数生成

### 8.2 算法优化
- 使用快速选择算法找到Top-K
- 使用堆结构维护Top-K
- 使用快速排序优化Top-P采样
- 使用Gumbel-max技巧加速温度采样

### 8.3 批量优化
- 对相同参数的序列进行分组处理
- 使用向量化操作处理批量数据
- 减少函数调用开销
- 提高缓存命中率

### 8.4 内存优化
- 使用mimalloc提高内存分配性能
- 使用预分配缓冲区减少动态分配
- 使用内存对齐提高访问速度
- 使用零拷贝技术减少数据复制

### 8.5 性能监控
- 记录采样操作时间
- 统计不同采样策略的使用比例
- 监控批量采样性能
- 记录错误率

## 9. 测试设计

### 9.1 单元测试
- 测试GreedySampler的采样功能
- 测试TopKSampler的采样功能
- 测试TopPSampler的采样功能
- 测试TemperatureSampler的采样功能
- 测试Sampler的批量采样功能

### 9.2 集成测试
- 测试Sampler与ModelExecutor的集成
- 测试Sampler与Tokenizer的集成
- 测试不同采样策略的组合
- 测试批量采样的正确性
- 测试性能统计功能

### 9.3 性能测试
- 测试贪婪采样的速度
- 测试Top-K采样的速度
- 测试Top-P采样的速度
- 测试温度采样的速度
- 测试批量采样的性能

### 9.4 压力测试
- 测试最大批量采样大小
- 测试最大采样速率
- 测试内存泄漏
- 测试资源耗尽情况
- 测试恢复能力

### 9.5 测试用例示例

#### 9.5.1 贪婪采样测试
```cpp
TEST(SamplerTest, GreedySampling) {
    Sampler sampler;
    
    FloatArray logits;
    logits.data = new float[5]{0.1f, 0.2f, 0.5f, 0.1f, 0.1f};
    logits.size = 5;
    
    int token = sampler.sampleGreedy(logits);
    
    EXPECT_EQ(token, 2);
    
    delete[] logits.data;
}
```

#### 9.5.2 Top-K采样测试
```cpp
TEST(SamplerTest, TopKSampling) {
    Sampler sampler;
    
    FloatArray logits;
    logits.data = new float[5]{0.1f, 0.2f, 0.5f, 0.1f, 0.1f};
    logits.size = 5;
    
    int token = sampler.sampleTopK(logits, 3);
    
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 5);
    
    delete[] logits.data;
}
```

#### 9.5.3 Top-P采样测试
```cpp
TEST(SamplerTest, TopPSampling) {
    Sampler sampler;
    
    FloatArray logits;
    logits.data = new float[5]{0.1f, 0.2f, 0.5f, 0.1f, 0.1f};
    logits.size = 5;
    
    int token = sampler.sampleTopP(logits, 0.9f, 1.0f);
    
    EXPECT_GE(token, 0);
    EXPECT_LT(token, 5);
    
    delete[] logits.data;
}
```

#### 9.5.4 批量采样测试
```cpp
TEST(SamplerTest, BatchSampling) {
    Sampler sampler;
    
    std::vector<FloatArray> logitsBatch(2);
    for (int i = 0; i < 2; i++) {
        logitsBatch[i].data = new float[5]{0.1f, 0.2f, 0.5f, 0.1f, 0.1f};
        logitsBatch[i].size = 5;
    }
    
    std::vector<float> temperatures = {1.0f, 1.0f};
    std::vector<int> topKs = {0, 0};
    std::vector<float> topPs = {0.0f, 0.0f};
    
    std::vector<int> results = sampler.sampleBatch(
        logitsBatch, temperatures, topKs, topPs
    );
    
    EXPECT_EQ(results.size(), 2);
    
    for (int i = 0; i < 2; i++) {
        delete[] logitsBatch[i].data;
    }
}
```

## 10. 部署和配置

### 10.1 配置参数
```cpp
struct SamplerConfig {
    float temperature;
    int topK;
    float topP;
    float greedyThreshold;
    int fastTopKThreshold;
    int vocabLimit;
    std::string optimizationStrategy;
};
```

### 10.2 环境变量
- `SAMPLER_TEMPERATURE`: 默认温度参数
- `SAMPLER_TOP_K`: 默认Top-K参数
- `SAMPLER_TOP_P`: 默认Top-P参数
- `SAMPLER_GREEDY_THRESHOLD`: 贪婪采样阈值
- `SAMPLER_FAST_TOPK_THRESHOLD`: 快速Top-K采样阈值
- `SAMPLER_VOCAB_LIMIT`: 词汇表限制
- `SAMPLER_OPTIMIZATION_STRATEGY`: 优化策略

### 10.3 启动流程
1. 读取配置文件和环境变量
2. 初始化Sampler对象
3. 设置优化策略
4. 等待采样请求
5. 处理采样请求
6. 更新性能统计
7. 清理资源

## 11. 日志和监控

### 11.1 日志记录
- 记录每个采样请求的基本信息
- 记录采样策略的选择
- 记录采样处理时间
- 记录错误和异常
- 记录性能指标

### 11.2 监控指标
- 采样请求数
- 贪婪采样比例
- Top-K采样比例
- Top-P采样比例
- 温度采样比例
- 标准采样比例
- 平均采样时间
- 批量采样性能

### 11.3 健康检查
- 检查采样器是否正常工作
- 检查随机数生成器是否正常
- 检查内存使用情况
- 提供健康检查接口

## 12. 安全考虑

### 12.1 输入验证
- 验证logits的有效性
- 验证温度参数的范围
- 验证Top-K参数的范围
- 验证Top-P参数的范围

### 12.2 资源限制
- 限制最大批量采样大小
- 限制最大采样速率
- 限制内存使用量
- 限制采样时间

### 12.3 数据保护
- 安全存储随机数种子
- 使用安全的随机数生成器
- 防止预测攻击
- 定期更新依赖库

## 13. 总结

Sampler模块是cLLM系统的核心组件之一，负责从模型输出的logits中进行token采样。本设计文档详细描述了模块的类设计、接口设计、算法设计、并发设计、内存管理、错误处理、性能优化和测试设计等方面。

模块采用线程安全的并发模型，使用Eigen作为数学库，C++11 <random>作为随机数生成器，AVX2/AVX-512指令集进行SIMD优化。模块提供了贪婪采样、Top-K采样、Top-P采样、温度采样等多种采样策略，支持批量采样优化，具有良好的性能和可扩展性。

模块遵循C++编程规范，使用简单的C++语法，避免复杂的模板和智能指针，确保代码的可读性和可维护性。
