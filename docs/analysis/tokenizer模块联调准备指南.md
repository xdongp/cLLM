# Tokenizer 模块联调准备指南

**文档版本**: 1.0  
**准备日期**: 2026-01-10  
**目标**: 为 Tokenizer 模块与其他系统组件的联调测试提供实操指南

---

## 📋 快速评估

### 模块就绪状态

| 维度 | 状态 | 完成度 | 说明 |
|-----|------|-------|------|
| **核心功能** | ✅ 就绪 | 100% | 所有接口已实现 |
| **性能优化** | ✅ 就绪 | 100% | 批处理、缓存、监控已完成 |
| **测试覆盖** | ✅ 就绪 | 88% | 155+ 测试用例 |
| **文档** | ✅ 就绪 | 85% | 设计和实现文档完整 |
| **联调准备** | ✅ 就绪 | 95% | 仅需 CI 配置 |

**总体评估**: ✅ **可立即开始联调测试**

---

## 1. 联调场景清单

### 1.1 场景优先级

| 场景 | 优先级 | 依赖模块 | 预计工作量 | 风险 |
|------|--------|---------|-----------|------|
| Tokenizer ↔ ModelExecutor | 🔴 P0 | ModelExecutor | 4-6h | 低 |
| Tokenizer ↔ Server/API | 🔴 P0 | Server, HTTPServer | 6-8h | 低 |
| Tokenizer ↔ KVCache | 🟡 P1 | KVCache | 2-4h | 低 |
| 批处理性能验证 | 🟡 P1 | - | 4-6h | 中 |
| 端到端集成测试 | 🟢 P2 | 所有模块 | 8-12h | 高 |

---

## 2. 场景 1: Tokenizer ↔ ModelExecutor 联调

### 2.1 测试目标

验证分词器能为模型执行器提供正确的 token 序列

### 2.2 环境准备

```bash
# 1. 准备测试模型
cd /path/to/cLLM
mkdir -p model_test
# 下载 Qwen2-7B-Instruct 模型（或使用已有模型）

# 2. 编译测试程序
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make test_tokenizer_executor_integration

# 3. 配置环境变量
export MODEL_PATH="../model_test/qwen2-7b-instruct"
export LOG_LEVEL="DEBUG"
```

### 2.3 测试用例

#### 测试用例 1: 基础编解码

```cpp
// tests/integration/test_tokenizer_executor.cpp
TEST(TokenizerExecutorIntegration, BasicEncodeDecode) {
    // 1. 初始化分词器
    auto tokenizer = std::make_unique<NativeTokenizer>();
    ASSERT_TRUE(tokenizer->load(MODEL_PATH));
    
    // 2. 编码文本
    std::string prompt = "Hello, how are you?";
    auto tokens = tokenizer->encode(prompt, true);
    
    // 3. 验证 token 格式
    EXPECT_GT(tokens.size(), 0);
    EXPECT_EQ(tokens[0], tokenizer->getBosId());  // BOS token
    
    // 4. 传递给 ModelExecutor
    ModelExecutor executor;
    ASSERT_TRUE(executor.load(MODEL_PATH));
    
    auto output = executor.execute(tokens);
    EXPECT_GT(output.size(), 0);
    
    // 5. 解码输出
    std::string decoded = tokenizer->decode(output, true);
    EXPECT_FALSE(decoded.empty());
}
```

#### 测试用例 2: 长文本处理

```cpp
TEST(TokenizerExecutorIntegration, LongTextProcessing) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    
    // 生成 1000 字长文本
    std::string longText = generateLongText(1000);
    
    auto tokens = tokenizer->encode(longText, true);
    EXPECT_LT(tokens.size(), 10000);  // 合理的 token 数量
    
    // 验证与 ModelExecutor 兼容
    ModelExecutor executor;
    executor.load(MODEL_PATH);
    EXPECT_NO_THROW(executor.execute(tokens));
}
```

#### 测试用例 3: 特殊 token 处理

```cpp
TEST(TokenizerExecutorIntegration, SpecialTokens) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    
    // 测试系统提示词格式
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    auto tokens = tokenizer->encode(prompt, true);
    
    // 验证特殊 token 被正确识别
    EXPECT_TRUE(containsSpecialToken(tokens, "<|im_start|>"));
    EXPECT_TRUE(containsSpecialToken(tokens, "<|im_end|>"));
    
    ModelExecutor executor;
    executor.load(MODEL_PATH);
    auto output = executor.execute(tokens);
    EXPECT_GT(output.size(), 0);
}
```

### 2.4 性能基准

```cpp
TEST(TokenizerExecutorIntegration, PerformanceBenchmark) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->enablePerformanceMonitor(true);
    tokenizer->load(MODEL_PATH);
    
    ModelExecutor executor;
    executor.load(MODEL_PATH);
    
    // 测试 100 次编码 + 推理
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        std::string text = "Test prompt " + std::to_string(i);
        auto tokens = tokenizer->encode(text, true);
        executor.execute(tokens);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double avgLatency = duration.count() / 100.0;
    
    // 验证性能目标
    EXPECT_LT(avgLatency, 100);  // 平均 < 100ms
    
    auto stats = tokenizer->getPerformanceStats();
    std::cout << "Average encode latency: " << stats.avgEncodeLatency << " ms" << std::endl;
    std::cout << "P95 encode latency: " << stats.p95EncodeLatency << " ms" << std::endl;
}
```

### 2.5 预期结果

| 指标 | 预期值 | 验证方法 |
|------|--------|---------|
| Token 格式正确性 | 100% | 检查 BOS/EOS |
| 编码速度 | ≥ 50 MB/s | PerformanceStats |
| 内存占用 | ≤ 50 MB | 系统监控 |
| 错误率 | 0% | 异常捕获 |

---

## 3. 场景 2: Tokenizer ↔ Server/API 联调

### 3.1 测试目标

验证分词器能正确处理 HTTP 请求中的文本

### 3.2 环境准备

```bash
# 1. 启动测试服务器
cd build
./cllm_server --config ../config/server_test.yaml --port 8080

# 2. 在另一个终端运行测试
cd tests/integration
python3 test_tokenizer_api.py
```

### 3.3 API 测试用例

#### 测试用例 1: 基础编码 API

```python
# tests/integration/test_tokenizer_api.py
import requests
import json

def test_encode_api():
    url = "http://localhost:8080/v1/tokenize"
    payload = {
        "text": "Hello, world!",
        "add_special_tokens": True
    }
    
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "tokens" in data
    assert len(data["tokens"]) > 0
    print(f"Encoded tokens: {data['tokens']}")
```

#### 测试用例 2: 批量编码 API

```python
def test_batch_encode_api():
    url = "http://localhost:8080/v1/tokenize/batch"
    payload = {
        "texts": [
            "Hello, world!",
            "How are you?",
            "This is a test."
        ],
        "add_special_tokens": True
    }
    
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 3
    
    # 验证批处理性能提升
    assert data.get("batch_speedup", 1.0) >= 2.0
```

#### 测试用例 3: UTF-8 编码测试

```python
def test_utf8_encoding():
    url = "http://localhost:8080/v1/tokenize"
    
    # 测试多语言文本
    test_cases = [
        "Hello, world!",           # 英语
        "你好，世界！",             # 中文
        "Привет, мир!",            # 俄语
        "مرحبا بالعالم",           # 阿拉伯语
        "Hello 世界 Привет 🌍",   # 混合
    ]
    
    for text in test_cases:
        payload = {"text": text, "add_special_tokens": True}
        response = requests.post(url, json=payload)
        assert response.status_code == 200, f"Failed for: {text}"
        
        data = response.json()
        assert len(data["tokens"]) > 0
        print(f"✓ {text[:30]}: {len(data['tokens'])} tokens")
```

#### 测试用例 4: 性能压力测试

```python
import concurrent.futures
import time

def test_concurrent_requests():
    url = "http://localhost:8080/v1/tokenize"
    
    def send_request(i):
        payload = {"text": f"Test request {i}", "add_special_tokens": True}
        start = time.time()
        response = requests.post(url, json=payload)
        latency = (time.time() - start) * 1000  # ms
        return response.status_code, latency
    
    # 并发 100 个请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(send_request, i) for i in range(100)]
        results = [f.result() for f in futures]
    
    # 验证结果
    success_count = sum(1 for status, _ in results if status == 200)
    latencies = [lat for _, lat in results]
    
    assert success_count == 100, f"Only {success_count}/100 succeeded"
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    
    assert avg_latency < 50, "Average latency too high"
    assert p95_latency < 100, "P95 latency too high"
```

### 3.4 C++ 集成测试

```cpp
// tests/integration/test_server_tokenizer.cpp
TEST(ServerTokenizerIntegration, HttpRequestHandling) {
    // 1. 创建测试服务器
    ServerConfig config;
    config.port = 8080;
    config.model_path = MODEL_PATH;
    
    Server server(config);
    server.start();
    
    // 2. 创建 HTTP 客户端
    HttpClient client("http://localhost:8080");
    
    // 3. 发送编码请求
    json request = {
        {"text", "Hello, world!"},
        {"add_special_tokens", true}
    };
    
    auto response = client.post("/v1/tokenize", request);
    EXPECT_EQ(response.status_code, 200);
    
    auto data = json::parse(response.body);
    EXPECT_TRUE(data.contains("tokens"));
    EXPECT_GT(data["tokens"].size(), 0);
    
    server.stop();
}
```

### 3.5 预期结果

| 指标 | 预期值 | 验证方法 |
|------|--------|---------|
| API 可用性 | 100% | HTTP 200 |
| UTF-8 支持 | 完整 | 多语言测试 |
| 并发处理 | 100 QPS | 压力测试 |
| 平均延迟 | < 50 ms | 性能监控 |
| P95 延迟 | < 100 ms | 性能监控 |

---

## 4. 场景 3: 批处理性能验证

### 4.1 测试目标

验证批处理相比单线程的性能提升 ≥ 3x

### 4.2 测试代码

```cpp
// tests/performance/test_batch_performance.cpp
TEST(BatchPerformance, ThroughputComparison) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    
    // 准备测试数据
    std::vector<std::string> texts;
    for (int i = 0; i < 1000; ++i) {
        texts.push_back("This is test text number " + std::to_string(i));
    }
    
    // 测试 1: 单线程处理
    auto start1 = std::chrono::high_resolution_clock::now();
    for (const auto& text : texts) {
        tokenizer->encode(text, true);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    
    // 测试 2: 批处理（默认并行度）
    auto start2 = std::chrono::high_resolution_clock::now();
    auto result = BatchTokenizer::batchEncode(tokenizer.get(), texts, true, 0);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    
    // 计算加速比
    double speedup = static_cast<double>(duration1.count()) / duration2.count();
    
    std::cout << "Single-thread time: " << duration1.count() << " ms" << std::endl;
    std::cout << "Batch time: " << duration2.count() << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // 验证性能目标
    EXPECT_GE(speedup, 3.0) << "Batch processing speedup below target";
    
    // 验证结果正确性
    size_t successCount = std::count(result.success.begin(), result.success.end(), true);
    EXPECT_EQ(successCount, texts.size()) << "Some batch items failed";
}
```

### 4.3 不同场景测试

```cpp
TEST(BatchPerformance, VariousBatchSizes) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    
    std::vector<int> batchSizes = {1, 4, 8, 16, 32, 64, 128};
    
    for (int batchSize : batchSizes) {
        // 生成测试数据
        std::vector<std::string> texts(batchSize, "Test text for batch processing");
        
        // 测量批处理时间
        auto start = std::chrono::high_resolution_clock::now();
        auto result = BatchTokenizer::batchEncode(tokenizer.get(), texts, true, 4);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avgLatency = duration.count() / static_cast<double>(batchSize);
        
        std::cout << "Batch size " << batchSize 
                  << ": avg latency " << avgLatency << " μs/item" << std::endl;
    }
}
```

### 4.4 预期结果

| Batch Size | 预期加速比 | 预期成功率 |
|-----------|-----------|-----------|
| 1 | 1.0x | 100% |
| 8 | 2.5-3.5x | 100% |
| 32 | 3.5-4.5x | 100% |
| 128 | 4.0-5.0x | 100% |

---

## 5. 场景 4: 缓存效果验证

### 5.1 测试目标

验证缓存命中率 ≥ 50%（重复文本场景）

### 5.2 测试代码

```cpp
TEST(CachePerformance, HitRateValidation) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    tokenizer->enablePerformanceMonitor(true);
    
    // 设置高性能配置（启用缓存）
    auto config = TokenizerPerformanceConfig::getHighPerformance();
    tokenizer->setPerformanceConfig(config);
    
    // 准备测试数据（50% 重复）
    std::vector<std::string> texts;
    for (int i = 0; i < 100; ++i) {
        texts.push_back("Repeated text " + std::to_string(i % 50));
    }
    
    // 随机打乱顺序
    std::shuffle(texts.begin(), texts.end(), std::mt19937{std::random_device{}()});
    
    // 执行编码
    for (const auto& text : texts) {
        tokenizer->encode(text, true);
    }
    
    // 获取统计数据
    auto stats = tokenizer->getPerformanceStats();
    double hitRate = stats.getCacheHitRate();
    
    std::cout << "Cache hits: " << stats.cacheHits << std::endl;
    std::cout << "Cache misses: " << stats.cacheMisses << std::endl;
    std::cout << "Hit rate: " << hitRate * 100 << "%" << std::endl;
    
    // 验证缓存效果
    EXPECT_GE(hitRate, 0.45) << "Cache hit rate too low";
}
```

### 5.3 缓存性能对比

```cpp
TEST(CachePerformance, SpeedupMeasurement) {
    auto tokenizer = std::make_unique<NativeTokenizer>();
    tokenizer->load(MODEL_PATH);
    
    std::string repeatedText = "This is a repeated text for cache testing";
    
    // 测试 1: 无缓存
    auto config1 = TokenizerPerformanceConfig::getDefault();
    config1.cacheEnabled = false;
    tokenizer->setPerformanceConfig(config1);
    
    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        tokenizer->encode(repeatedText, true);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
    
    // 测试 2: 有缓存
    auto config2 = TokenizerPerformanceConfig::getDefault();
    config2.cacheEnabled = true;
    tokenizer->setPerformanceConfig(config2);
    
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        tokenizer->encode(repeatedText, true);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    
    // 计算加速比
    double speedup = static_cast<double>(duration1.count()) / duration2.count();
    
    std::cout << "Without cache: " << duration1.count() << " μs" << std::endl;
    std::cout << "With cache: " << duration2.count() << " μs" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    EXPECT_GE(speedup, 5.0) << "Cache speedup below expectation";
}
```

---

## 6. 问题排查指南

### 6.1 常见问题清单

#### 问题 1: 编码结果不一致

**症状**:
```
FAIL: Expected token count 10, got 12
```

**排查步骤**:
1. 检查特殊 token 设置
   ```cpp
   auto tokens = tokenizer->encode(text, false);  // 不添加特殊 token
   ```

2. 验证 Unicode 规范化
   ```cpp
   std::string normalized = UnicodeUtils::normalizeNFC(text);
   EXPECT_EQ(text, normalized);
   ```

3. 检查词汇表加载
   ```cpp
   EXPECT_GT(tokenizer->getVocabSize(), 0);
   EXPECT_NE(tokenizer->getBosId(), -1);
   ```

---

#### 问题 2: 批处理性能未达预期

**症状**:
```
Speedup: 1.5x (expected >= 3.0x)
```

**排查步骤**:
1. 检查并行线程数
   ```cpp
   auto config = tokenizer->getPerformanceConfig();
   std::cout << "Num threads: " << config.numThreads << std::endl;
   ```

2. 验证 CPU 核心数
   ```cpp
   int cores = std::thread::hardware_concurrency();
   EXPECT_GE(cores, 4);
   ```

3. 检查任务数量
   ```cpp
   // 任务数应 >= 线程数
   EXPECT_GE(texts.size(), config.numThreads * 2);
   ```

---

#### 问题 3: 缓存未生效

**症状**:
```
Cache hit rate: 0% (expected >= 50%)
```

**排查步骤**:
1. 确认缓存已启用
   ```cpp
   auto config = tokenizer->getPerformanceConfig();
   EXPECT_TRUE(config.cacheEnabled);
   ```

2. 检查缓存大小
   ```cpp
   EXPECT_GT(config.cacheMaxSize, 0);
   ```

3. 验证文本完全相同（包括空格）
   ```cpp
   std::string text1 = "Hello";
   std::string text2 = "Hello ";  // 末尾有空格
   EXPECT_NE(text1, text2);
   ```

---

#### 问题 4: 内存占用过高

**症状**:
```
Peak memory usage: 500 MB (expected <= 50 MB)
```

**排查步骤**:
1. 检查缓存配置
   ```cpp
   auto config = TokenizerPerformanceConfig::getLowMemory();
   tokenizer->setPerformanceConfig(config);
   ```

2. 监控内存使用
   ```cpp
   auto stats = tokenizer->getPerformanceStats();
   std::cout << "Current memory: " << stats.currentMemoryUsage / 1024 / 1024 << " MB" << std::endl;
   ```

3. 定期清理缓存
   ```cpp
   if (stats.currentMemoryUsage > 50 * 1024 * 1024) {
       tokenizer->clearCache();
   }
   ```

---

### 6.2 调试工具

#### 启用详细日志

```cpp
// 在代码中启用
tokenizer->setLogLevel(LogLevel::DEBUG);

// 或通过环境变量
export TOKENIZER_LOG_LEVEL=DEBUG
```

#### 性能监控

```cpp
// 启用性能监控
tokenizer->enablePerformanceMonitor(true);

// 定期打印统计
auto stats = tokenizer->getPerformanceStats();
std::cout << "=== Tokenizer Performance Stats ===" << std::endl;
std::cout << "Total encodes: " << stats.totalEncodes << std::endl;
std::cout << "Avg latency: " << stats.avgEncodeLatency << " ms" << std::endl;
std::cout << "P95 latency: " << stats.p95EncodeLatency << " ms" << std::endl;
std::cout << "Cache hit rate: " << stats.getCacheHitRate() * 100 << "%" << std::endl;
std::cout << "Memory usage: " << stats.currentMemoryUsage / 1024 / 1024 << " MB" << std::endl;
```

---

## 7. 持续集成配置

### 7.1 GitHub Actions 配置

```yaml
# .github/workflows/tokenizer_test.yml
name: Tokenizer Integration Tests

on:
  push:
    branches: [ main, dev ]
    paths:
      - 'src/tokenizer/**'
      - 'src/CTokenizer/**'
      - 'tests/**'
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake build-essential libgtest-dev
    
    - name: Download test model
      run: |
        mkdir -p model_test
        # 下载轻量级测试模型
        wget https://example.com/qwen2-test-model.bin -O model_test/qwen2.bin
    
    - name: Build tests
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON
        make -j$(nproc)
    
    - name: Run unit tests
      run: |
        cd build
        ctest --output-on-failure -L tokenizer
    
    - name: Run integration tests
      run: |
        cd build
        ./test_tokenizer_executor_integration
        ./test_tokenizer_api_integration
    
    - name: Performance benchmark
      run: |
        cd build
        ./test_batch_performance > perf_results.txt
        cat perf_results.txt
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: build/perf_results.txt
```

---

## 8. 验收标准

### 8.1 功能验收

| 验收项 | 标准 | 验证方法 |
|-------|------|---------|
| 基础编解码 | 100% 通过 | 单元测试 |
| 批处理功能 | 加速 ≥ 3x | 性能测试 |
| 缓存效果 | 命中率 ≥ 50% | 性能监控 |
| 特殊字符 | UTF-8 完整支持 | 字符集测试 |
| 错误处理 | 无崩溃 | 异常测试 |

### 8.2 性能验收

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| 编码速度 | ≥ 50 MB/s | 性能测试 |
| 平均延迟 | ≤ 10 ms | 监控数据 |
| P95 延迟 | ≤ 20 ms | 监控数据 |
| P99 延迟 | ≤ 50 ms | 监控数据 |
| 内存占用 | ≤ 50 MB | 系统监控 |
| 并发处理 | ≥ 100 QPS | 压力测试 |

### 8.3 稳定性验收

| 验收项 | 标准 | 验证方法 |
|-------|------|---------|
| 长时间运行 | 24h 无崩溃 | 稳定性测试 |
| 并发安全 | 无数据竞争 | 线程安全测试 |
| 内存泄漏 | 无泄漏 | Valgrind |
| 错误恢复 | 自动恢复 | 错误注入测试 |

---

## 9. 联调时间表

### 9.1 建议时间安排

| 阶段 | 任务 | 工作量 | 负责人 | 完成标志 |
|------|------|-------|--------|---------|
| **Week 1** | Tokenizer ↔ ModelExecutor | 6h | 开发 A | 所有集成测试通过 |
| **Week 1** | Tokenizer ↔ Server/API | 8h | 开发 B | API 测试通过 |
| **Week 2** | 批处理性能验证 | 6h | 开发 A | 性能达标 |
| **Week 2** | 缓存效果验证 | 4h | 开发 B | 命中率达标 |
| **Week 3** | 端到端集成测试 | 12h | 全员 | 所有场景通过 |
| **Week 3** | 性能调优 | 8h | 全员 | 性能指标达标 |
| **Week 4** | 文档和验收 | 6h | 全员 | 验收完成 |

**总工作量**: ~50 小时  
**建议团队规模**: 2-3 人  
**预计周期**: 4 周

---

## 10. 总结

### 10.1 就绪状态

✅ **Tokenizer 模块已完全就绪，可立即开始联调测试**

**关键优势**:
- ✅ 所有核心功能已实现且经过测试
- ✅ 性能优化（批处理、缓存）已完成
- ✅ 监控和配置系统完善
- ✅ 文档齐全，易于集成

### 10.2 关键建议

1. **优先进行 Tokenizer ↔ ModelExecutor 联调**（最关键）
2. **尽早配置 CI/CD**（自动化测试）
3. **启用性能监控**（及时发现问题）
4. **保持文档同步**（便于后续维护）

### 10.3 联系支持

如遇到问题，请参考：
- 📄 设计文档: `docs/modules/Tokenizer模块设计.md`
- 📊 完整性报告: `docs/analysis/src_tokenizer模块完整性分析报告_v2.md`
- 🔍 测试用例: `tests/test_tokenizer*.cpp`

---

**文档维护**: 请在联调过程中及时更新本文档  
**最后更新**: 2026-01-10
