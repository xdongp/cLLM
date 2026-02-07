# cLLM 测试指南

## 目录
- [概述](#概述)
- [测试分类](#测试分类)
- [测试工具库](#测试工具库)
- [编写测试](#编写测试)
- [运行测试](#运行测试)
- [最佳实践](#最佳实践)

## 概述

cLLM 项目采用分层、模块化的测试结构，包含以下测试类型：
- **单元测试**：测试单个类或函数的功能
- **集成测试**：测试多个模块的协作
- **性能测试**：测量和对比性能指标
- **压力测试**：测试系统在极端条件下的表现

## 测试分类

### 单元测试 (Unit Tests)
位于 `tests/unit/` 目录下，按模块组织：

```
tests/unit/
├── tokenizer/      # Tokenizer相关测试
├── scheduler/      # 调度器相关测试
├── cache/          # 缓存相关测试
├── backend/        # 后端相关测试
├── http/           # HTTP相关测试
└── ...
```

**特点**：
- 快速执行
- 无外部依赖
- 测试单一功能点
- 使用Mock对象

### 集成测试 (Integration Tests)
位于 `tests/integration/` 目录下：

```
tests/integration/
├── api/            # API集成测试
├── pipeline/       # 流水线集成测试
├── backend/        # 后端集成测试
└── system/         # 系统集成测试
```

**特点**：
- 测试真实交互
- 可能涉及文件I/O
- 测试端到端流程
- 执行时间较长

### 性能测试 (Performance Tests)
位于 `tests/performance/` 目录下：

```
tests/performance/
├── benchmark_inference.cpp    # 推理性能
├── benchmark_tokenizer.cpp    # Tokenizer性能
├── benchmark_throughput.cpp   # 吞吐量
└── benchmark_latency.cpp      # 延迟
```

**特点**：
- 测量性能指标
- 生成性能报告
- 用于性能回归检测

### 压力测试 (Stress Tests)
位于 `tests/stress/` 目录下：

```
tests/stress/
├── concurrency_stress_test.cpp  # 并发压力
├── memory_stress_test.cpp       # 内存压力
└── stability_stress_test.cpp    # 稳定性
```

**特点**：
- 测试极限情况
- 长时间运行
- 检测内存泄漏
- 验证系统稳定性

## 测试工具库

### 1. TestBase - 测试基类

所有测试的基类，提供通用功能：

```cpp
#include "utils/test_base.h"

class MyTest : public cllm::test::TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        // 自动创建临时目录
        auto tempDir = getTempTestDir();
    }
    // TearDown() 自动清理临时目录
};
```

**功能**：
- 自动管理临时测试目录
- 配置文件路径解析
- 资源清理

### 2. SchedulerTestBase - 调度器测试基类

用于需要Scheduler的测试：

```cpp
#include "utils/test_base.h"

class MySchedulerTest : public cllm::test::SchedulerTestBase {
protected:
    void SetUp() override {
        SchedulerTestBase::SetUp();
        createScheduler("", "", 4, 2048);
        // scheduler_ 已经创建并启动
    }
};
```

**功能**：
- 自动创建和销毁Scheduler
- 处理初始化异常
- 自动调用stop()

### 3. HttpEndpointTestBase - HTTP端点测试基类

用于HTTP端点测试：

```cpp
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"

class MyHttpTest : public cllm::test::HttpEndpointTestBase {
protected:
    void SetUp() override {
        HttpEndpointTestBase::SetUp();
        createScheduler();
        tokenizer_ = new MockTokenizer();
        endpoint_ = new GenerateEndpoint(scheduler_, tokenizer_);
    }
};
```

### 4. MockTokenizer - Mock Tokenizer

提供两种Mock实现：

#### MockTokenizer - 完整实现
```cpp
#include "utils/mock_tokenizer.h"

auto tokenizer = std::make_unique<cllm::test::MockTokenizer>();
tokenizer->setVocabSize(50000);
tokenizer->setSpecialTokenIds(1, 2, 0, 3);  // bos, eos, pad, unk

auto tokens = tokenizer->encode("Hello", true);
auto text = tokenizer->decode(tokens, true);
```

#### SimpleMockTokenizer - 简化实现
```cpp
#include "utils/mock_tokenizer.h"

// 返回固定token序列，适用于不关心具体编码的测试
auto tokenizer = std::make_unique<cllm::test::SimpleMockTokenizer>();
```

### 5. HttpTestHelpers - HTTP测试工具

创建和验证HTTP请求/响应：

```cpp
#include "utils/http_test_helpers.h"

using cllm::test::HttpTestHelpers;

// 创建请求
auto request = HttpTestHelpers::createGenerateRequest(
    "Hello, world!",  // prompt
    10,               // max_tokens
    0.7f,             // temperature
    0.9f,             // top_p
    false             // stream
);

// 验证响应
auto response = endpoint->handle(request);
auto jsonResponse = HttpTestHelpers::verifySuccessResponse(response, 200);
HttpTestHelpers::verifyGenerateResponseFields(jsonResponse);
```

**可用方法**：
- `createGenerateRequest()` - 创建生成请求
- `createTokenizeRequest()` - 创建tokenize请求
- `createDetokenizeRequest()` - 创建detokenize请求
- `verifySuccessResponse()` - 验证成功响应
- `verifyErrorResponse()` - 验证错误响应
- `verifyGenerateResponseFields()` - 验证生成响应字段
- `verifyConcurrencyLimitResponse()` - 验证并发限制响应

### 6. PerformanceTestHelpers - 性能测试工具

性能测量和基准测试：

```cpp
#include "utils/performance_test_helpers.h"

using cllm::test::PerformanceTestHelpers;

// 简单计时
PerformanceTestHelpers::Timer timer;
// ... 执行操作 ...
std::cout << "Elapsed: " << timer.elapsedMs() << " ms" << std::endl;

// 基准测试
auto testFunc = []() {
    // 要测试的操作
};

auto stats = PerformanceTestHelpers::benchmark(
    testFunc,
    1000,  // 迭代次数
    "My Benchmark"
);

stats.print("My Benchmark");

// 验证性能指标
EXPECT_LT(stats.avgTime, 10.0);  // 平均时间 < 10ms
EXPECT_GT(stats.throughput, 100.0);  // 吞吐量 > 100 ops/sec

// 并发基准测试
auto concurrentStats = PerformanceTestHelpers::concurrentBenchmark(
    testFunc,
    10,    // 线程数
    100,   // 每个线程的迭代次数
    "Concurrent Benchmark"
);
```

### 7. TestDataHelpers - 测试数据生成工具

生成各种测试数据：

```cpp
#include "utils/test_data_helpers.h"

using cllm::test::TestDataHelpers;

// 生成随机字符串
auto text = TestDataHelpers::generateRandomString(100);

// 生成随机中文字符串
auto chineseText = TestDataHelpers::generateRandomChineseString(50);

// 生成随机token序列
auto tokens = TestDataHelpers::generateRandomTokens(100, 10000);

// 获取测试提示词
auto prompts = TestDataHelpers::generateTestPrompts();
auto chinesePrompts = TestDataHelpers::generateChineseTestPrompts();

// 生成压力测试数据
auto stressData = TestDataHelpers::generateStressTestData(
    1000,  // 请求数量
    10,    // 最小tokens
    100    // 最大tokens
);

for (const auto& data : stressData) {
    auto request = HttpTestHelpers::createGenerateRequest(
        data.prompt,
        data.maxTokens,
        data.temperature,
        data.topP
    );
    // 执行请求...
}

// 创建临时测试文件
auto tempFile = TestDataHelpers::createTempFile(
    "file content",
    "test.txt",
    getTempTestDir()
);
```

## 编写测试

### 单元测试示例

```cpp
// tests/unit/tokenizer/tokenizer_interface_test.cpp
#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"
#include <cllm/tokenizer/i_tokenizer.h>

using namespace cllm;
using namespace cllm::test;

class TokenizerInterfaceTest : public TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        tokenizer_ = std::make_unique<MockTokenizer>();
    }
    
    std::unique_ptr<MockTokenizer> tokenizer_;
};

TEST_F(TokenizerInterfaceTest, EncodeBasicText) {
    std::string text = "Hello, world!";
    auto tokens = tokenizer_->encode(text, true);
    
    EXPECT_FALSE(tokens.empty());
    EXPECT_EQ(tokens[0], tokenizer_->getBosId());
}

TEST_F(TokenizerInterfaceTest, DecodeTokens) {
    std::vector<int> tokens = {1, 72, 101, 108, 108, 111};
    auto text = tokenizer_->decode(tokens, true);
    
    EXPECT_FALSE(text.empty());
}

TEST_F(TokenizerInterfaceTest, SpecialTokens) {
    EXPECT_NE(tokenizer_->getBosId(), tokenizer_->getEosId());
    EXPECT_TRUE(tokenizer_->isSpecialToken(tokenizer_->getBosId()));
    EXPECT_TRUE(tokenizer_->isSpecialToken(tokenizer_->getEosId()));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 集成测试示例

```cpp
// tests/integration/api/generate_api_test.cpp
#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"
#include "utils/http_test_helpers.h"
#include <cllm/http/generate_endpoint.h>

using namespace cllm;
using namespace cllm::test;

class GenerateApiTest : public HttpEndpointTestBase {
protected:
    void SetUp() override {
        HttpEndpointTestBase::SetUp();
        createScheduler();
        tokenizer_ = new MockTokenizer();
        endpoint_ = new GenerateEndpoint(scheduler_, tokenizer_);
    }
};

TEST_F(GenerateApiTest, BasicGeneration) {
    auto request = HttpTestHelpers::createGenerateRequest("Hello!", 10);
    auto response = endpoint_->handle(request);
    
    auto jsonResponse = HttpTestHelpers::verifySuccessResponse(response);
    HttpTestHelpers::verifyGenerateResponseFields(jsonResponse);
    
    auto data = jsonResponse["data"];
    EXPECT_FALSE(data["text"].get<std::string>().empty());
    EXPECT_GT(data["tokens_per_second"].get<float>(), 0.0f);
}

TEST_F(GenerateApiTest, MultipleRequests) {
    auto prompts = TestDataHelpers::generateTestPrompts();
    
    for (const auto& prompt : prompts) {
        auto request = HttpTestHelpers::createGenerateRequest(prompt, 10);
        auto response = endpoint_->handle(request);
        
        EXPECT_EQ(response.getStatusCode(), 200);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 性能测试示例

```cpp
// tests/performance/benchmark_tokenizer.cpp
#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"
#include "utils/performance_test_helpers.h"
#include "utils/test_data_helpers.h"

using namespace cllm;
using namespace cllm::test;

class TokenizerBenchmark : public TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        tokenizer_ = std::make_unique<MockTokenizer>();
        testTexts_ = TestDataHelpers::generateTestPrompts();
    }
    
    std::unique_ptr<MockTokenizer> tokenizer_;
    std::vector<std::string> testTexts_;
};

TEST_F(TokenizerBenchmark, EncodingPerformance) {
    auto testFunc = [this]() {
        for (const auto& text : testTexts_) {
            tokenizer_->encode(text, true);
        }
    };
    
    auto stats = PerformanceTestHelpers::benchmark(
        testFunc, 100, "Tokenizer Encoding");
    
    stats.print("Tokenizer Encoding");
    
    // 性能要求：平均时间 < 50ms
    EXPECT_LT(stats.avgTime, 50.0);
}

TEST_F(TokenizerBenchmark, ConcurrentEncoding) {
    auto testFunc = [this]() {
        tokenizer_->encode(testTexts_[0], true);
    };
    
    auto stats = PerformanceTestHelpers::concurrentBenchmark(
        testFunc, 10, 100, "Concurrent Encoding");
    
    stats.print("Concurrent Encoding");
    
    // 吞吐量要求：> 1000 ops/sec
    EXPECT_GT(stats.throughput, 1000.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 压力测试示例

```cpp
// tests/stress/concurrency_stress_test.cpp
#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"
#include "utils/http_test_helpers.h"
#include "utils/test_data_helpers.h"
#include <thread>
#include <atomic>

using namespace cllm;
using namespace cllm::test;

class ConcurrencyStressTest : public HttpEndpointTestBase {
protected:
    void SetUp() override {
        HttpEndpointTestBase::SetUp();
        createScheduler();
        tokenizer_ = new MockTokenizer();
        endpoint_ = new GenerateEndpoint(scheduler_, tokenizer_);
    }
};

TEST_F(ConcurrencyStressTest, HighConcurrencyLoad) {
    const int numThreads = 50;
    const int requestsPerThread = 100;
    
    std::atomic<int> successCount{0};
    std::atomic<int> errorCount{0};
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < requestsPerThread; ++j) {
                auto request = HttpTestHelpers::createGenerateRequest(
                    "Test prompt " + std::to_string(i * requestsPerThread + j),
                    10
                );
                
                auto response = endpoint_->handle(request);
                
                if (response.getStatusCode() == 200) {
                    successCount++;
                } else {
                    errorCount++;
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Success: " << successCount << std::endl;
    std::cout << "Errors: " << errorCount << std::endl;
    
    // 验证：至少80%的请求成功
    int totalRequests = numThreads * requestsPerThread;
    EXPECT_GT(successCount.load(), totalRequests * 0.8);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

## 运行测试

### 编译测试

```bash
cd build
cmake ..
make -j$(nproc)
```

### 运行所有测试

```bash
ctest --verbose
```

### 运行特定类别的测试

```bash
# 运行单元测试
ctest -R "unit_.*" --verbose

# 运行集成测试
ctest -R "integration_.*" --verbose

# 运行性能测试
ctest -R "benchmark_.*" --verbose

# 运行压力测试
ctest -R "stress_.*" --verbose
```

### 运行单个测试

```bash
./bin/tokenizer_interface_test
./bin/generate_api_test
./bin/benchmark_tokenizer
```

## 最佳实践

### 1. 测试命名
- 使用描述性的测试名称
- 格式：`TEST_F(TestClass, DoSomething_WhenCondition_ThenResult)`
- 示例：`TEST_F(TokenizerTest, Encode_WithSpecialTokens_ReturnsCorrectTokens)`

### 2. 测试结构
遵循 AAA 模式：
```cpp
TEST_F(MyTest, TestName) {
    // Arrange - 准备测试数据和环境
    auto tokenizer = std::make_unique<MockTokenizer>();
    std::string text = "Hello";
    
    // Act - 执行被测试的操作
    auto tokens = tokenizer->encode(text, true);
    
    // Assert - 验证结果
    EXPECT_FALSE(tokens.empty());
    EXPECT_EQ(tokens[0], tokenizer->getBosId());
}
```

### 3. 使用合适的断言
- `EXPECT_*` - 失败后继续执行
- `ASSERT_*` - 失败后停止执行
- `EXPECT_EQ`, `EXPECT_NE`, `EXPECT_LT`, `EXPECT_GT` 等

### 4. 避免测试依赖
- 每个测试应该独立运行
- 不要依赖其他测试的结果
- 在 SetUp() 中初始化，在 TearDown() 中清理

### 5. 使用Mock对象
- 单元测试使用Mock对象隔离依赖
- 集成测试使用真实对象

### 6. 性能测试注意事项
- 预热：在测量前运行几次
- 多次运行：取平均值
- 设置合理的性能阈值

### 7. 测试覆盖率
- 覆盖正常情况
- 覆盖边界情况
- 覆盖错误情况

### 8. 文档和注释
- 复杂的测试添加注释说明
- 说明测试的目的和验证的行为

## 故障排查

### 测试编译失败
- 检查头文件路径
- 确保包含了必要的依赖
- 查看编译错误信息

### 测试运行失败
- 查看详细错误信息
- 检查 SetUp/TearDown 是否正确
- 验证测试数据是否有效

### 测试超时
- 检查是否有死锁
- 增加超时时间
- 使用调试器定位问题

### 内存泄漏
- 使用 valgrind 检测：
  ```bash
  valgrind --leak-check=full ./bin/my_test
  ```
- 确保资源正确清理

## 贡献测试

添加新测试时：
1. 选择合适的测试类型和目录
2. 使用公共测试工具库
3. 遵循命名规范
4. 添加到 CMakeLists.txt
5. 确保测试通过
6. 提交代码审查

## 参考资料

- [Google Test 文档](https://google.github.io/googletest/)
- [cLLM 测试重构方案](TEST_REFACTORING_PLAN.md)
- [cLLM 架构设计](../docs/architecture/cLLM详细设计.md)
