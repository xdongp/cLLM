# Phase 3: 子系统测试阶段 执行计划

**负责Agent**: Agent-3  
**预计耗时**: 15小时  
**依赖**: Phase 2 完成  
**执行时间**: T+29h ~ T+44h  

---

## 📋 阶段目标

验证子系统级别的功能集成，测试完整的功能流程和子系统性能。

---

## 📊 任务清单

| 子阶段 | 任务数 | 耗时 | 依赖 | 状态 |
|--------|--------|------|------|------|
| P3.1: 前端子系统（HTTP + Tokenizer） | 4 | 4h | P2.1 | ⏳ 待执行 |
| P3.2: 推理子系统（Executor + Backend + Qwen3） | 4 | 5h | P2.3, P2.4 | ⏳ 待执行 |
| P3.3: E2E子系统（Tokenizer → Executor → Backend） | 4 | 6h | P2.2, P3.2 | ⏳ 待执行 |

**总计**: 12个任务，15小时

---

## 📝 详细任务说明

### P3.1: 前端子系统测试（HTTP + Tokenizer） (4小时)

#### 测试重点
- 完整的 HTTP 请求 → 响应流程
- 并发处理能力
- 性能指标（延迟、吞吐量）
- 容错能力

#### 任务列表

**P3.1.1: 完整流程测试** (60分钟)
```cpp
TEST(FrontendSubsystem, CompleteFlow) {
  // 启动 HTTP Server
  HTTPServer server("0.0.0.0", 8080);
  
  // 加载 Tokenizer
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  server.registerTokenizer(&tokenizer);
  
  server.start();
  
  // 发送HTTP请求
  HTTPClient client;
  json request_body = {
    {"text", "Hello, world! This is a test."}
  };
  
  auto response = client.post("http://localhost:8080/v1/tokenize", request_body);
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  EXPECT_TRUE(result.contains("tokens"));
  EXPECT_GT(result["tokens"].size(), 0);
  
  server.stop();
}
```

**P3.1.2: 并发测试（50 并发）** (60分钟)
```cpp
TEST(FrontendSubsystem, ConcurrentRequests) {
  HTTPServer server("0.0.0.0", 8080);
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  server.registerTokenizer(&tokenizer);
  server.start();
  
  const int NUM_THREADS = 50;
  const int REQUESTS_PER_THREAD = 10;
  
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};
  std::atomic<int> error_count{0};
  
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads.emplace_back([&, i]() {
      HTTPClient client;
      for (int j = 0; j < REQUESTS_PER_THREAD; ++j) {
        json request = {{"text", "Test " + std::to_string(i * 100 + j)}};
        auto response = client.post("http://localhost:8080/v1/tokenize", request);
        
        if (response.status_code == 200) {
          success_count++;
        } else {
          error_count++;
        }
      }
    });
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  EXPECT_EQ(success_count, NUM_THREADS * REQUESTS_PER_THREAD);
  EXPECT_EQ(error_count, 0);
  
  server.stop();
}
```

**P3.1.3: 性能测试（延迟/吞吐量）** (60分钟)
```cpp
TEST(FrontendSubsystem, PerformanceMetrics) {
  HTTPServer server("0.0.0.0", 8080);
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  server.registerTokenizer(&tokenizer);
  server.start();
  
  HTTPClient client;
  const int NUM_REQUESTS = 100;
  std::vector<double> latencies;
  
  auto start_time = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < NUM_REQUESTS; ++i) {
    auto req_start = std::chrono::high_resolution_clock::now();
    
    json request = {{"text", "Performance test text"}};
    auto response = client.post("http://localhost:8080/v1/tokenize", request);
    
    auto req_end = std::chrono::high_resolution_clock::now();
    double latency = std::chrono::duration_cast<std::chrono::milliseconds>(
      req_end - req_start
    ).count();
    latencies.push_back(latency);
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::seconds>(
    end_time - start_time
  ).count();
  
  // 计算统计指标
  std::sort(latencies.begin(), latencies.end());
  double p50 = latencies[NUM_REQUESTS * 50 / 100];
  double p95 = latencies[NUM_REQUESTS * 95 / 100];
  double p99 = latencies[NUM_REQUESTS * 99 / 100];
  double throughput = NUM_REQUESTS / total_time;
  
  LOG(INFO) << "Performance Metrics:";
  LOG(INFO) << "  P50 Latency: " << p50 << " ms";
  LOG(INFO) << "  P95 Latency: " << p95 << " ms";
  LOG(INFO) << "  P99 Latency: " << p99 << " ms";
  LOG(INFO) << "  Throughput: " << throughput << " req/s";
  
  // 验证性能目标
  EXPECT_LT(p99, 100); // P99 < 100ms
  EXPECT_GT(throughput, 10); // > 10 req/s
  
  server.stop();
}
```

**P3.1.4: 容错测试** (60分钟)
```cpp
TEST(FrontendSubsystem, FaultTolerance) {
  HTTPServer server("0.0.0.0", 8080);
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  server.registerTokenizer(&tokenizer);
  server.start();
  
  HTTPClient client;
  
  // 测试1: 无效输入
  json invalid_request = {{"invalid_field", "test"}};
  auto response1 = client.post("http://localhost:8080/v1/tokenize", invalid_request);
  EXPECT_EQ(response1.status_code, 400); // Bad Request
  
  // 测试2: 空输入
  json empty_request = {{"text", ""}};
  auto response2 = client.post("http://localhost:8080/v1/tokenize", empty_request);
  EXPECT_TRUE(response2.status_code == 200 || response2.status_code == 400);
  
  // 测试3: 超长输入
  std::string long_text(10000, 'a');
  json long_request = {{"text", long_text}};
  auto response3 = client.post("http://localhost:8080/v1/tokenize", long_request);
  EXPECT_TRUE(response3.status_code == 200 || response3.status_code == 413); // Payload Too Large
  
  // 测试4: 无效路径
  auto response4 = client.post("http://localhost:8080/invalid/path", {});
  EXPECT_EQ(response4.status_code, 404);
  
  // 系统应该仍然正常工作
  json valid_request = {{"text", "Test after errors"}};
  auto response5 = client.post("http://localhost:8080/v1/tokenize", valid_request);
  EXPECT_EQ(response5.status_code, 200);
  
  server.stop();
}
```

**验收标准**:
- ✅ 完整请求→响应流程正常
- ✅ 50 并发无错误
- ✅ P99 延迟 < 100ms
- ✅ 吞吐量 > 10 req/s
- ✅ 异常情况正确处理

---

### P3.2: 推理子系统测试（Executor + Backend + Qwen3） (5小时)

#### 测试重点
- 完整推理流程
- 批处理性能
- 推理吞吐量
- 输出质量

#### 任务列表

**P3.2.1: 完整推理流程** (75分钟)
```cpp
TEST(InferenceSubsystem, CompletePipeline) {
  // 初始化组件
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  
  executor.setBackend(&backend);
  
  // 执行推理
  std::vector<int> input_ids = {1, 15339, 11, 1917, 0, 2}; // "Hello, world!"
  auto output_ids = executor.generate(input_ids, 50);
  
  EXPECT_GT(output_ids.size(), input_ids.size());
  EXPECT_LE(output_ids.size(), input_ids.size() + 50);
  
  // 验证输出的合理性
  for (auto id : output_ids) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, 32000); // Qwen3 vocab size
  }
}
```

**P3.2.2: 批处理测试** (75分钟)
**P3.2.3: 性能测试** (75分钟)
**P3.2.4: 输出质量测试** (75分钟)

**验收标准**:
- ✅ 推理流程完整正确
- ✅ 批处理性能达标
- ✅ 吞吐量 > 100 tokens/s
- ✅ 输出质量良好

---

### P3.3: E2E 子系统测试（Tokenizer → Executor → Backend） (6小时)

#### 测试重点
- 文本到文本完整链路
- 流式输出
- 多轮对话
- 边界测试（长输入/输出）

#### 任务列表

**P3.3.1: 文本到文本完整链路** (90分钟)
```cpp
TEST(E2ESubsystem, TextToText) {
  // 初始化所有组件
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  executor.setBackend(&backend);
  
  // 完整流程：文本 → Token IDs → 推理 → Token IDs → 文本
  std::string input_text = "What is artificial intelligence?";
  
  auto input_ids = tokenizer.encode(input_text);
  auto output_ids = executor.generate(input_ids, 100);
  auto output_text = tokenizer.decode(output_ids);
  
  EXPECT_FALSE(output_text.empty());
  EXPECT_GT(output_text.length(), input_text.length());
  
  LOG(INFO) << "Input: " << input_text;
  LOG(INFO) << "Output: " << output_text;
}
```

**P3.3.2: 流式输出测试** (90分钟)
```cpp
TEST(E2ESubsystem, StreamingOutput) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  executor.setBackend(&backend);
  
  std::string prompt = "Write a short story:";
  auto input_ids = tokenizer.encode(prompt);
  
  std::vector<std::string> chunks;
  
  executor.generateStreaming(input_ids, 100, [&](const std::vector<int>& new_ids) {
    std::string chunk = tokenizer.decode(new_ids);
    chunks.push_back(chunk);
    LOG(INFO) << "Chunk: " << chunk;
  });
  
  EXPECT_GT(chunks.size(), 0);
  
  // 拼接所有chunk
  std::string full_output;
  for (const auto& chunk : chunks) {
    full_output += chunk;
  }
  
  EXPECT_FALSE(full_output.empty());
}
```

**P3.3.3: 多轮对话测试** (90分钟)
**P3.3.4: 边界测试（长输入/输出）** (90分钟)

**验收标准**:
- ✅ 端到端流程正常
- ✅ 流式输出正确
- ✅ 多轮上下文正确
- ✅ 边界情况处理正确（长输入不崩溃）

---

## ✅ 总体验收标准

### 必须完成

- [ ] P3.1: 前端子系统测试通过
- [ ] P3.2: 推理子系统测试通过
- [ ] P3.3: E2E子系统测试通过

### 质量指标

- [ ] 子系统测试覆盖率 > 80%
- [ ] 性能指标达标
- [ ] 并发测试通过
- [ ] 容错能力良好

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P3.1: ☐ 完成 / ☐ 失败
- P3.2: ☐ 完成 / ☐ 失败
- P3.3: ☐ 完成 / ☐ 失败

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

---

## 🔄 下一步

Phase 3 完成后，通知 Agent-4 启动 Phase 4:

```bash
touch /tmp/cllm_test_locks/phase3.done
echo "✅ Phase 3 完成，Agent-4 可以启动 Phase 4"
```
