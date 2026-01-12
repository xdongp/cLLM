# Phase 4: 系统集成测试阶段 执行计划

**负责Agent**: Agent-4  
**预计耗时**: 24小时  
**依赖**: Phase 3 完成  
**执行时间**: T+44h ~ T+68h  

---

## 📋 阶段目标

对整个系统进行全面测试，包括功能测试、性能基准测试和压力稳定性测试。

---

## 📊 任务清单

| 子阶段 | 任务数 | 耗时 | 依赖 | 状态 |
|--------|--------|------|------|------|
| P4.1: 系统功能测试 | 4 | 6h | P3.3 | ⏳ 待执行 |
| P4.2: 性能基准测试 | 4 | 10h | P4.1 | ⏳ 待执行 |
| P4.3: 压力和稳定性测试 | 4 | 8h | P4.2 | ⏳ 待执行 |

**总计**: 16个任务，24小时

---

## 📝 详细任务说明

### P4.1: 系统功能测试 (6小时)

#### 测试重点
- Chat completion API
- Text completion API
- Streaming API
- Token counting
- API兼容性（OpenAI格式）
- 错误处理

#### 任务列表

**P4.1.1: 核心功能测试** (90分钟)

**Chat Completion测试**:
```cpp
TEST(SystemFunctionality, ChatCompletion) {
  // 启动完整系统
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {
      {{"role", "user"}, {"content", "What is 2+2?"}}
    }},
    {"max_tokens", 50},
    {"temperature", 0.7}
  };
  
  auto response = client.post("http://localhost:8080/v1/chat/completions", request);
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  EXPECT_TRUE(result.contains("choices"));
  EXPECT_GT(result["choices"].size(), 0);
  EXPECT_TRUE(result["choices"][0].contains("message"));
  EXPECT_TRUE(result["choices"][0]["message"].contains("content"));
  
  std::string answer = result["choices"][0]["message"]["content"];
  LOG(INFO) << "Answer: " << answer;
  
  server.stop();
}
```

**Text Completion测试**:
```cpp
TEST(SystemFunctionality, TextCompletion) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"model", "qwen2-0.5b"},
    {"prompt", "Once upon a time"},
    {"max_tokens", 100},
    {"temperature", 0.8}
  };
  
  auto response = client.post("http://localhost:8080/v1/completions", request);
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  EXPECT_TRUE(result.contains("choices"));
  EXPECT_FALSE(result["choices"][0]["text"].empty());
  
  server.stop();
}
```

**Streaming测试**:
```cpp
TEST(SystemFunctionality, Streaming) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Tell me a story."}}}},
    {"stream", true},
    {"max_tokens", 200}
  };
  
  std::vector<std::string> chunks;
  
  client.postStream("http://localhost:8080/v1/chat/completions", request, 
    [&](const std::string& chunk) {
      if (!chunk.empty() && chunk != "data: [DONE]\n\n") {
        chunks.push_back(chunk);
      }
    }
  );
  
  EXPECT_GT(chunks.size(), 0);
  
  LOG(INFO) << "Received " << chunks.size() << " chunks";
  
  server.stop();
}
```

**Token Counting测试**:
```cpp
TEST(SystemFunctionality, TokenCounting) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"text", "Hello, world! This is a test."}
  };
  
  auto response = client.post("http://localhost:8080/v1/tokenize", request);
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  EXPECT_TRUE(result.contains("tokens"));
  EXPECT_TRUE(result.contains("count"));
  EXPECT_GT(result["count"], 0);
  
  server.stop();
}
```

---

**P4.1.2: API 兼容性测试** (90分钟)

**OpenAI格式兼容性**:
```cpp
TEST(SystemFunctionality, OpenAICompatibility) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  // 使用标准OpenAI请求格式
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {
      {{"role", "system"}, {"content", "You are a helpful assistant."}},
      {{"role", "user"}, {"content", "Hello!"}}
    }},
    {"temperature", 0.7},
    {"top_p", 0.9},
    {"max_tokens", 100},
    {"presence_penalty", 0.0},
    {"frequency_penalty", 0.0}
  };
  
  auto response = client.post("http://localhost:8080/v1/chat/completions", request);
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  
  // 验证响应格式符合OpenAI规范
  EXPECT_TRUE(result.contains("id"));
  EXPECT_TRUE(result.contains("object"));
  EXPECT_EQ(result["object"], "chat.completion");
  EXPECT_TRUE(result.contains("created"));
  EXPECT_TRUE(result.contains("model"));
  EXPECT_TRUE(result.contains("choices"));
  EXPECT_TRUE(result.contains("usage"));
  EXPECT_TRUE(result["usage"].contains("prompt_tokens"));
  EXPECT_TRUE(result["usage"].contains("completion_tokens"));
  EXPECT_TRUE(result["usage"].contains("total_tokens"));
  
  server.stop();
}
```

---

**P4.1.3: 多场景测试** (90分钟)

**场景1: 事实问答**:
```cpp
TEST(SystemFunctionality, FactualQA) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<std::pair<std::string, std::vector<std::string>>> qa_pairs = {
    {"What is the capital of France?", {"Paris"}},
    {"Who wrote Romeo and Juliet?", {"Shakespeare", "William Shakespeare"}},
    {"What is 15 + 27?", {"42"}}
  };
  
  for (const auto& [question, expected_keywords] : qa_pairs) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", question}}}},
      {"max_tokens", 50}
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto result = json::parse(response.body);
    
    std::string answer = result["choices"][0]["message"]["content"];
    
    // 检查答案中是否包含预期关键词
    bool found = false;
    for (const auto& keyword : expected_keywords) {
      if (answer.find(keyword) != std::string::npos) {
        found = true;
        break;
      }
    }
    
    EXPECT_TRUE(found) << "Question: " << question << ", Answer: " << answer;
  }
  
  server.stop();
}
```

**场景2: 代码生成**:
```cpp
TEST(SystemFunctionality, CodeGeneration) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Write a Python function to calculate factorial."}}}},
    {"max_tokens", 200}
  };
  
  auto response = client.post("http://localhost:8080/v1/chat/completions", request);
  auto result = json::parse(response.body);
  
  std::string code = result["choices"][0]["message"]["content"];
  
  // 验证代码包含关键要素
  EXPECT_TRUE(code.find("def") != std::string::npos || code.find("function") != std::string::npos);
  EXPECT_TRUE(code.find("factorial") != std::string::npos);
  
  LOG(INFO) << "Generated code:\n" << code;
  
  server.stop();
}
```

---

**P4.1.4: 错误处理测试** (90分钟)

**各种错误情况**:
```cpp
TEST(SystemFunctionality, ErrorHandling) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  // 错误1: 缺少必要字段
  json invalid_request1 = {
    {"model", "qwen2-0.5b"}
    // 缺少 messages
  };
  auto response1 = client.post("http://localhost:8080/v1/chat/completions", invalid_request1);
  EXPECT_EQ(response1.status_code, 400);
  
  // 错误2: 无效的模型名
  json invalid_request2 = {
    {"model", "invalid-model"},
    {"messages", {{{"role", "user"}, {"content", "test"}}}}
  };
  auto response2 = client.post("http://localhost:8080/v1/chat/completions", invalid_request2);
  EXPECT_TRUE(response2.status_code == 400 || response2.status_code == 404);
  
  // 错误3: max_tokens 超限
  json invalid_request3 = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "test"}}}},
    {"max_tokens", 100000} // 远超模型限制
  };
  auto response3 = client.post("http://localhost:8080/v1/chat/completions", invalid_request3);
  EXPECT_TRUE(response3.status_code == 400 || response3.status_code == 200);
  
  // 错误4: 无效的 temperature
  json invalid_request4 = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "test"}}}},
    {"temperature", -1.0} // 无效值
  };
  auto response4 = client.post("http://localhost:8080/v1/chat/completions", invalid_request4);
  EXPECT_EQ(response4.status_code, 400);
  
  // 系统应该仍然正常工作
  json valid_request = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Are you OK?"}}}},
    {"max_tokens", 20}
  };
  auto response5 = client.post("http://localhost:8080/v1/chat/completions", valid_request);
  EXPECT_EQ(response5.status_code, 200);
  
  server.stop();
}
```

**验收标准**:
- ✅ 所有核心功能正常
- ✅ API 响应符合 OpenAI 格式
- ✅ 多场景测试通过
- ✅ 错误处理健壮

---

### P4.2: 性能基准测试 (10小时)

#### 测试重点
- 吞吐量测试
- 延迟测试（P50/P95/P99）
- 资源使用测试
- 扩展性测试

#### 任务列表

**P4.2.1: 吞吐量测试** (150分钟)

**单请求吞吐量**:
```cpp
TEST(PerformanceBenchmark, SingleRequestThroughput) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Count from 1 to 100."}}}},
    {"max_tokens", 200}
  };
  
  auto start = std::chrono::high_resolution_clock::now();
  
  auto response = client.post("http://localhost:8080/v1/chat/completions", request);
  
  auto end = std::chrono::high_resolution_clock::now();
  
  EXPECT_EQ(response.status_code, 200);
  
  auto result = json::parse(response.body);
  int completion_tokens = result["usage"]["completion_tokens"];
  
  double duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
    end - start
  ).count() / 1000.0;
  
  double tokens_per_sec = completion_tokens / duration_sec;
  
  LOG(INFO) << "Single Request Throughput: " << tokens_per_sec << " tokens/sec";
  LOG(INFO) << "Completion tokens: " << completion_tokens;
  LOG(INFO) << "Duration: " << duration_sec << " sec";
  
  // 目标: > 100 tokens/sec
  EXPECT_GT(tokens_per_sec, 100);
  
  server.stop();
}
```

**批处理吞吐量**:
```cpp
TEST(PerformanceBenchmark, BatchThroughput) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  const int BATCH_SIZE = 8;
  std::vector<std::thread> threads;
  std::atomic<int> total_tokens{0};
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < BATCH_SIZE; ++i) {
    threads.emplace_back([&, i]() {
      HTTPClient client;
      json request = {
        {"model", "qwen2-0.5b"},
        {"messages", {{{"role", "user"}, {"content", "Test " + std::to_string(i)}}}},
        {"max_tokens", 50}
      };
      
      auto response = client.post("http://localhost:8080/v1/chat/completions", request);
      
      if (response.status_code == 200) {
        auto result = json::parse(response.body);
        total_tokens += result["usage"]["completion_tokens"].get<int>();
      }
    });
  }
  
  for (auto& thread : threads) {
    thread.join();
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  
  double duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
    end - start
  ).count() / 1000.0;
  
  double throughput = total_tokens.load() / duration_sec;
  
  LOG(INFO) << "Batch Throughput (" << BATCH_SIZE << "): " << throughput << " tokens/sec";
  LOG(INFO) << "Total tokens: " << total_tokens.load();
  LOG(INFO) << "Duration: " << duration_sec << " sec";
  
  server.stop();
}
```

---

**P4.2.2: 延迟测试（P50/P95/P99）** (150分钟)

```cpp
TEST(PerformanceBenchmark, LatencyDistribution) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  const int NUM_REQUESTS = 100;
  std::vector<double> latencies;
  
  HTTPClient client;
  
  for (int i = 0; i < NUM_REQUESTS; ++i) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", "Hello"}}}},
      {"max_tokens", 10}
    };
    
    auto start = std::chrono::high_resolution_clock::now();
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (response.status_code == 200) {
      double latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
      ).count();
      latencies.push_back(latency);
    }
  }
  
  std::sort(latencies.begin(), latencies.end());
  
  double p50 = latencies[NUM_REQUESTS * 50 / 100];
  double p95 = latencies[NUM_REQUESTS * 95 / 100];
  double p99 = latencies[NUM_REQUESTS * 99 / 100];
  double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
  
  LOG(INFO) << "Latency Distribution:";
  LOG(INFO) << "  Mean: " << mean << " ms";
  LOG(INFO) << "  P50: " << p50 << " ms";
  LOG(INFO) << "  P95: " << p95 << " ms";
  LOG(INFO) << "  P99: " << p99 << " ms";
  
  // 性能目标
  EXPECT_LT(p50, 50);   // P50 < 50ms
  EXPECT_LT(p95, 100);  // P95 < 100ms
  EXPECT_LT(p99, 200);  // P99 < 200ms
  
  server.stop();
}
```

---

**P4.2.3: 资源使用测试** (150分钟)
**P4.2.4: 扩展性测试** (150分钟)

**验收标准**:
- ✅ 吞吐量 > 100 tokens/sec
- ✅ P50 延迟 < 50ms
- ✅ P95 延迟 < 100ms
- ✅ P99 延迟 < 200ms
- ✅ 内存使用 < 8GB
- ✅ CPU 使用合理

---

### P4.3: 压力和稳定性测试 (8小时)

#### 测试重点
- 高并发测试（8 并发）
- 长时间运行（5 分钟）
- 异常注入测试
- 恢复测试

#### 任务列表

**P4.3.1: 高并发测试（8 并发）** (8)
**P4.3.2: 长时间运行（5 分钟）** (5)


**验收标准**:
- ✅ 8 并发无错误
- ✅ 长时间运行稳定（无内存泄漏）
- ✅ 异常情况正确处理
- ✅ 系统可恢复

---

## ✅ 总体验收标准

### 必须完成

- [ ] P4.1: 系统功能测试通过
- [ ] P4.2: 性能基准达标
- [ ] P4.3: 压力稳定性测试通过

### 质量指标

- [ ] 所有API正常工作
- [ ] 性能指标全部达标
- [ ] 100 并发测试通过
- [ ] 长时间运行无崩溃

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P4.1: ☐ 完成 / ☐ 失败
- P4.2: ☐ 完成 / ☐ 失败
- P4.3: ☐ 完成 / ☐ 失败

**性能指标**:
- 吞吐量: ________ tokens/sec
- P50 延迟: ________ ms
- P95 延迟: ________ ms
- P99 延迟: ________ ms
- 最大并发: ________ 
- 长时间运行: ________ 分钟

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

---

## 🔄 下一步

Phase 4 完成后，通知 Agent-5 启动 Phase 5:

```bash
touch /tmp/cllm_test_locks/phase4.done
echo "✅ Phase 4 完成，Agent-5 可以启动 Phase 5"
```
