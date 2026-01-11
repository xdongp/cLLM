# Phase 2: 模块集成测试阶段 执行计划

**负责Agent**: Agent-2  
**预计耗时**: 14小时  
**依赖**: Phase 1 完成  
**执行时间**: T+15h ~ T+29h  

---

## 📋 阶段目标

验证相邻模块之间的接口集成和数据流，确保模块间可以正确协作。

---

## 📊 任务清单

| 子阶段 | 任务数 | 耗时 | 依赖 | 状态 |
|--------|--------|------|------|------|
| P2.1: HTTP + Tokenizer 集成 | 4 | 3h | P1.1, P1.2 | ⏳ 待执行 |
| P2.2: Tokenizer + Executor 集成 | 4 | 4h | P1.2, P1.3 | ⏳ 待执行 |
| P2.3: Executor + Backend 集成 | 4 | 4h | P1.3, P1.4 | ⏳ 待执行 |
| P2.4: Backend + Qwen3 集成 | 4 | 3h | P1.4, P1.5 | ⏳ 待执行 |

**总计**: 16个任务，14小时

---

## 📝 详细任务说明

### P2.1: HTTP + Tokenizer 集成 (3小时)

#### 测试重点
- `/v1/tokenize` 端点集成
- `/v1/detokenize` 端点集成
- HTTP → Tokenizer 数据流
- 错误传播

#### 关键测试用例

**P2.1.1: `/v1/tokenize` 端点测试** (45分钟)
```cpp
TEST(HTTPTokenizerIntegration, TokenizeEndpoint) {
  HTTPServer server;
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  server.registerTokenizer(&tokenizer);
  
  HTTPRequest request;
  request.path = "/v1/tokenize";
  request.method = "POST";
  request.body = R"({"text":"Hello, world!"})";
  
  auto response = server.handle(request);
  
  EXPECT_EQ(response.statusCode, 200);
  auto json = response.parseJSON();
  EXPECT_TRUE(json.contains("tokens"));
  EXPECT_GT(json["tokens"].size(), 0);
}
```

**P2.1.2: `/v1/detokenize` 端点测试** (45分钟)
```cpp
TEST(HTTPTokenizerIntegration, DetokenizeEndpoint) {
  HTTPServer server;
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  server.registerTokenizer(&tokenizer);
  
  HTTPRequest request;
  request.path = "/v1/detokenize";
  request.method = "POST";
  request.body = R"({"tokens":[100, 200, 300]})";
  
  auto response = server.handle(request);
  
  EXPECT_EQ(response.statusCode, 200);
  auto json = response.parseJSON();
  EXPECT_TRUE(json.contains("text"));
}
```

**P2.1.3: 错误传播测试** (45分钟)
```cpp
TEST(HTTPTokenizerIntegration, ErrorPropagation) {
  HTTPServer server;
  HFTokenizer tokenizer;
  // 不加载tokenizer，模拟错误
  
  server.registerTokenizer(&tokenizer);
  
  HTTPRequest request;
  request.path = "/v1/tokenize";
  request.body = R"({"text":"test"})";
  
  auto response = server.handle(request);
  
  EXPECT_NE(response.statusCode, 200);
  EXPECT_TRUE(response.body.find("error") != std::string::npos);
}
```

**P2.1.4: 批量请求测试** (45分钟)
```cpp
TEST(HTTPTokenizerIntegration, BatchRequests) {
  HTTPServer server;
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  server.registerTokenizer(&tokenizer);
  
  HTTPRequest request;
  request.path = "/v1/tokenize/batch";
  request.body = R"({"texts":["First text", "Second text", "Third text"]})";
  
  auto response = server.handle(request);
  
  EXPECT_EQ(response.statusCode, 200);
  auto json = response.parseJSON();
  EXPECT_EQ(json["results"].size(), 3);
}
```

**验收标准**:
- ✅ Tokenize 端点正常工作
- ✅ Detokenize 端点正常工作
- ✅ 错误正确传播到 HTTP 响应
- ✅ 批量请求正确处理

---

### P2.2: Tokenizer + Executor 集成 (4小时)

#### 测试重点
- Token IDs → Tensor 转换
- Tokenizer → Executor 数据流
- 批处理集成
- 状态同步

#### 关键测试用例

**P2.2.1: 数据格式转换测试** (60分钟)
```cpp
TEST(TokenizerExecutorIntegration, DataConversion) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  executor.setBackend(std::make_unique<MockBackend>());
  
  std::string text = "Hello, world!";
  auto token_ids = tokenizer.encode(text);
  
  auto output = executor.forward(token_ids);
  
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(output.size(), token_ids.size());
}
```

**P2.2.2: 推理流程测试** (60分钟)
```cpp
TEST(TokenizerExecutorIntegration, InferencePipeline) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  executor.setBackend(std::make_unique<MockBackend>());
  
  std::string prompt = "What is AI?";
  auto input_ids = tokenizer.encode(prompt);
  auto output_ids = executor.generate(input_ids, 20);
  auto output_text = tokenizer.decode(output_ids);
  
  EXPECT_FALSE(output_text.empty());
  EXPECT_GT(output_text.length(), prompt.length());
}
```

**P2.2.3: 批处理测试** (60分钟)
**P2.2.4: 状态同步测试** (60分钟)

**验收标准**:
- ✅ Token IDs 正确转换为 Executor 输入
- ✅ 推理流程完整
- ✅ 批量推理正确
- ✅ 状态一致性

---

### P2.3: Executor + Backend 集成 (4小时)

#### 测试重点
- Executor → LibTorch 推理流程
- Tensor 传递
- 内存管理
- 错误恢复

#### 关键测试用例

**P2.3.1: 推理流程测试** (60分钟)
```cpp
TEST(ExecutorBackendIntegration, ForwardPass) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}");
  
  executor.setBackend(&backend);
  
  std::vector<int> input_ids = {1, 100, 200, 300, 2};
  auto output = executor.forward(input_ids);
  
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(output.size(), input_ids.size());
}
```

**P2.3.2: 内存管理测试** (60分钟)
```cpp
TEST(ExecutorBackendIntegration, MemoryManagement) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}");
  executor.setBackend(&backend);
  
  size_t initial_memory = backend.getMemoryUsage();
  
  // 执行多次推理
  for (int i = 0; i < 100; ++i) {
    std::vector<int> input_ids = {1, 100 + i, 2};
    executor.forward(input_ids);
  }
  
  size_t final_memory = backend.getMemoryUsage();
  
  // 内存增长应该有限
  EXPECT_LT(final_memory - initial_memory, 100 * 1024 * 1024); // < 100MB
}
```

**P2.3.3: 性能测试** (60分钟)
**P2.3.4: 错误恢复测试** (60分钟)

**验收标准**:
- ✅ 推理流程正确
- ✅ 内存使用合理
- ✅ 吞吐量达标（> 10 tokens/s）
- ✅ 错误正确恢复

---

### P2.4: Backend + Qwen3 集成 (3小时)

#### 测试重点
- Qwen3 模型加载到 LibTorch
- 推理正确性验证
- 性能测试
- 长时间稳定性

#### 关键测试用例

**P2.4.1: 模型加载集成** (45分钟)
```cpp
TEST(BackendQwen3Integration, ModelLoading) {
  LibTorchBackend backend;
  
  bool success = backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  
  EXPECT_TRUE(success);
  EXPECT_TRUE(backend.isModelLoaded());
  EXPECT_EQ(backend.getModelName(), "Qwen3");
}
```

**P2.4.2: 推理正确性验证** (45分钟)
```cpp
TEST(BackendQwen3Integration, InferenceCorrectness) {
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  
  // 使用已知输入输出对
  torch::Tensor input = torch::tensor({{1, 100, 200, 2}});
  auto output = backend.forward(input);
  
  EXPECT_EQ(output.sizes()[0], 1); // batch size
  EXPECT_EQ(output.sizes()[1], 4); // sequence length
  EXPECT_GT(output.sizes()[2], 0); // vocab size
}
```

**P2.4.3: 性能测试** (45分钟)
```cpp
TEST(BackendQwen3Integration, Performance) {
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  
  torch::Tensor input = torch::randint(0, 32000, {1, 100});
  
  auto start = std::chrono::high_resolution_clock::now();
  auto output = backend.forward(input);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  // 推理延迟应该合理
  EXPECT_LT(duration.count(), 1000); // < 1秒
}
```

**P2.4.4: 稳定性测试（长时间运行）** (45分钟)
```cpp
TEST(BackendQwen3Integration, LongRunningStability) {
  LibTorchBackend backend;
  backend.loadModel("${CLLM_TEST_MODEL_PATH}/qwen3.pt");
  
  // 运行1000次推理
  for (int i = 0; i < 1000; ++i) {
    torch::Tensor input = torch::randint(0, 32000, {1, 50});
    auto output = backend.forward(input);
    
    EXPECT_FALSE(output.numel() == 0);
    
    if (i % 100 == 0) {
      LOG(INFO) << "Progress: " << i << "/1000";
    }
  }
  
  // 检查内存泄漏
  size_t final_memory = backend.getMemoryUsage();
  EXPECT_LT(final_memory, 10 * 1024 * 1024 * 1024); // < 10GB
}
```

**验收标准**:
- ✅ Qwen3 模型正确加载
- ✅ 推理输出正确
- ✅ 性能达标
- ✅ 长时间运行稳定（无内存泄漏）

---

## ✅ 总体验收标准

### 必须完成

- [ ] P2.1: HTTP + Tokenizer 集成通过
- [ ] P2.2: Tokenizer + Executor 集成通过
- [ ] P2.3: Executor + Backend 集成通过
- [ ] P2.4: Backend + Qwen3 集成通过

### 质量指标

- [ ] 集成测试覆盖率 > 70%
- [ ] 所有测试用例通过率 = 100%
- [ ] 数据流无丢失
- [ ] 错误传播正确

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P2.1: ☐ 完成 / ☐ 失败
- P2.2: ☐ 完成 / ☐ 失败
- P2.3: ☐ 完成 / ☐ 失败
- P2.4: ☐ 完成 / ☐ 失败

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

---

## 🔄 下一步

Phase 2 完成后，通知 Agent-3 启动 Phase 3:

```bash
touch /tmp/cllm_test_locks/phase2.done
echo "✅ Phase 2 完成，Agent-3 可以启动 Phase 3"
```
