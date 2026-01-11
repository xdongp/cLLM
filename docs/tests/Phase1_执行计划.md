# Phase 1: 单元测试阶段 执行计划

**负责Agent**: Agent-1  
**预计耗时**: 12小时  
**依赖**: Phase 0 完成  
**执行时间**: T+3h ~ T+15h  

---

## 📋 阶段目标

对5个核心模块（HTTP Server、HFTokenizer、ModelExecutor、LibTorch Backend、Qwen3 Model）进行独立的单元测试，验证每个模块的基本功能正确性。

---

## 📊 任务清单

| 子阶段 | 任务数 | 耗时 | 优先级 | 状态 |
|--------|--------|------|--------|------|
| P1.1: HTTP Server 单元测试 | 4 | 2h | 高 | ⏳ 待执行 |
| P1.2: HFTokenizer 单元测试 | 4 | 2h | 高 | ⏳ 待执行 |
| P1.3: ModelExecutor 单元测试 | 4 | 3h | 高 | ⏳ 待执行 |
| P1.4: LibTorch Backend 单元测试 | 4 | 3h | 高 | ⏳ 待执行 |
| P1.5: Qwen3 Model 单元测试 | 3 | 2h | 高 | ⏳ 待执行 |

**总计**: 19个任务，12小时

---

## 📝 详细任务说明

### P1.1: HTTP Server 单元测试 (2小时)

#### P1.1.1: 路由注册和匹配测试 (30分钟)

**目标**: 验证HTTP路由的注册和匹配功能

**测试用例**:
```cpp
TEST(HTTPServerTest, RouteRegistration) {
  HTTPServer server;
  
  // 注册路由
  server.registerRoute("/api/health", HTTPMethod::GET, healthHandler);
  server.registerRoute("/v1/chat/completions", HTTPMethod::POST, chatHandler);
  
  // 验证路由存在
  EXPECT_TRUE(server.hasRoute("/api/health"));
  EXPECT_TRUE(server.hasRoute("/v1/chat/completions"));
  EXPECT_FALSE(server.hasRoute("/invalid/route"));
}

TEST(HTTPServerTest, RouteMatching) {
  HTTPServer server;
  server.registerRoute("/api/users/:id", HTTPMethod::GET, userHandler);
  
  // 测试路径参数匹配
  auto match = server.matchRoute("/api/users/123");
  EXPECT_TRUE(match.matched);
  EXPECT_EQ(match.params["id"], "123");
}

TEST(HTTPServerTest, MethodFiltering) {
  HTTPServer server;
  server.registerRoute("/api/data", HTTPMethod::GET, getHandler);
  server.registerRoute("/api/data", HTTPMethod::POST, postHandler);
  
  // 验证方法过滤
  EXPECT_TRUE(server.matchRoute("/api/data", HTTPMethod::GET).matched);
  EXPECT_TRUE(server.matchRoute("/api/data", HTTPMethod::POST).matched);
  EXPECT_FALSE(server.matchRoute("/api/data", HTTPMethod::DELETE).matched);
}
```

**执行命令**:
```bash
cd build/bin
./test_http_server --gtest_filter="HTTPServerTest.Route*"
```

---

#### P1.1.2: 请求解析测试 (30分钟)

**目标**: 验证HTTP请求的解析功能

**测试用例**:
```cpp
TEST(HTTPRequestTest, QueryParamsParsing) {
  std::string url = "/api/search?q=test&limit=10&offset=0";
  HTTPRequest request = HTTPRequest::parse(url);
  
  EXPECT_EQ(request.path, "/api/search");
  EXPECT_EQ(request.queryParams["q"], "test");
  EXPECT_EQ(request.queryParams["limit"], "10");
  EXPECT_EQ(request.queryParams["offset"], "0");
}

TEST(HTTPRequestTest, BodyParsing) {
  std::string body = R"({"model":"qwen2","prompt":"Hello"})";
  HTTPRequest request;
  request.body = body;
  
  auto json = request.parseJSON();
  EXPECT_EQ(json["model"], "qwen2");
  EXPECT_EQ(json["prompt"], "Hello");
}

TEST(HTTPRequestTest, HeadersParsing) {
  HTTPRequest request;
  request.headers["Content-Type"] = "application/json";
  request.headers["Authorization"] = "Bearer token123";
  
  EXPECT_EQ(request.getHeader("Content-Type"), "application/json");
  EXPECT_EQ(request.getHeader("Authorization"), "Bearer token123");
}
```

---

#### P1.1.3: 响应构建测试 (30分钟)

**测试用例**:
```cpp
TEST(HTTPResponseTest, StatusCode) {
  HTTPResponse response;
  response.setStatus(200);
  EXPECT_EQ(response.statusCode, 200);
  EXPECT_EQ(response.statusText, "OK");
  
  response.setStatus(404);
  EXPECT_EQ(response.statusCode, 404);
  EXPECT_EQ(response.statusText, "Not Found");
}

TEST(HTTPResponseTest, JSONResponse) {
  HTTPResponse response;
  json data = {{"status", "success"}, {"data", "test"}};
  response.setJSON(data);
  
  EXPECT_EQ(response.getHeader("Content-Type"), "application/json");
  EXPECT_TRUE(response.body.find("\"status\":\"success\"") != std::string::npos);
}

TEST(HTTPResponseTest, HeadersSetting) {
  HTTPResponse response;
  response.setHeader("X-Custom-Header", "value");
  response.setHeader("Cache-Control", "no-cache");
  
  EXPECT_EQ(response.getHeader("X-Custom-Header"), "value");
  EXPECT_EQ(response.getHeader("Cache-Control"), "no-cache");
}
```

---

#### P1.1.4: 错误处理测试 (30分钟)

**测试用例**:
```cpp
TEST(HTTPServerTest, Handle404) {
  HTTPServer server;
  HTTPRequest request;
  request.path = "/invalid/route";
  
  auto response = server.handle(request);
  EXPECT_EQ(response.statusCode, 404);
  EXPECT_TRUE(response.body.find("Not Found") != std::string::npos);
}

TEST(HTTPServerTest, Handle500) {
  HTTPServer server;
  server.registerRoute("/api/error", HTTPMethod::GET, [](const HTTPRequest&) {
    throw std::runtime_error("Internal error");
  });
  
  HTTPRequest request;
  request.path = "/api/error";
  
  auto response = server.handle(request);
  EXPECT_EQ(response.statusCode, 500);
}

TEST(HTTPServerTest, HandleTimeout) {
  HTTPServer server;
  server.setTimeout(1000); // 1秒超时
  
  server.registerRoute("/api/slow", HTTPMethod::GET, [](const HTTPRequest&) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return HTTPResponse();
  });
  
  HTTPRequest request;
  request.path = "/api/slow";
  
  auto response = server.handle(request);
  EXPECT_EQ(response.statusCode, 408); // Request Timeout
}
```

**P1.1 验收标准**:
- ✅ 所有路由测试通过
- ✅ 请求解析正确
- ✅ 响应构建符合HTTP规范
- ✅ 错误处理健壮

---

### P1.2: HFTokenizer 单元测试 (2小时)

#### P1.2.1: 模型加载测试 (30分钟)

**测试用例**:
```cpp
TEST(HFTokenizerTest, LoadValidModel) {
  HFTokenizer tokenizer;
  bool success = tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  EXPECT_TRUE(success);
  EXPECT_TRUE(tokenizer.isLoaded());
  EXPECT_GT(tokenizer.vocabSize(), 0);
}

TEST(HFTokenizerTest, LoadInvalidPath) {
  HFTokenizer tokenizer;
  bool success = tokenizer.load("/invalid/path");
  
  EXPECT_FALSE(success);
  EXPECT_FALSE(tokenizer.isLoaded());
}

TEST(HFTokenizerTest, ModelType) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  std::string type = tokenizer.modelType();
  EXPECT_FALSE(type.empty());
  EXPECT_TRUE(type == "BPE" || type == "WordPiece" || type == "Unigram");
}
```

---

#### P1.2.2: 编码测试（多语言） (30分钟)

**测试用例**:
```cpp
TEST(HFTokenizerTest, EncodeEnglish) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  auto ids = tokenizer.encode("Hello, world!");
  EXPECT_GT(ids.size(), 0);
  EXPECT_LT(ids.size(), 10); // 合理的token数量
}

TEST(HFTokenizerTest, EncodeChinese) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  auto ids = tokenizer.encode("你好，世界！");
  EXPECT_GT(ids.size(), 0);
}

TEST(HFTokenizerTest, EncodeMixed) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  auto ids = tokenizer.encode("Hello 世界!");
  EXPECT_GT(ids.size(), 0);
}

TEST(HFTokenizerTest, EncodeSpecialChars) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  auto ids1 = tokenizer.encode("😀🎉🚀");
  auto ids2 = tokenizer.encode("@#$%^&*()");
  
  EXPECT_GT(ids1.size(), 0);
  EXPECT_GT(ids2.size(), 0);
}
```

---

#### P1.2.3: 解码测试 (30分钟)

**测试用例**:
```cpp
TEST(HFTokenizerTest, DecodeBasic) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  std::string original = "Hello, world!";
  auto ids = tokenizer.encode(original);
  auto decoded = tokenizer.decode(ids);
  
  EXPECT_EQ(decoded, original);
}

TEST(HFTokenizerTest, DecodeEmpty) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  std::vector<int> empty_ids;
  auto decoded = tokenizer.decode(empty_ids);
  
  EXPECT_TRUE(decoded.empty());
}

TEST(HFTokenizerTest, DecodeSpecialTokens) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  // 包含特殊token的ID序列
  std::vector<int> ids = {tokenizer.bosTokenId(), 100, 200, tokenizer.eosTokenId()};
  auto decoded = tokenizer.decode(ids);
  
  EXPECT_FALSE(decoded.empty());
}
```

---

#### P1.2.4: 批量处理测试 (30分钟)

**测试用例**:
```cpp
TEST(HFTokenizerTest, BatchEncode) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  std::vector<std::string> texts = {
    "Hello, world!",
    "How are you?",
    "Machine learning is amazing."
  };
  
  auto batch_ids = tokenizer.encodeBatch(texts);
  
  EXPECT_EQ(batch_ids.size(), 3);
  for (const auto& ids : batch_ids) {
    EXPECT_GT(ids.size(), 0);
  }
}

TEST(HFTokenizerTest, BatchDecode) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  std::vector<std::string> originals = {
    "First text",
    "Second text",
    "Third text"
  };
  
  auto batch_ids = tokenizer.encodeBatch(originals);
  auto decoded = tokenizer.decodeBatch(batch_ids);
  
  EXPECT_EQ(decoded.size(), 3);
  for (size_t i = 0; i < decoded.size(); ++i) {
    EXPECT_EQ(decoded[i], originals[i]);
  }
}

TEST(HFTokenizerTest, BatchPerformance) {
  HFTokenizer tokenizer;
  tokenizer.load("${CLLM_TEST_MODEL_PATH}");
  
  // 生成100个测试文本
  std::vector<std::string> texts(100, "This is a test sentence.");
  
  auto start = std::chrono::high_resolution_clock::now();
  auto batch_ids = tokenizer.encodeBatch(texts);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  EXPECT_LT(duration.count(), 1000); // 应该在1秒内完成
  EXPECT_EQ(batch_ids.size(), 100);
}
```

**P1.2 验收标准**:
- ✅ 模型正确加载
- ✅ 多语言编码正确
- ✅ 编解码可逆
- ✅ 批量处理性能达标

---

### P1.3: ModelExecutor 单元测试 (3小时)

#### P1.3.1: 初始化测试 (45分钟)

**测试用例**:
```cpp
TEST(ModelExecutorTest, Initialize) {
  ModelExecutor executor;
  ExecutorConfig config;
  config.max_batch_size = 8;
  config.max_seq_len = 2048;
  
  bool success = executor.initialize(config);
  EXPECT_TRUE(success);
  EXPECT_TRUE(executor.isReady());
}

TEST(ModelExecutorTest, LoadConfig) {
  ModelExecutor executor;
  executor.loadConfigFromFile("config/executor_config.yaml");
  
  EXPECT_GT(executor.getMaxBatchSize(), 0);
  EXPECT_GT(executor.getMaxSeqLen(), 0);
}

TEST(ModelExecutorTest, InvalidConfig) {
  ModelExecutor executor;
  ExecutorConfig config;
  config.max_batch_size = 0; // 无效配置
  
  bool success = executor.initialize(config);
  EXPECT_FALSE(success);
}
```

---

#### P1.3.2: 推理接口测试 (45分钟)

**测试用例**:
```cpp
TEST(ModelExecutorTest, Forward) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  // 使用 Mock Backend
  executor.setBackend(std::make_unique<MockBackend>());
  
  std::vector<int> input_ids = {1, 100, 200, 300, 2};
  auto output = executor.forward(input_ids);
  
  EXPECT_FALSE(output.empty());
  EXPECT_EQ(output.size(), input_ids.size());
}

TEST(ModelExecutorTest, Generate) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  executor.setBackend(std::make_unique<MockBackend>());
  
  std::vector<int> prompt_ids = {1, 100, 200};
  auto generated = executor.generate(prompt_ids, /*max_new_tokens=*/10);
  
  EXPECT_GT(generated.size(), prompt_ids.size());
  EXPECT_LE(generated.size(), prompt_ids.size() + 10);
}

TEST(ModelExecutorTest, BatchForward) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  executor.setBackend(std::make_unique<MockBackend>());
  
  std::vector<std::vector<int>> batch_inputs = {
    {1, 100, 200, 2},
    {1, 150, 250, 350, 2},
    {1, 180, 2}
  };
  
  auto batch_outputs = executor.forwardBatch(batch_inputs);
  
  EXPECT_EQ(batch_outputs.size(), 3);
}
```

---

#### P1.3.3: 批处理管理测试 (45分钟)

**测试用例**:
```cpp
TEST(ModelExecutorTest, BatchManagement) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  // 添加请求
  RequestId id1 = executor.addRequest({1, 100, 200});
  RequestId id2 = executor.addRequest({1, 150, 250});
  
  EXPECT_NE(id1, id2);
  EXPECT_TRUE(executor.hasRequest(id1));
  EXPECT_TRUE(executor.hasRequest(id2));
}

TEST(ModelExecutorTest, BatchScheduling) {
  ModelExecutor executor;
  ExecutorConfig config;
  config.max_batch_size = 4;
  executor.initialize(config);
  
  // 添加多个请求
  for (int i = 0; i < 8; ++i) {
    executor.addRequest({1, 100 + i, 2});
  }
  
  // 获取下一批
  auto batch = executor.getNextBatch();
  EXPECT_LE(batch.size(), 4);
}

TEST(ModelExecutorTest, RequestCompletion) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  RequestId id = executor.addRequest({1, 100, 2});
  executor.markComplete(id);
  
  EXPECT_FALSE(executor.hasRequest(id));
}
```

---

#### P1.3.4: 状态管理测试 (45分钟)

**测试用例**:
```cpp
TEST(ModelExecutorTest, StateReset) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  executor.addRequest({1, 100, 2});
  executor.addRequest({1, 200, 2});
  
  executor.reset();
  
  EXPECT_EQ(executor.getPendingRequestCount(), 0);
}

TEST(ModelExecutorTest, StateSave) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  executor.addRequest({1, 100, 2});
  
  auto state = executor.saveState();
  EXPECT_FALSE(state.empty());
}

TEST(ModelExecutorTest, StateRestore) {
  ModelExecutor executor;
  executor.initialize(defaultConfig());
  
  executor.addRequest({1, 100, 2});
  auto state = executor.saveState();
  
  executor.reset();
  EXPECT_EQ(executor.getPendingRequestCount(), 0);
  
  executor.restoreState(state);
  EXPECT_GT(executor.getPendingRequestCount(), 0);
}
```

**P1.3 验收标准**:
- ✅ 初始化和配置加载正确
- ✅ 推理接口工作正常
- ✅ 批处理管理正确
- ✅ 状态管理可靠

---

### P1.4: LibTorch Backend 单元测试 (3小时)

_(省略详细测试用例，结构类似，包含4个子任务)_

**P1.4.1**: 模型加载测试 (45分钟)  
**P1.4.2**: Tensor 操作测试 (45分钟)  
**P1.4.3**: 前向推理测试 (45分钟)  
**P1.4.4**: 内存管理测试 (45分钟)

---

### P1.5: Qwen3 Model 单元测试 (2小时)

_(省略详细测试用例，包含3个子任务)_

**P1.5.1**: 模型加载测试 (30分钟)  
**P1.5.2**: Tokenizer 兼容性测试 (30分钟)  
**P1.5.3**: 基本推理测试 (60分钟)

---

## ✅ 总体验收标准

### 必须完成

- [ ] HTTP Server: 所有4个子任务通过
- [ ] HFTokenizer: 所有4个子任务通过
- [ ] ModelExecutor: 所有4个子任务通过
- [ ] LibTorch Backend: 所有4个子任务通过
- [ ] Qwen3 Model: 所有3个子任务通过

### 质量指标

- [ ] 单元测试覆盖率 > 80%
- [ ] 所有测试用例通过率 = 100%
- [ ] 无内存泄漏
- [ ] 无编译警告

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P1.1 HTTP Server: ☐ 完成 / ☐ 失败
- P1.2 HFTokenizer: ☐ 完成 / ☐ 失败
- P1.3 ModelExecutor: ☐ 完成 / ☐ 失败
- P1.4 LibTorch Backend: ☐ 完成 / ☐ 失败
- P1.5 Qwen3 Model: ☐ 完成 / ☐ 失败

**测试统计**:
- 总测试用例数: ________
- 通过: ________
- 失败: ________
- 跳过: ________

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

---

## 🔄 下一步

Phase 1 完成后，通知 Agent-2 启动 Phase 2:

```bash
touch /tmp/cllm_test_locks/phase1.done
echo "✅ Phase 1 完成，Agent-2 可以启动 Phase 2"
```
