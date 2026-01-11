# Phase 5: E2E 场景测试阶段 执行计划

**负责Agent**: Agent-5  
**预计耗时**: 6小时  
**依赖**: Phase 4 完成  
**执行时间**: T+68h ~ T+74h  

---

## 📋 阶段目标

进行端到端的真实场景测试，验证系统在实际使用场景中的表现和质量。

---

## 📊 任务清单

| 任务ID | 任务名称 | 耗时 | 依赖 | 状态 |
|--------|---------|------|------|------|
| P5.1.1 | 单轮问答场景 | 90min | P4.3 | ⏳ 待执行 |
| P5.1.2 | 多轮对话场景 | 90min | P5.1.1 | ⏳ 待执行 |
| P5.1.3 | 专业任务场景 | 90min | P5.1.2 | ⏳ 待执行 |
| P5.1.4 | 质量评估 | 90min | P5.1.3 | ⏳ 待执行 |

**总计**: 4个任务，6小时

---

## 📝 详细任务说明

### P5.1.1: 单轮问答场景 (90分钟)

#### 测试重点
- 事实问答
- 推理问答
- 常识问答
- 数学问答

#### 场景测试

**事实问答**:
```cpp
TEST(E2EScenarios, FactualQuestions) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
    {"What is the capital of China?", {"Beijing", "北京"}},
    {"Who invented the telephone?", {"Bell", "Alexander"}},
    {"What is the largest planet in our solar system?", {"Jupiter", "木星"}},
    {"When did World War II end?", {"1945"}},
    {"What is the speed of light?", {"300000", "3×10^8", "299792458"}}
  };
  
  int passed = 0;
  int total = test_cases.size();
  
  for (const auto& [question, expected_keywords] : test_cases) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", question}}}},
      {"max_tokens", 100},
      {"temperature", 0.3} // 低温度，更确定的回答
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    ASSERT_EQ(response.status_code, 200);
    
    auto result = json::parse(response.body);
    std::string answer = result["choices"][0]["message"]["content"];
    
    LOG(INFO) << "Q: " << question;
    LOG(INFO) << "A: " << answer;
    
    // 检查答案中是否包含预期关键词
    bool correct = false;
    for (const auto& keyword : expected_keywords) {
      if (answer.find(keyword) != std::string::npos) {
        correct = true;
        break;
      }
    }
    
    if (correct) {
      passed++;
      LOG(INFO) << "✅ PASS";
    } else {
      LOG(WARNING) << "❌ FAIL (expected keywords: " 
                   << join(expected_keywords, ", ") << ")";
    }
    LOG(INFO) << "---";
  }
  
  double accuracy = static_cast<double>(passed) / total;
  LOG(INFO) << "Factual QA Accuracy: " << accuracy * 100 << "% (" 
            << passed << "/" << total << ")";
  
  // 期望至少60%正确率
  EXPECT_GE(accuracy, 0.6);
  
  server.stop();
}
```

**推理问答**:
```cpp
TEST(E2EScenarios, ReasoningQuestions) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
    {
      "If A is taller than B, and B is taller than C, who is the tallest?",
      {"A"}
    },
    {
      "If all roses are flowers, and some flowers fade quickly, can we conclude all roses fade quickly?",
      {"No", "cannot", "not necessarily"}
    },
    {
      "A train leaves Station A at 60 km/h, and another train leaves Station B (120km away) at 40 km/h towards each other. When do they meet?",
      {"1.2", "72", "minutes"}
    }
  };
  
  int passed = 0;
  int total = test_cases.size();
  
  for (const auto& [question, expected_indicators] : test_cases) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", question}}}},
      {"max_tokens", 200},
      {"temperature", 0.5}
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto result = json::parse(response.body);
    std::string answer = result["choices"][0]["message"]["content"];
    
    LOG(INFO) << "Q: " << question;
    LOG(INFO) << "A: " << answer;
    
    bool correct = false;
    for (const auto& indicator : expected_indicators) {
      if (answer.find(indicator) != std::string::npos) {
        correct = true;
        break;
      }
    }
    
    if (correct) {
      passed++;
      LOG(INFO) << "✅ PASS";
    } else {
      LOG(WARNING) << "❌ FAIL";
    }
    LOG(INFO) << "---";
  }
  
  double accuracy = static_cast<double>(passed) / total;
  LOG(INFO) << "Reasoning QA Accuracy: " << accuracy * 100 << "%";
  
  // 推理问题更难，期望至少40%正确率
  EXPECT_GE(accuracy, 0.4);
  
  server.stop();
}
```

**数学问答**:
```cpp
TEST(E2EScenarios, MathQuestions) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<std::pair<std::string, std::string>> test_cases = {
    {"What is 15 + 27?", "42"},
    {"What is 8 × 7?", "56"},
    {"What is 100 - 37?", "63"},
    {"What is 144 ÷ 12?", "12"},
    {"What is 2^10?", "1024"}
  };
  
  int passed = 0;
  
  for (const auto& [question, expected_answer] : test_cases) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", question}}}},
      {"max_tokens", 50},
      {"temperature", 0.1}
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto result = json::parse(response.body);
    std::string answer = result["choices"][0]["message"]["content"];
    
    LOG(INFO) << "Q: " << question;
    LOG(INFO) << "A: " << answer;
    
    if (answer.find(expected_answer) != std::string::npos) {
      passed++;
      LOG(INFO) << "✅ PASS";
    } else {
      LOG(WARNING) << "❌ FAIL (expected: " << expected_answer << ")";
    }
    LOG(INFO) << "---";
  }
  
  double accuracy = static_cast<double>(passed) / test_cases.size();
  LOG(INFO) << "Math QA Accuracy: " << accuracy * 100 << "%";
  
  EXPECT_GE(accuracy, 0.7); // 简单数学题期望70%+
  
  server.stop();
}
```

**验收标准**:
- ✅ 事实问答正确率 > 60%
- ✅ 推理问答正确率 > 40%
- ✅ 数学问答正确率 > 70%
- ✅ 所有回答格式正确

---

### P5.1.2: 多轮对话场景 (90分钟)

#### 测试重点
- 上下文保持
- 指代消解
- 话题切换
- 对话连贯性

#### 场景测试

**上下文保持**:
```cpp
TEST(E2EScenarios, ContextRetention) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  // 多轮对话
  std::vector<json> messages;
  
  // 第1轮：介绍信息
  messages.push_back({{"role", "user"}, {"content", "My name is Alice and I'm 25 years old."}});
  
  json request1 = {
    {"model", "qwen2-0.5b"},
    {"messages", messages},
    {"max_tokens", 50}
  };
  
  auto response1 = client.post("http://localhost:8080/v1/chat/completions", request1);
  auto result1 = json::parse(response1.body);
  std::string answer1 = result1["choices"][0]["message"]["content"];
  
  messages.push_back({{"role", "assistant"}, {"content", answer1}});
  LOG(INFO) << "Round 1 - User: My name is Alice and I'm 25 years old.";
  LOG(INFO) << "Round 1 - Assistant: " << answer1;
  
  // 第2轮：询问之前提到的信息
  messages.push_back({{"role", "user"}, {"content", "What's my name?"}});
  
  json request2 = {
    {"model", "qwen2-0.5b"},
    {"messages", messages},
    {"max_tokens", 20}
  };
  
  auto response2 = client.post("http://localhost:8080/v1/chat/completions", request2);
  auto result2 = json::parse(response2.body);
  std::string answer2 = result2["choices"][0]["message"]["content"];
  
  LOG(INFO) << "Round 2 - User: What's my name?";
  LOG(INFO) << "Round 2 - Assistant: " << answer2;
  
  // 验证：答案中应包含 "Alice"
  EXPECT_TRUE(answer2.find("Alice") != std::string::npos) 
    << "Context not retained: " << answer2;
  
  // 第3轮：询问年龄
  messages.push_back({{"role", "assistant"}, {"content", answer2}});
  messages.push_back({{"role", "user"}, {"content", "How old am I?"}});
  
  json request3 = {
    {"model", "qwen2-0.5b"},
    {"messages", messages},
    {"max_tokens", 20}
  };
  
  auto response3 = client.post("http://localhost:8080/v1/chat/completions", request3);
  auto result3 = json::parse(response3.body);
  std::string answer3 = result3["choices"][0]["message"]["content"];
  
  LOG(INFO) << "Round 3 - User: How old am I?";
  LOG(INFO) << "Round 3 - Assistant: " << answer3;
  
  // 验证：答案中应包含 "25"
  EXPECT_TRUE(answer3.find("25") != std::string::npos)
    << "Context not retained: " << answer3;
  
  server.stop();
}
```

**指代消解**:
```cpp
TEST(E2EScenarios, Coreference) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<json> messages;
  
  // 第1轮：引入主题
  messages.push_back({{"role", "user"}, {"content", "Tell me about Python programming language."}});
  
  json request1 = {
    {"model", "qwen2-0.5b"},
    {"messages", messages},
    {"max_tokens", 100}
  };
  
  auto response1 = client.post("http://localhost:8080/v1/chat/completions", request1);
  auto result1 = json::parse(response1.body);
  std::string answer1 = result1["choices"][0]["message"]["content"];
  
  messages.push_back({{"role", "assistant"}, {"content", answer1}});
  
  // 第2轮：使用指代词 "it"
  messages.push_back({{"role", "user"}, {"content", "What is it mainly used for?"}});
  
  json request2 = {
    {"model", "qwen2-0.5b"},
    {"messages", messages},
    {"max_tokens", 100}
  };
  
  auto response2 = client.post("http://localhost:8080/v1/chat/completions", request2);
  auto result2 = json::parse(response2.body);
  std::string answer2 = result2["choices"][0]["message"]["content"];
  
  LOG(INFO) << "User: What is it mainly used for?";
  LOG(INFO) << "Assistant: " << answer2;
  
  // 验证：答案应该与Python相关
  bool relevant = answer2.find("Python") != std::string::npos ||
                  answer2.find("programming") != std::string::npos ||
                  answer2.find("web") != std::string::npos ||
                  answer2.find("data") != std::string::npos ||
                  answer2.find("AI") != std::string::npos;
  
  EXPECT_TRUE(relevant) << "Failed to resolve coreference";
  
  server.stop();
}
```

**话题切换**:
```cpp
TEST(E2EScenarios, TopicSwitch) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<json> messages;
  
  // 话题1：天气
  messages.push_back({{"role", "user"}, {"content", "What's the weather like today?"}});
  
  json request1 = {{"model", "qwen2-0.5b"}, {"messages", messages}, {"max_tokens", 50}};
  auto response1 = client.post("http://localhost:8080/v1/chat/completions", request1);
  auto result1 = json::parse(response1.body);
  messages.push_back({{"role", "assistant"}, {"content", result1["choices"][0]["message"]["content"]}});
  
  // 话题切换到：编程
  messages.push_back({{"role", "user"}, {"content", "By the way, can you help me write a Python function?"}});
  
  json request2 = {{"model", "qwen2-0.5b"}, {"messages", messages}, {"max_tokens", 100}};
  auto response2 = client.post("http://localhost:8080/v1/chat/completions", request2);
  auto result2 = json::parse(response2.body);
  std::string answer2 = result2["choices"][0]["message"]["content"];
  
  LOG(INFO) << "Assistant (after topic switch): " << answer2;
  
  // 验证：应该能够处理话题切换
  bool handled = answer2.find("Python") != std::string::npos ||
                 answer2.find("function") != std::string::npos ||
                 answer2.find("def") != std::string::npos ||
                 answer2.find("code") != std::string::npos;
  
  EXPECT_TRUE(handled) << "Failed to handle topic switch";
  
  server.stop();
}
```

**验收标准**:
- ✅ 上下文保持正确
- ✅ 指代消解正确
- ✅ 话题切换处理正确
- ✅ 对话连贯性良好

---

### P5.1.3: 专业任务场景 (90分钟)

#### 测试重点
- 代码生成
- 文本摘要
- 翻译
- 文档撰写

#### 场景测试

**代码生成**:
```cpp
TEST(E2EScenarios, CodeGeneration) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::vector<std::string> prompts = {
    "Write a Python function to calculate Fibonacci numbers.",
    "Write a JavaScript function to check if a string is a palindrome.",
    "Write a C++ function to sort an array using quicksort."
  };
  
  for (const auto& prompt : prompts) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", prompt}}}},
      {"max_tokens", 300}
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto result = json::parse(response.body);
    std::string code = result["choices"][0]["message"]["content"];
    
    LOG(INFO) << "Prompt: " << prompt;
    LOG(INFO) << "Generated code:\n" << code;
    LOG(INFO) << "---";
    
    // 验证代码包含基本要素
    bool has_function = code.find("def ") != std::string::npos ||
                       code.find("function ") != std::string::npos ||
                       code.find("void ") != std::string::npos;
    
    EXPECT_TRUE(has_function) << "No function definition found";
  }
  
  server.stop();
}
```

**文本摘要**:
```cpp
TEST(E2EScenarios, TextSummarization) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  std::string long_text = R"(
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" 
    had previously been used to describe machines that mimic and display "human" 
    cognitive skills that are associated with the human mind, such as "learning" 
    and "problem-solving". This definition has since been rejected by major AI 
    researchers who now describe AI in terms of rationality and acting rationally, 
    which does not limit how intelligence can be articulated.
  )";
  
  json request = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Please summarize the following text in 2-3 sentences: " + long_text}}}},
    {"max_tokens", 150}
  };
  
  auto response = client.post("http://localhost:8080/v1/chat/completions", request);
  auto result = json::parse(response.body);
  std::string summary = result["choices"][0]["message"]["content"];
  
  LOG(INFO) << "Original length: " << long_text.length();
  LOG(INFO) << "Summary: " << summary;
  LOG(INFO) << "Summary length: " << summary.length();
  
  // 验证摘要更短
  EXPECT_LT(summary.length(), long_text.length() * 0.5);
  
  // 验证摘要包含关键词
  bool has_keywords = summary.find("AI") != std::string::npos ||
                      summary.find("artificial intelligence") != std::string::npos ||
                      summary.find("intelligence") != std::string::npos;
  
  EXPECT_TRUE(has_keywords) << "Summary doesn't contain key concepts";
  
  server.stop();
}
```

**翻译**:
```cpp
TEST(E2EScenarios, Translation) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  // 英译中
  json request_en_to_zh = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "Translate to Chinese: Artificial Intelligence is changing the world."}}}},
    {"max_tokens", 100}
  };
  
  auto response_en_to_zh = client.post("http://localhost:8080/v1/chat/completions", request_en_to_zh);
  auto result_en_to_zh = json::parse(response_en_to_zh.body);
  std::string translation_zh = result_en_to_zh["choices"][0]["message"]["content"];
  
  LOG(INFO) << "EN->ZH: " << translation_zh;
  
  // 验证中文翻译包含中文字符
  bool has_chinese = std::any_of(translation_zh.begin(), translation_zh.end(), 
    [](char c) { return static_cast<unsigned char>(c) > 127; });
  
  EXPECT_TRUE(has_chinese) << "Translation doesn't contain Chinese characters";
  
  // 中译英
  json request_zh_to_en = {
    {"model", "qwen2-0.5b"},
    {"messages", {{{"role", "user"}, {"content", "翻译成英文：人工智能正在改变世界。"}}}},
    {"max_tokens", 100}
  };
  
  auto response_zh_to_en = client.post("http://localhost:8080/v1/chat/completions", request_zh_to_en);
  auto result_zh_to_en = json::parse(response_zh_to_en.body);
  std::string translation_en = result_zh_to_en["choices"][0]["message"]["content"];
  
  LOG(INFO) << "ZH->EN: " << translation_en;
  
  // 验证英文翻译包含关键词
  bool has_keywords = translation_en.find("AI") != std::string::npos ||
                      translation_en.find("artificial") != std::string::npos ||
                      translation_en.find("intelligence") != std::string::npos ||
                      translation_en.find("world") != std::string::npos;
  
  EXPECT_TRUE(has_keywords) << "Translation doesn't contain key words";
  
  server.stop();
}
```

**验收标准**:
- ✅ 代码生成格式正确
- ✅ 摘要长度合理
- ✅ 翻译包含正确语言
- ✅ 专业任务完成度良好

---

### P5.1.4: 质量评估 (90分钟)

#### 评估维度
- 准确性（Accuracy）
- 流畅性（Fluency）
- 相关性（Relevance）
- 完整性（Completeness）

#### 评估方法

```cpp
TEST(E2EScenarios, QualityEvaluation) {
  cLLMServer server;
  server.initialize("${CLLM_TEST_MODEL_PATH}");
  server.start();
  
  HTTPClient client;
  
  // 测试集
  std::vector<std::tuple<std::string, std::string, std::vector<std::string>>> test_set = {
    {
      "Question", 
      "What is machine learning?",
      {"machine", "learn", "data", "algorithm", "model"}
    },
    {
      "Question",
      "Explain quantum computing.",
      {"quantum", "qubit", "superposition", "computing"}
    },
    // ... 更多测试用例
  };
  
  struct QualityScore {
    double accuracy = 0.0;
    double fluency = 0.0;
    double relevance = 0.0;
    double completeness = 0.0;
  };
  
  std::vector<QualityScore> scores;
  
  for (const auto& [type, question, keywords] : test_set) {
    json request = {
      {"model", "qwen2-0.5b"},
      {"messages", {{{"role", "user"}, {"content", question}}}},
      {"max_tokens", 200}
    };
    
    auto response = client.post("http://localhost:8080/v1/chat/completions", request);
    auto result = json::parse(response.body);
    std::string answer = result["choices"][0]["message"]["content"];
    
    QualityScore score;
    
    // 1. 准确性：检查关键词覆盖
    int keywords_found = 0;
    for (const auto& keyword : keywords) {
      if (answer.find(keyword) != std::string::npos) {
        keywords_found++;
      }
    }
    score.accuracy = static_cast<double>(keywords_found) / keywords.size();
    
    // 2. 流畅性：检查基本语法（简单启发式）
    bool has_punctuation = answer.find(".") != std::string::npos || 
                           answer.find("。") != std::string::npos;
    bool reasonable_length = answer.length() > 20 && answer.length() < 1000;
    score.fluency = (has_punctuation && reasonable_length) ? 1.0 : 0.5;
    
    // 3. 相关性：答案长度合理
    score.relevance = (answer.length() > 30) ? 1.0 : 0.5;
    
    // 4. 完整性：答案不是太短
    score.completeness = (answer.length() > 50) ? 1.0 : 0.5;
    
    scores.push_back(score);
    
    LOG(INFO) << "Question: " << question;
    LOG(INFO) << "Answer: " << answer;
    LOG(INFO) << "Scores - Accuracy: " << score.accuracy 
              << ", Fluency: " << score.fluency
              << ", Relevance: " << score.relevance
              << ", Completeness: " << score.completeness;
    LOG(INFO) << "---";
  }
  
  // 计算平均分
  double avg_accuracy = 0.0, avg_fluency = 0.0, avg_relevance = 0.0, avg_completeness = 0.0;
  
  for (const auto& score : scores) {
    avg_accuracy += score.accuracy;
    avg_fluency += score.fluency;
    avg_relevance += score.relevance;
    avg_completeness += score.completeness;
  }
  
  int n = scores.size();
  avg_accuracy /= n;
  avg_fluency /= n;
  avg_relevance /= n;
  avg_completeness /= n;
  
  double overall_score = (avg_accuracy + avg_fluency + avg_relevance + avg_completeness) / 4.0;
  
  LOG(INFO) << "===== Quality Evaluation Results =====";
  LOG(INFO) << "Average Accuracy: " << avg_accuracy * 5 << " / 5.0";
  LOG(INFO) << "Average Fluency: " << avg_fluency * 5 << " / 5.0";
  LOG(INFO) << "Average Relevance: " << avg_relevance * 5 << " / 5.0";
  LOG(INFO) << "Average Completeness: " << avg_completeness * 5 << " / 5.0";
  LOG(INFO) << "Overall Score: " << overall_score * 5 << " / 5.0";
  
  // 期望总分 > 4.0/5.0
  EXPECT_GT(overall_score * 5, 4.0);
  
  server.stop();
}
```

**验收标准**:
- ✅ 平均准确性 > 0.7
- ✅ 平均流畅性 > 0.8
- ✅ 平均相关性 > 0.8
- ✅ 平均完整性 > 0.7
- ✅ 总体评分 > 4.0/5.0

---

## ✅ 总体验收标准

### 必须完成

- [ ] P5.1.1: 单轮问答场景通过
- [ ] P5.1.2: 多轮对话场景通过
- [ ] P5.1.3: 专业任务场景通过
- [ ] P5.1.4: 质量评估达标

### 质量指标

- [ ] 事实问答正确率 > 60%
- [ ] 推理问答正确率 > 40%
- [ ] 数学问答正确率 > 70%
- [ ] 上下文保持正确
- [ ] 代码生成格式正确
- [ ] 翻译质量良好
- [ ] 总体评分 > 4.0/5.0

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P5.1.1: ☐ 完成 / ☐ 失败
- P5.1.2: ☐ 完成 / ☐ 失败
- P5.1.3: ☐ 完成 / ☐ 失败
- P5.1.4: ☐ 完成 / ☐ 失败

**质量得分**:
- 事实问答正确率: ________%
- 推理问答正确率: ________%
- 数学问答正确率: ________%
- 总体质量评分: ________ / 5.0

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

---

## 🎉 测试完成

Phase 5 是最后一个测试阶段。完成后：

```bash
touch /tmp/cllm_test_locks/phase5.done
touch /tmp/cllm_test_locks/all_phases.done

# 生成最终测试报告
python3 scripts/generate_final_report.py

echo "========================================="
echo "🎉 所有测试阶段完成！"
echo "========================================="
echo "总耗时: 74小时"
echo "测试阶段: 6个"
echo "测试任务: 72个"
echo ""
echo "最终报告: test_reports/final_report.md"
echo "========================================="
```

---

**✅ cLLM 分阶段集成测试全部完成！**
