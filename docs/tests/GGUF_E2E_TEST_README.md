# GGUF 格式端到端测试指南

## 测试目标

测试 GGUF 格式模型 (`qwen3-0.6b-q8_0.gguf`) 的端到端推理流程：
- 模型格式: GGUF (Q8_0 量化)
- 后端: LibTorch (如果支持) 或 Kylin
- Tokenizer: tokenizers-cpp (HFTokenizer)
- 接口: `/generate` (POST)

## 当前状态

### ✅ 已实现
- GGUF 加载器 (`GGUFLoader`) 已实现
- `ModelLoaderFactory` 支持 GGUF 格式检测
- Kylin Backend 通过 `ModelLoaderFactory` 支持 GGUF

### ⚠️ 待确认
- LibTorch Backend 的 `loadFromGGUF()` 方法尚未实现（TODO）
- 如果使用 LibTorch backend，可能需要先实现 GGUF 支持

## 测试步骤

### 1. 准备环境

确保模型文件存在：
```bash
ls -lh /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf
```

确保 Tokenizer 目录存在：
```bash
ls -d /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B
```

### 2. 编译项目

```bash
cd build
cmake ..
make -j$(nproc)
```

### 3. 运行测试

#### 选项 A: 使用完整测试脚本（推荐）

```bash
chmod +x test_gguf_e2e.sh
./test_gguf_e2e.sh
```

#### 选项 B: 手动测试

**步骤 1: 启动服务器**

使用 Kylin backend（推荐，因为 GGUF 支持已实现）：
```bash
./build/bin/cllm_server \
    --model-path /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf \
    --port 8080 \
    --host 0.0.0.0 \
    --log-level info
```

或者使用 LibTorch backend（如果已实现 GGUF 支持）：
```bash
./build/bin/cllm_server \
    --model-path /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf \
    --use-libtorch \
    --port 8080 \
    --host 0.0.0.0 \
    --log-level info
```

**步骤 2: 测试 /generate 接口**

在另一个终端运行：
```bash
chmod +x test_gguf_curl.sh
./test_gguf_curl.sh
```

或者直接使用 curl：
```bash
curl -X POST http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "hello",
        "max_tokens": 50,
        "temperature": 0.7
    }' | python3 -m json.tool
```

### 4. 预期结果

成功响应示例：
```json
{
    "text": "hello, how can I help you today?",
    "tokens": 8,
    "finish_reason": "length"
}
```

## 故障排查

### 问题 1: 服务器启动失败

**症状**: 服务器无法启动或立即退出

**检查**:
1. 查看日志: `tail -50 /tmp/cllm_server_test.log`
2. 检查模型文件是否存在
3. 检查 Tokenizer 目录是否存在

**可能原因**:
- LibTorch backend 不支持 GGUF（需要实现 `loadFromGGUF()`）
- 模型文件损坏
- 缺少依赖库

**解决方案**:
- 使用 Kylin backend（`--use-libtorch` 参数）
- 检查模型文件完整性

### 问题 2: /generate 返回错误

**症状**: curl 请求返回错误或空响应

**检查**:
1. 服务器是否正常运行: `curl http://localhost:8080/health`
2. 查看服务器日志
3. 检查请求格式是否正确

**可能原因**:
- Tokenizer 未正确加载
- 模型权重未正确加载
- 请求格式错误

### 问题 3: LibTorch backend 不支持 GGUF

**症状**: 使用 `--use-libtorch` 时启动失败

**解决方案**:
1. 暂时使用 Kylin backend（去掉 `--use-libtorch` 参数）
2. 或者实现 `LibTorchBackend::loadFromGGUF()` 方法

## 实现 LibTorch Backend 的 GGUF 支持

如果需要使用 LibTorch backend，需要实现 `loadFromGGUF()` 方法：

```cpp
bool LibTorchBackend::loadFromGGUF(const std::string& ggufPath) {
    // 1. 使用 GGUFLoader 加载权重
    GGUFLoader loader(ggufPath);
    if (!loader.load()) {
        return false;
    }
    
    // 2. 加载权重到 ModelWeights
    ModelWeights weights;
    if (!loader.loadWeights(weights)) {
        return false;
    }
    
    // 3. 转换为 torch::Tensor
    auto torchTensors = convertToTorchTensors(weights, device_);
    
    // 4. 构建 PyTorch 模型并加载权重
    // ... 实现模型构建逻辑
    
    return true;
}
```

## 测试验证清单

- [ ] 服务器成功启动
- [ ] 健康检查端点返回正常
- [ ] `/generate` 接口返回有效响应
- [ ] 生成的文本符合预期（包含 "hello" 的合理回复）
- [ ] 日志中没有错误信息

## 相关文件

- 测试脚本: `test_gguf_e2e.sh`, `test_gguf_curl.sh`
- GGUF 加载器: `src/model/gguf_loader.cpp`
- ModelLoaderFactory: `src/model/loader_interface.cpp`
- Kylin Backend: `src/inference/kylin_backend.cpp`
- LibTorch Backend: `src/inference/libtorch_backend.cpp`

## 下一步

1. 运行测试脚本验证功能
2. 如果 LibTorch backend 不支持，实现 `loadFromGGUF()` 方法
3. 验证生成的文本质量
4. 性能测试和优化
