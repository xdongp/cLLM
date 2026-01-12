# 硬编码值分类整理

## 1. 服务器配置

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/main.cpp | 服务器主机 | 字符串 | "0.0.0.0" | server.host |
| src/main.cpp | 服务器端口 | 数字 | 8080 | server.port |
| src/client/cllm_client.py | 默认服务器URL | 字符串 | "http://localhost:18080" | client.default_server_url |
| src/main.cpp | 默认量化类型 | 字符串 | "fp16" | model.quantization |
| src/main.cpp | 默认日志级别 | 字符串 | "info" | logging.level |

## 2. API端点配置

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/main.cpp | 健康检查端点 | 字符串 | "/health" | api.endpoints.health.path |
| src/main.cpp | 生成端点 | 字符串 | "/generate" | api.endpoints.generate.path |
| src/main.cpp | 流式生成端点 | 字符串 | "/generate_stream" | api.endpoints.generate_stream.path |
| src/main.cpp | 编码端点 | 字符串 | "/encode" | api.endpoints.encode.path |
| src/http/generate_endpoint.cpp | 端点名称、路径、方法 | 字符串 | "generate", "/generate", "POST" | api.endpoints.generate.{name,path,method} |

## 3. 模型推理配置

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/main.cpp | 默认最大批大小 | 数字 | 8 | model.max_batch_size |
| src/main.cpp | 默认上下文长度 | 数字 | 2048 | model.max_context_length |
| src/inference/libtorch_backend.cpp | 候选序列长度 | 数组 | {8, 16, 32, 64, 128, 256} | backend.libtorch.seq_len_candidates |
| src/inference/libtorch_backend.cpp | 备选序列长度 | 数字 | 8 | backend.libtorch.fallback_seq_len |

## 4. 超时和限制

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/http/generate_endpoint.cpp | 最大输入令牌数 | 数字 | 120 | api.limits.max_input_tokens |
| src/http/generate_endpoint.cpp | 最小超时时间 | 数字 | 60.0f | api.timeouts.min |
| src/http/generate_endpoint.cpp | 最大超时时间 | 数字 | 600.0f | api.timeouts.max |
| src/http/generate_endpoint.cpp | 超时时间系数 | 数字 | 10.0f | api.timeouts.token_factor |
| src/client/cllm_client.py | 健康检查超时 | 数字 | 5 | client.timeouts.health_check |
| src/client/cllm_client.py | 请求超时 | 数字 | 60 | client.timeouts.request |

## 5. 默认参数

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/http/generate_endpoint.cpp | 默认提示词 | 字符串 | "" | api.defaults.prompt |
| src/http/generate_endpoint.cpp | 默认最大令牌数 | 数字 | 3 | api.defaults.max_tokens |
| src/http/generate_endpoint.cpp | 默认温度 | 数字 | 0.7f | api.defaults.temperature |
| src/http/generate_endpoint.cpp | 默认top_p | 数字 | 0.9f | api.defaults.top_p |
| src/client/cllm_client.py | 默认最大令牌数 | 数字 | 50 | client.defaults.max_tokens |
| src/client/cllm_client.py | 默认温度 | 数字 | 0.7 | client.defaults.temperature |
| src/client/cllm_client.py | 默认top_p | 数字 | 0.9 | client.defaults.top_p |
| src/client/cllm_client.py | 默认top_k | 数字 | 50 | client.defaults.top_k |

## 6. 其他配置

| 文件 | 硬编码值 | 类型 | 当前值 | 建议配置键 |
|------|----------|------|--------|------------|
| src/http/generate_endpoint.cpp | 响应内容类型 | 字符串 | "application/json" | api.response.content_type.json |
| src/http/generate_endpoint.cpp | 流式响应内容类型 | 字符串 | "text/event-stream" | api.response.content_type.stream |
| src/http/generate_endpoint.cpp | Cache-Control头 | 字符串 | "no-cache" | api.response.headers.cache_control |
| src/http/generate_endpoint.cpp | Connection头 | 字符串 | "keep-alive" | api.response.headers.connection |
