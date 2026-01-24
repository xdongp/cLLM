# 动态 Batch Size 策略测试方案

## 测试目标

对比三种 batch size 调整策略（static/dynamic/hybrid）在不同负载下的性能表现，形成推荐配置。

## 策略说明

### 1. Static 策略
- **特点**: 使用固定的 batch size，不进行动态调整
- **适用场景**: 负载稳定、资源受限的环境
- **配置**: `strategy: "static"`, `fixed_batch_size: 8`

### 2. Dynamic 策略
- **特点**: 指数增加 + 二分下降探测，持续在线调整
- **适用场景**: 负载动态变化、追求最优性能
- **配置**: `strategy: "dynamic"`, `autoAdjustEnabled: true`

### 3. Hybrid 策略
- **特点**: 指数增加 + 二分下降探测，探测完成后锁定
- **适用场景**: 启动时优化，运行时稳定
- **配置**: `strategy: "hybrid"`, `autoAdjustEnabled: false`

## 测试配置

### 基础配置
```yaml
model:
  path: "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf"
  max_context_length: 2048

scheduler:
  max_iterations: 2000
  batch_timeout_ms: 100
  max_batch_size: 64
  loop_interval: 5

dynamic_batch_tuner:
  enabled: true
  min_batch_size: 1
  max_batch_size: 64
  initial_batch_size: 8
  probing_growth_factor: 2.0
  max_probing_attempts: 10
  time_increase_threshold: 0.30
  adjustment_factor: 0.30
  validation_interval: 50
  exploration_interval: 200
  probe_batch_count: 10
  validation_batch_count: 10
  max_consecutive_time_increases: 5
```

### 策略特定配置

#### Static 策略配置
```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "static"
  fixed_batch_size: 8
```

#### Dynamic 策略配置
```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "dynamic"
  autoAdjustEnabled: true
```

#### Hybrid 策略配置
```yaml
dynamic_batch_tuner:
  enabled: true
  strategy: "hybrid"
  autoAdjustEnabled: false
```

## 测试场景

### 场景 1: 低并发测试
- **并发级别**: 8
- **max_tokens**: 50
- **请求数**: 72
- **预期**: Static 可能表现最佳，dynamic/hybrid 探测开销明显

### 场景 2: 中并发测试
- **并发级别**: 16
- **max_tokens**: 50
- **请求数**: 72
- **预期**: Hybrid 可能表现最佳，平衡了探测开销和性能

### 场景 3: 高并发测试
- **并发级别**: 24
- **max_tokens**: 50
- **请求数**: 72
- **预期**: Dynamic 可能表现最佳，充分利用 GPU 资源

### 场景 4: 超高并发测试
- **并发级别**: 32
- **max_tokens**: 50
- **请求数**: 72
- **预期**: Dynamic 可能表现最佳，但需要关注内存使用

### 场景 5: 长文本生成测试
- **并发级别**: 16
- **max_tokens**: 100
- **请求数**: 72
- **预期**: Hybrid 可能表现最佳，避免动态调整带来的延迟

## 测试指标

### 核心指标
1. **Throughput** (tokens/sec) - 吞吐量
2. **TPS** (tokens/sec per request) - 单请求速度
3. **Avg Response Time** (ms) - 平均响应时间
4. **P95 Response Time** (ms) - 95分位响应时间
5. **P99 Response Time** (ms) - 99分位响应时间

### 辅助指标
1. **Success Rate** (%) - 成功率
2. **Memory Usage** (MB) - 内存使用量
3. **GPU Utilization** (%) - GPU 利用率
4. **Batch Size Distribution** - Batch size 分布（dynamic/hybrid）

### 稳定性指标
1. **Throughput Variance** - 吞吐量方差
2. **Response Time Jitter** - 响应时间抖动
3. **Batch Size Changes** - Batch size 变化次数（dynamic）

## 测试步骤

### 步骤 1: 准备环境
```bash
# 1. 停止现有服务
pkill -f cllm_server

# 2. 清理缓存
rm -rf /tmp/cllm_cache/*

# 3. 预热 GPU（可选）
# 运行几个请求预热系统
```

### 步骤 2: Static 策略测试
```bash
# 2.1 修改配置
# 编辑 config/config.yaml，设置 strategy: "static"

# 2.2 编译（如果需要）
cd build && make -j8 && cd ..

# 2.3 启动服务
./build/cllm_server --config config/config.yaml

# 2.4 运行测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/static_benchmark_${concurrency}.json
done

# 2.5 停止服务
pkill -f cllm_server
```

### 步骤 3: Dynamic 策略测试
```bash
# 3.1 修改配置
# 编辑 config/config.yaml，设置 strategy: "dynamic"

# 3.2 启动服务
./build/cllm_server --config config/config.yaml

# 3.3 等待探测完成（约 30-60 秒）
sleep 60

# 3.4 运行测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/dynamic_benchmark_${concurrency}.json
done

# 3.5 停止服务
pkill -f cllm_server
```

### 步骤 4: Hybrid 策略测试
```bash
# 4.1 修改配置
# 编辑 config/config.yaml，设置 strategy: "hybrid"

# 4.2 启动服务
./build/cllm_server --config config/config.yaml

# 4.3 等待探测完成（约 30-60 秒）
sleep 60

# 4.4 运行测试
for concurrency in 8 16 24 32; do
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency $concurrency \
    --max-tokens 50 \
    --output-file /tmp/hybrid_benchmark_${concurrency}.json
done

# 4.5 停止服务
pkill -f cllm_server
```

### 步骤 5: 长文本生成测试
```bash
# 对三种策略分别测试 max_tokens=100
for strategy in static dynamic hybrid; do
  # 修改配置
  # 编辑 config/config.yaml，设置 strategy: $strategy
  
  # 启动服务
  ./build/cllm_server --config config/config.yaml
  
  # 等待探测完成（dynamic/hybrid）
  sleep 60
  
  # 运行测试
  python3 tools/unified_benchmark.py \
    --server-type cllm \
    --test-type api-concurrent \
    --requests 72 \
    --concurrency 16 \
    --max-tokens 100 \
    --output-file /tmp/${strategy}_benchmark_16_100.json
  
  # 停止服务
  pkill -f cllm_server
done
```

## 数据收集

### 测试日志
- 服务启动日志（探测过程）
- 批处理日志（batch size 变化）
- 性能指标日志（throughput, latency）

### 测试结果
- JSON 格式的测试结果文件
- 响应时间分布
- 内存使用情况

## 分析方法

### 1. 性能对比
- 对比三种策略在不同并发级别下的 Throughput
- 对比三种策略在不同并发级别下的 TPS
- 对比三种策略在不同并发级别下的响应时间

### 2. 稳定性分析
- 计算 Throughput 的标准差
- 计算响应时间的 P95/P99
- 分析 Dynamic 策略的 batch size 变化频率

### 3. 资源利用
- 对比内存使用量
- 对比 GPU 利用率（如果可用）
- 分析批处理效率

### 4. 推荐策略
- 根据测试结果，为不同场景推荐最佳策略
- 提供配置建议
- 说明权衡和注意事项

## 预期结果

### 低并发（8）
- Static: 最佳（无探测开销）
- Hybrid: 次优（探测后锁定）
- Dynamic: 较差（探测开销明显）

### 中并发（16）
- Hybrid: 最佳（平衡探测开销和性能）
- Dynamic: 次优（持续调整）
- Static: 较差（未充分利用资源）

### 高并发（24）
- Dynamic: 最佳（充分利用资源）
- Hybrid: 次优（探测后锁定）
- Static: 较差（资源利用不足）

### 超高并发（32）
- Dynamic: 最佳（充分利用资源）
- Hybrid: 次优（探测后锁定）
- Static: 较差（资源利用不足）

### 长文本（max_tokens=100）
- Hybrid: 最佳（稳定性能）
- Static: 次优（固定性能）
- Dynamic: 较差（动态调整带来延迟）

## 风险和注意事项

1. **内存溢出**: Dynamic 策略可能尝试过大的 batch size
2. **探测时间**: Dynamic/Hybrid 需要预热时间
3. **配置复杂度**: Dynamic 需要调整更多参数
4. **稳定性**: Dynamic 可能在某些情况下不稳定

## 时间估算

- Static 测试: ~10 分钟
- Dynamic 测试: ~20 分钟（含预热）
- Hybrid 测试: ~20 分钟（含预热）
- 长文本测试: ~30 分钟
- 数据分析: ~30 分钟
- **总计**: ~2 小时
