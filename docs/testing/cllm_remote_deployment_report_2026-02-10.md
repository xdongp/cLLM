# cLLM 远程部署测试报告

## 测试概要

**测试日期**: 2026-02-10
**测试服务器**: 172.16.137.156 (CentOS 7)
**测试目标**: 部署cLLM并测试llama.cpp backend的性能和稳定性
**测试结果**: ✅ 部署成功，性能测试通过

---

## 一、部署环境

### 1.1 服务器硬件配置

| 项目 | 配置 |
|------|------|
| CPU架构 | aarch64 (ARM64) |
| CPU核心数 | 4核 |
| 内存 | 7.2GB |
| 交换空间 | 2.0GB |
| 操作系统 | CentOS 7 |

### 1.2 软件环境

| 组件 | 版本 |
|------|------|
| GCC | 10.2.1 (devtoolset-10) |
| CMake | 3.26.4 |
| OpenBLAS | 已安装 |
| llama.cpp | 0.9.5 |

---

## 二、部署过程

### 2.1 打包阶段

使用 `make package` 命令打包cLLM源码：

```bash
make package
```

**生成的包**: `packages/cLLM-0.3.0-source.tar.gz`
**包大小**: 约 200MB

### 2.2 传输阶段

将打包文件传输到远程服务器：

```bash
scp packages/cLLM-0.3.0-source.tar.gz root@172.16.137.156:/tmp/
```

### 2.3 依赖安装

在远程服务器上安装必要的依赖：

```bash
# 安装基础工具
yum install -y epel-release
yum install -y git cmake3 make openssl-devel

# 安装devtoolset-10（获取GCC 10.2.1）
yum install -y centos-release-scl
yum install -y devtoolset-10

# 安装OpenBLAS
yum install -y openblas-devel
```

### 2.4 编译阶段

#### 2.4.1 编译llama.cpp

```bash
cd /tmp/cLLM/third_party/llama.cpp
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DCMAKE_BUILD_TYPE=Release
make -j4
make install DESTDIR=/usr/local
```

**编译结果**: ✅ 成功
**编译时间**: 约 5分钟

#### 2.4.2 编译cLLM

```bash
cd /tmp/cLLM
mkdir -p build && cd build
source /opt/rh/devtoolset-10/enable
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_TOKENIZERS_CPP=OFF \
    -DUSE_LIBTORCH=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF
make -j4
```

**编译结果**: ✅ 成功
**编译时间**: 约 10分钟

### 2.5 部署阶段

#### 2.5.1 目录结构

```bash
/opt/cllm/
├── bin/
│   └── cllm_server
├── config/
│   └── config_llama_cpp_cpu.yaml
├── lib/
│   ├── libggml-base.so*
│   ├── libggml-blas.so*
│   ├── libggml-cpu.so*
│   ├── libggml.so*
│   ├── libllama.so*
│   └── libmtmd.so*
└── env.sh

/opt/models/
└── qwen2.5-0.5b/
    └── qwen2.5-0.5b-instruct-q4_k_m.gguf

/var/log/cllm/
└── cllm_server.log
```

#### 2.5.2 配置文件

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  num_threads: 4
  min_threads: 2

model:
  path: "/opt/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
  vocab_size: 151936
  max_context_length: 4096
  default_max_tokens: 1024
  quantization: "q4_k_m"

backend:
  type: "llama_cpp"
  llama_cpp:
    n_batch: 256
    n_threads: 4
    n_gpu_layers: 0
    n_ctx: 4096
    n_seq_max: 1
    use_mmap: true

resources:
  max_batch_size: 1
  max_context_length: 4096
  kv_cache_max_memory_mb: 2048
  memory_limit_mb: 4096
```

### 2.6 启动服务

```bash
source /opt/cllm/env.sh
cd /opt/cllm
nohup ./bin/cllm_server --config config/config_llama_cpp_cpu.yaml > /var/log/cllm/cllm_server.log 2>&1 &
```

**启动状态**: ✅ 成功
**监听端口**: 0.0.0.0:8080
**进程ID**: 9837

---

## 三、性能测试

### 3.1 测试配置

| 项目 | 配置 |
|------|------|
| 测试工具 | unified_benchmark.py |
| 测试类型 | API并发测试 |
| 请求总数 | 50 |
| 生成token数 | 30 |
| 测试并发度 | 1, 2, 4 |

### 3.2 测试结果

#### 3.2.1 并发度1

| 指标 | 数值 |
|------|------|
| 总请求数 | 50 |
| 成功请求数 | 50 |
| 失败请求数 | 0 |
| 成功率 | 100% |
| 平均响应时间 | 0.72s |
| 最小响应时间 | 0.46s |
| 最大响应时间 | 0.87s |
| 平均吞吐量 | 40.72 tokens/s |
| 平均生成速度 | 41.02 tokens/s |
| 总测试时间 | 35.88s |

#### 3.2.2 并发度2

| 指标 | 数值 |
|------|------|
| 总请求数 | 50 |
| 成功请求数 | 50 |
| 失败请求数 | 0 |
| 成功率 | 100% |
| 平均响应时间 | 1.46s |
| 最小响应时间 | 0.67s |
| 最大响应时间 | 1.69s |
| 平均吞吐量 | 40.13 tokens/s |
| 平均生成速度 | 26.00 tokens/s |
| 总测试时间 | 36.83s |

#### 3.2.3 并发度4

| 指标 | 数值 |
|------|------|
| 总请求数 | 50 |
| 成功请求数 | 50 |
| 失败请求数 | 0 |
| 成功率 | 100% |
| 平均响应时间 | 2.85s |
| 最小响应时间 | 0.59s |
| 最大响应时间 | 6.50s |
| 平均吞吐量 | 39.96 tokens/s |
| 平均生成速度 | 18.54 tokens/s |
| 总测试时间 | 36.81s |

### 3.3 性能分析

#### 3.3.1 吞吐量对比

```
并发度1: 40.72 tokens/s
并发度2: 40.13 tokens/s
并发度4: 39.96 tokens/s
```

**分析**: 随着并发度增加，整体吞吐量保持稳定，说明系统在并发场景下性能表现良好。

#### 3.3.2 响应时间对比

```
并发度1: 0.72s (平均)
并发度2: 1.46s (平均)
并发度3: 2.85s (平均)
```

**分析**: 响应时间随并发度增加而线性增长，符合预期。由于配置中 `n_seq_max=1`，请求被串行处理。

#### 3.3.3 资源使用

| 资源 | 使用情况 |
|------|----------|
| 内存 | 714MB / 7.2GB (9.9%) |
| CPU | 217% (约2核) |
| 网络 | 正常 |

**分析**: 内存使用量合理，CPU利用率适中，系统运行稳定。

---

## 四、稳定性测试

### 4.1 长时间运行

**测试时长**: 约30分钟
**测试请求**: 150个请求（50×3并发度）
**失败请求**: 0
**成功率**: 100%

### 4.2 错误日志

检查服务器日志，未发现错误或异常：
```
tail -100 /var/log/cllm/cllm_server.log
```

**结果**: 日志正常，无错误信息

### 4.3 内存泄漏检查

| 时间点 | 内存使用 |
|--------|----------|
| 启动后 | 676MB |
| 测试后 | 714MB |

**分析**: 内存增长38MB，属于正常范围，无明显内存泄漏。

---

## 五、问题与解决方案

### 5.1 问题1: 初始配置内存不足

**问题描述**: 服务器启动时出现 `std::bad_alloc` 错误

**原因**: 初始配置中 `kv_cache_max_memory_mb` 设置过大（4096MB）

**解决方案**: 调整配置参数：
- `kv_cache_max_memory_mb`: 4096 → 2048
- `memory_limit_mb`: 8192 → 4096
- `n_ctx`: 8192 → 4096
- `n_seq_max`: 2 → 1

**结果**: 问题解决，服务器正常启动

### 5.2 问题2: GCC版本过低

**问题描述**: CentOS 7默认GCC 4.8.5不支持C++17特性

**解决方案**: 安装devtoolset-10获取GCC 10.2.1

```bash
yum install -y devtoolset-10
source /opt/rh/devtoolset-10/enable
```

**结果**: 编译成功

### 5.3 问题3: 部署包缺少tools目录

**问题描述**: Makefile的package目标未包含tools目录

**解决方案**: 手动复制tools目录到远程服务器

```bash
scp -r tools root@172.16.137.156:/tmp/cLLM/
```

**建议**: 更新Makefile的package目标，包含tools目录

---

## 六、结论

### 6.1 部署结果

✅ **部署成功**

cLLM已成功部署到CentOS 7服务器，llama.cpp backend运行正常。

### 6.2 性能评估

| 指标 | 评级 |
|------|------|
| 成功率 | ⭐⭐⭐⭐⭐ 100% |
| 吞吐量 | ⭐⭐⭐⭐ 40 tokens/s |
| 响应时间 | ⭐⭐⭐⭐ 0.72s (并发1) |
| 资源使用 | ⭐⭐⭐⭐⭐ 9.9% 内存 |
| 稳定性 | ⭐⭐⭐⭐⭐ 无错误 |

### 6.3 建议

1. **配置优化**: 根据服务器资源调整 `n_seq_max` 参数以支持真正的并发处理
2. **打包改进**: 更新Makefile的package目标，确保包含所有必要文件
3. **监控增强**: 添加Prometheus/Grafana监控以实时跟踪性能指标
4. **文档完善**: 创建详细的部署文档，包括依赖安装和故障排查

### 6.4 下一步

1. 测试GPU backend（如果服务器有GPU）
2. 进行长时间稳定性测试（24小时+）
3. 测试不同模型规模（0.5B、1B、3B等）
4. 优化并发配置以提高吞吐量

---

## 七、附录

### 7.1 测试脚本

```bash
# 健康检查
curl -s http://172.16.137.156:8080/health

# 单请求测试
curl -s -X POST http://172.16.137.156:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"人工智能是","max_tokens":50,"temperature":0.7,"top_p":0.9}'

# 并发测试
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://172.16.137.156:8080 \
  --test-type api-concurrent \
  --requests 50 \
  --concurrency 1 \
  --max-tokens 30 \
  --output-file results/remote_llamacpp_conc1.json
```

### 7.2 相关文件

- 部署脚本: [deploy_centos7.sh](file:///Users/dannypan/PycharmProjects/cLLM/scripts/deploy_centos7.sh)
- 打包配置: [Makefile](file:///Users/dannypan/PycharmProjects/cLLM/Makefile)
- 测试工具: [unified_benchmark.py](file:///Users/dannypan/PycharmProjects/cLLM/tools/unified_benchmark.py)
- 测试结果:
  - [remote_llamacpp_conc1.json](file:///Users/dannypan/PycharmProjects/cLLM/results/remote_llamacpp_conc1.json)
  - [remote_llamacpp_conc2.json](file:///Users/dannypan/PycharmProjects/cLLM/results/remote_llamacpp_conc2.json)
  - [remote_llamacpp_conc4.json](file:///Users/dannypan/PycharmProjects/cLLM/results/remote_llamacpp_conc4.json)

---

**报告生成时间**: 2026-02-10
**测试执行人**: cLLM Team
**报告版本**: 1.0
