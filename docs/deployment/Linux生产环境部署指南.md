# cLLM Linux 生产环境部署指南

本文档详细介绍如何在 Linux 生产环境（Ubuntu/CentOS）+ NVIDIA GPU 上部署 cLLM 服务。

> **生产环境说明**：本指南仅涵盖 **GGUF 模型 + llama.cpp 后端** 的部署，这是当前推荐的生产配置。
> 其他后端（Kylin、LibTorch）为实验性功能，暂不支持生产部署。

## 目录

1. [系统要求](#1-系统要求)
2. [环境准备](#2-环境准备)
3. [依赖安装](#3-依赖安装)
4. [项目编译](#4-项目编译)
5. [模型准备](#5-模型准备)
6. [配置文件](#6-配置文件)
7. [服务部署](#7-服务部署)
8. [监控与日志](#8-监控与日志)
9. [性能调优](#9-性能调优)
10. [故障排查](#10-故障排查)

---

## 1. 系统要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8+ 核 |
| 内存 | 16 GB | 32+ GB |
| GPU | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 3090/4090 (24GB) |
| 磁盘 | 50 GB SSD | 100+ GB NVMe SSD |

### 1.2 软件要求

| 组件 | 版本要求 | 说明 |
|------|----------|------|
| 操作系统 | Ubuntu 20.04/22.04 LTS 或 CentOS 7/8/Stream | |
| NVIDIA 驱动 | >= 525.x | 必需 |
| CUDA | >= 11.8（推荐 12.x） | 必需 |
| cuDNN | >= 8.6 | 可选（llama.cpp 不依赖） |
| GCC | >= 9.0（推荐 11.x） | 必需 |
| CMake | >= 3.18 | 必需 |
| Rust | >= 1.70 | 方案 A 需要（tokenizers-cpp） |

### 1.3 GPU 显存要求（参考）

| 模型大小 | 最低显存 | 推荐显存 |
|----------|----------|----------|
| 0.5B-1B  | 4 GB     | 8 GB     |
| 3B-7B    | 8 GB     | 16 GB    |
| 13B-14B  | 16 GB    | 24 GB    |
| 32B+     | 24 GB    | 48+ GB   |

---

## 2. 环境准备

### 2.1 Ubuntu 22.04

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git wget curl vim htop

# 安装编译依赖
sudo apt install -y cmake pkg-config libssl-dev libcurl4-openssl-dev
```

### 2.2 CentOS 7/8

```bash
# CentOS 7 - 启用 SCL 获取新版 GCC
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++
scl enable devtoolset-11 bash

# CentOS 8/Stream
sudo dnf install -y gcc-toolset-11
scl enable gcc-toolset-11 bash

# 通用依赖
sudo yum install -y git wget curl vim htop cmake3 openssl-devel
```

---

## 3. 依赖安装

### 3.1 NVIDIA 驱动安装

#### Ubuntu

```bash
# 方法 1: 使用 ubuntu-drivers（推荐）
sudo ubuntu-drivers autoinstall

# 方法 2: 手动安装指定版本
sudo apt install -y nvidia-driver-535

# 重启
sudo reboot

# 验证安装
nvidia-smi
```

#### CentOS

```bash
# 禁用 nouveau 驱动
sudo bash -c "echo 'blacklist nouveau' >> /etc/modprobe.d/blacklist.conf"
sudo bash -c "echo 'options nouveau modeset=0' >> /etc/modprobe.d/blacklist.conf"
sudo dracut --force
sudo reboot

# 安装驱动（从 NVIDIA 官网下载 .run 文件）
sudo chmod +x NVIDIA-Linux-x86_64-535.xxx.run
sudo ./NVIDIA-Linux-x86_64-535.xxx.run

# 验证
nvidia-smi
```

### 3.2 CUDA 安装

```bash
# 下载 CUDA 12.x（以 Ubuntu 22.04 为例）
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# 安装（跳过驱动，因为已安装）
sudo sh cuda_12.4.0_550.54.14_linux.run --toolkit --silent

# 配置环境变量
cat >> ~/.bashrc << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
source ~/.bashrc

# 验证
nvcc --version
```

### 3.3 cuDNN 安装（可选）

> **注意**：llama.cpp 后端使用 cuBLAS，**不依赖 cuDNN**。如果只部署 llama.cpp 后端，可以跳过此步骤。

```bash
# 从 NVIDIA 开发者网站下载 cuDNN（需要注册）
# https://developer.nvidia.com/cudnn

# 解压并安装
tar -xvf cudnn-linux-x86_64-8.9.x.xx_cuda12-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.9.x.xx_cuda12-archive/include/* /usr/local/cuda/include/
sudo cp cudnn-linux-x86_64-8.9.x.xx_cuda12-archive/lib/* /usr/local/cuda/lib64/
sudo ldconfig

# 验证
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

### 3.4 系统依赖安装

#### Ubuntu

```bash
sudo apt install -y \
    libyaml-cpp-dev \
    libspdlog-dev \
    nlohmann-json3-dev \
    libomp-dev \
    libopenblas-dev \
    python3-pip
```

#### CentOS

```bash
# EPEL 仓库
sudo yum install -y epel-release

# 依赖包
sudo yum install -y \
    yaml-cpp-devel \
    spdlog-devel \
    openblas-devel \
    python3-pip

# nlohmann-json 需要手动安装
git clone https://github.com/nlohmann/json.git
cd json && mkdir build && cd build
cmake .. && sudo make install
```

### 3.5 vcpkg 安装（推荐的包管理方式）

```bash
# 安装 vcpkg
git clone https://github.com/microsoft/vcpkg.git ~/vcpkg
cd ~/vcpkg && ./bootstrap-vcpkg.sh

# 安装依赖
~/vcpkg/vcpkg install \
    nlohmann-json \
    yaml-cpp \
    spdlog \
    asio

# 配置环境变量
export VCPKG_ROOT=~/vcpkg
export CMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
```

---

## 4. 项目编译

### 4.1 获取源码

```bash
# 克隆项目
git clone https://github.com/xdongp/cLLM.git
cd cLLM

# 初始化子模块
git submodule update --init --recursive
```

### 4.2 编译 llama.cpp（带 CUDA 支持）

```bash
cd third_party/llama.cpp

# 创建构建目录
mkdir -p build && cd build

# 配置（启用 CUDA）
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

cd ../../..
```

**CUDA 架构代号说明**：

`CMAKE_CUDA_ARCHITECTURES` 指定编译器为哪些 GPU 架构生成优化代码。数字代表 NVIDIA GPU 的**计算能力（Compute Capability）**。

| 代号 | 架构名称 | GPU 系列 | 说明 |
|------|----------|----------|------|
| 61 | Pascal | GTX 10xx (1060, 1070, 1080) | 较老，可不支持 |
| 75 | Turing | GTX 16xx, RTX 20xx | 消费级入门 |
| 80 | Ampere | A100, RTX 30xx | 数据中心/消费级 |
| 86 | Ampere | RTX 30xx Ti | 消费级主流 |
| 89 | Ada Lovelace | RTX 40xx (4060-4090) | 最新消费级 |
| 90 | Hopper | H100, H200 | 数据中心 |

**查看你的 GPU 架构代号**：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# 输出示例: 8.6 表示架构代号 86（RTX 3080）
```

**配置建议**：
```bash
# 只编译你实际使用的架构（减少编译时间）
-DCMAKE_CUDA_ARCHITECTURES="86"           # 只有 RTX 3080
-DCMAKE_CUDA_ARCHITECTURES="89"           # 只有 RTX 4090
-DCMAKE_CUDA_ARCHITECTURES="80;90"        # 数据中心 A100 + H100
-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"  # 兼容多种消费级 GPU
```

### 4.3 编译 tokenizers-cpp（二选一）

cLLM 支持两种 Tokenizer 方案，根据需求选择：

| 方案 | 依赖 | Tokenizer 来源 | 推荐场景 |
|------|------|----------------|----------|
| **方案 A** | 需要编译 tokenizers-cpp | `tokenizer.json` 文件 | 推荐，更精确 |
| **方案 B** | 无额外依赖 | GGUF 模型内置 | 简化部署 |

#### 方案 A：使用 tokenizers-cpp（推荐）

```bash
cd third_party/tokenizers-cpp

# 安装 Rust（如果没有）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 编译
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

cd ../../..
```

选择此方案后，需要下载 `tokenizer.json`（见第 5 节）。

#### 方案 B：使用 GGUF 内置 Tokenizer（简化）

跳过 tokenizers-cpp 编译，直接使用 GGUF 模型内置的 tokenizer。

编译 cLLM 时禁用 tokenizers-cpp：
```bash
cmake .. -DUSE_TOKENIZERS_CPP=OFF -DCMAKE_BUILD_TYPE=Release
```

选择此方案后，**不需要下载** `tokenizer.json`，模型目录只需 GGUF 文件。

> **注意**：方案 B 的 tokenizer 精度略低于方案 A，但对于大多数场景足够使用。

### 4.4 编译 cLLM

```bash
mkdir -p build && cd build

# 配置（使用 vcpkg）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

# 或不使用 vcpkg（依赖系统安装）
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"

# 编译
make -j$(nproc)

# 验证
./bin/cllm_server --help
```

---

## 5. 模型准备

### 5.1 下载 GGUF 模型

```bash
# 创建模型目录
sudo mkdir -p /opt/models
sudo chown $USER:$USER /opt/models

# 使用 huggingface-cli 下载
pip3 install huggingface_hub
huggingface-cli download \
    Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir /opt/models/qwen2.5-7b

# 或使用 wget 直接下载
wget -P /opt/models/qwen2.5-7b \
    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
```

### 5.2 下载 Tokenizer 文件（方案 A 需要）

> **重要**：仅当使用 **方案 A（tokenizers-cpp）** 时需要下载。方案 B 跳过此步骤。

```bash
# 下载 tokenizer.json（HuggingFace tokenizers 需要）
huggingface-cli download \
    Qwen/Qwen2.5-7B-Instruct \
    tokenizer.json tokenizer_config.json \
    --local-dir /opt/models/qwen2.5-7b
```

### 5.3 模型文件结构

**方案 A（tokenizers-cpp）**：
```
/opt/models/qwen2.5-7b/
├── qwen2.5-7b-instruct-q4_k_m.gguf    # GGUF 模型文件（必需）
├── tokenizer.json                      # HuggingFace tokenizer（必需）
└── tokenizer_config.json               # tokenizer 配置（可选）
```

**方案 B（GGUF 内置）**：
```
/opt/models/qwen2.5-7b/
└── qwen2.5-7b-instruct-q4_k_m.gguf    # GGUF 模型文件（仅需此文件）
```

---

## 6. 配置文件

### 6.1 生产环境配置示例

> **重要**：生产环境必须使用 `backend.type: "llama_cpp"`，这是唯一支持生产部署的后端。

创建 `/opt/cllm/config/production.yaml`：

```yaml
# cLLM 生产环境配置 - GGUF + llama.cpp 后端 + NVIDIA GPU

# ============================================================
# 服务器配置
# ============================================================
server:
  host: "0.0.0.0"          # 监听所有网卡
  port: 8080               # 服务端口
  num_threads: 16          # HTTP 工作线程数（建议 = CPU 核心数）
  min_threads: 8           # 最小线程数

# ============================================================
# 模型配置
# ============================================================
model:
  path: "/opt/models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf"
  vocab_size: 152064       # Qwen2.5 词表大小
  max_context_length: 32768
  default_max_tokens: 2048

# ============================================================
# 后端配置（关键）
# ============================================================
backend:
  type: "llama_cpp"
  
  llama_cpp:
    n_batch: 2048          # 批处理大小（GPU 可以设置更大）
    n_threads: 8           # CPU 线程数（用于非 GPU 操作）
    n_gpu_layers: 99       # 所有层放 GPU（99 = 全部）
    n_ctx: 32768           # 上下文长度
    n_seq_max: 8           # 最大并发序列数
    use_mmap: true         # 内存映射
    use_mlock: true        # 锁定内存（防止换出）
    flash_attn: true       # Flash Attention（如果支持）

# ============================================================
# Tokenizer 配置
# ============================================================
# 方案 A（tokenizers-cpp）：
tokenizer:
  type: "huggingface"
  path: "/opt/models/qwen2.5-7b/tokenizer.json"
  add_bos_token: false
  add_eos_token: false

# 方案 B（GGUF 内置）：使用 "auto" 或 "gguf"
# tokenizer:
#   type: "auto"  # 自动从 GGUF 模型加载

# ============================================================
# 调度器配置
# ============================================================
scheduler:
  max_batch_size: 8        # 最大批处理大小
  request_timeout: 600.0   # 请求超时（秒）
  default_max_tokens: 2048
  loop_interval: 1         # 调度循环间隔（毫秒）
  
  # 采样参数
  default_temperature: 0.7
  default_top_p: 0.9
  default_top_k: 50

# ============================================================
# 资源配置
# ============================================================
resources:
  max_context_length: 32768
  kv_cache_max_size: 32    # KV cache 最大条目数
  memory_limit_mb: 0       # 0 = 不限制

# ============================================================
# 日志配置
# ============================================================
logging:
  level: "info"            # debug | info | warn | error
  file: "/var/log/cllm/cllm.log"
  max_size_mb: 100
  max_files: 10
```

### 6.2 环境变量配置

创建 `/opt/cllm/env.sh`：

```bash
#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/opt/cllm/lib:$LD_LIBRARY_PATH

# CUDA 设备选择（多卡时指定）
export CUDA_VISIBLE_DEVICES=0

# llama.cpp 日志级别
export LLAMA_LOG_LEVEL=warn

# 线程配置
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

---

## 7. 服务部署

### 7.1 目录结构

```bash
# 创建部署目录
sudo mkdir -p /opt/cllm/{bin,config,lib,logs}
sudo chown -R $USER:$USER /opt/cllm

# 复制文件
cp build/bin/cllm_server /opt/cllm/bin/
cp -r config/* /opt/cllm/config/
cp build/lib/*.so /opt/cllm/lib/ 2>/dev/null || true

# 创建日志目录
sudo mkdir -p /var/log/cllm
sudo chown $USER:$USER /var/log/cllm
```

### 7.2 Systemd 服务配置

创建 `/etc/systemd/system/cllm.service`：

```ini
[Unit]
Description=cLLM Large Language Model Server
After=network.target

[Service]
Type=simple
User=cllm
Group=cllm
WorkingDirectory=/opt/cllm

# 环境变量
Environment="CUDA_HOME=/usr/local/cuda"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/cllm/lib"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OMP_NUM_THREADS=8"

# 启动命令
ExecStart=/opt/cllm/bin/cllm_server --config /opt/cllm/config/production.yaml

# 重启策略
Restart=always
RestartSec=10

# 资源限制
LimitNOFILE=65535
LimitNPROC=65535

# 日志
StandardOutput=append:/var/log/cllm/cllm.log
StandardError=append:/var/log/cllm/cllm.error.log

[Install]
WantedBy=multi-user.target
```

### 7.3 创建服务用户

```bash
# 创建专用用户
sudo useradd -r -s /bin/false -d /opt/cllm cllm

# 设置权限
sudo chown -R cllm:cllm /opt/cllm /var/log/cllm

# 添加用户到 video 组（GPU 访问权限）
sudo usermod -aG video cllm
```

### 7.4 启动服务

```bash
# 重新加载 systemd
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start cllm

# 开机自启
sudo systemctl enable cllm

# 查看状态
sudo systemctl status cllm

# 查看日志
sudo journalctl -u cllm -f
```

### 7.5 健康检查

```bash
# 检查服务状态
curl http://localhost:8080/health

# 查看模型信息
curl http://localhost:8080/model/info

# 测试生成
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```

---

## 8. 监控与日志

### 8.1 GPU 监控

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# GPU 监控脚本
cat > /opt/cllm/monitor_gpu.sh << 'EOF'
#!/bin/bash
while true; do
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
        --format=csv >> /var/log/cllm/gpu_stats.csv
    sleep 60
done
EOF
chmod +x /opt/cllm/monitor_gpu.sh
```

### 8.2 日志轮转

创建 `/etc/logrotate.d/cllm`：

```
/var/log/cllm/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 cllm cllm
    postrotate
        systemctl reload cllm > /dev/null 2>&1 || true
    endscript
}
```

### 8.3 Prometheus 指标（可选）

如果需要集成 Prometheus 监控：

```bash
# 使用 nvidia_gpu_exporter
docker run -d \
  --name nvidia_exporter \
  --gpus all \
  -p 9835:9835 \
  utkuozdemir/nvidia_gpu_exporter:1.2.0
```

---

## 9. 性能调优

### 9.1 GPU 优化

```bash
# 设置 GPU 持久模式（减少启动延迟）
sudo nvidia-smi -pm 1

# 设置 GPU 时钟（可选，提升性能）
sudo nvidia-smi -lgc 1500,1500  # 锁定 GPU 时钟

# 设置 GPU 功耗限制（可选）
sudo nvidia-smi -pl 350  # 设置功耗上限
```

### 9.2 系统优化

```bash
# 增加文件描述符限制
cat >> /etc/security/limits.conf << EOF
cllm soft nofile 65535
cllm hard nofile 65535
cllm soft nproc 65535
cllm hard nproc 65535
EOF

# 优化网络参数
cat >> /etc/sysctl.conf << EOF
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
EOF
sudo sysctl -p
```

### 9.3 配置调优建议

| 参数 | 小模型 (< 3B) | 中等模型 (3-14B) | 大模型 (> 14B) |
|------|---------------|------------------|----------------|
| `n_batch` | 512 | 1024-2048 | 512-1024 |
| `n_ctx` | 8192 | 16384 | 8192-16384 |
| `n_seq_max` | 8-16 | 4-8 | 2-4 |
| `max_batch_size` | 16 | 8 | 4 |

---

## 10. 故障排查

### 10.1 常见问题

#### CUDA 内存不足

```
Error: CUDA out of memory
```

**解决方案**：
1. 减少 `n_ctx` 上下文长度
2. 减少 `n_batch` 批处理大小
3. 使用更小的量化版本（如 Q4_K_M → Q4_K_S）
4. 减少 `n_gpu_layers`（部分层放 CPU）

#### GPU 驱动问题

```
Error: CUDA driver version is insufficient
```

**解决方案**：
```bash
# 更新驱动
sudo apt install nvidia-driver-535
sudo reboot
```

#### 模型加载失败

```
Error: Failed to load model
```

**解决方案**：
1. 检查模型文件路径是否正确
2. 检查模型文件完整性（MD5/SHA256）
3. 检查磁盘空间和内存

### 10.2 诊断命令

```bash
# 检查 GPU 状态
nvidia-smi -q

# 检查 CUDA 版本
nvcc --version
cat /usr/local/cuda/version.txt

# 检查 llama.cpp CUDA 支持
ldd /opt/cllm/bin/cllm_server | grep -i cuda

# 检查服务日志
sudo journalctl -u cllm -n 100 --no-pager

# 检查端口占用
sudo netstat -tlnp | grep 8080

# 检查内存使用
free -h
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"

# 检查 GPU 显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 10.3 性能诊断

```bash
# 使用内置 benchmark
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 100, "iterations": 10}'

# 使用 unified_benchmark.py
python3 tools/unified_benchmark.py \
  --server-type cllm \
  --server-url http://localhost:8080 \
  --requests 50 \
  --concurrency 4 \
  --max-tokens 100
```

---

## 附录

### A. 快速部署脚本

```bash
#!/bin/bash
# deploy.sh - cLLM 快速部署脚本

set -e

INSTALL_DIR="/opt/cllm"
MODEL_DIR="/opt/models"
LOG_DIR="/var/log/cllm"

echo "=== cLLM 部署脚本 ==="

# 检查 root 权限
if [[ $EUID -ne 0 ]]; then
   echo "请使用 root 权限运行此脚本"
   exit 1
fi

# 创建目录
mkdir -p $INSTALL_DIR/{bin,config,lib}
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

# 创建用户
useradd -r -s /bin/false -d $INSTALL_DIR cllm 2>/dev/null || true
usermod -aG video cllm

# 设置权限
chown -R cllm:cllm $INSTALL_DIR $LOG_DIR

echo "=== 部署完成 ==="
echo "请手动完成以下步骤："
echo "1. 复制编译好的 cllm_server 到 $INSTALL_DIR/bin/"
echo "2. 配置 $INSTALL_DIR/config/production.yaml"
echo "3. 下载模型到 $MODEL_DIR/"
echo "4. 启动服务: systemctl start cllm"
```

### B. 参考链接

- [NVIDIA CUDA 安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [cuDNN 安装指南](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)
- [llama.cpp 文档](https://github.com/ggerganov/llama.cpp)
- [HuggingFace 模型下载](https://huggingface.co/models)

---

*文档版本: 1.1*  
*最后更新: 2026-02-03*  
*支持后端: llama.cpp (GGUF)*
