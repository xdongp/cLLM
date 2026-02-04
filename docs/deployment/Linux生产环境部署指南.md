# cLLM Linux 生产环境部署指南

本文档详细介绍如何在 Linux 生产环境（Ubuntu/CentOS）上部署 cLLM 服务，支持 **GPU 模式** 和 **纯 CPU 模式**。

> **生产环境说明**：
> - 本指南仅涵盖 **GGUF 模型 + llama.cpp 后端** 的部署，这是唯一推荐的生产配置
> - 使用 **GGUF 内置 Tokenizer**，无需额外的 tokenizer 文件
> - 禁用 tokenizers-cpp 和 LibTorch，简化依赖和部署流程
> - 其他后端（Kylin、LibTorch）为实验性功能，仅供开发测试

## 部署模式选择

| 模式 | 适用场景 | 性能 | 依赖 |
|------|----------|------|------|
| **GPU 模式** | 有 NVIDIA 显卡 | 高吞吐、低延迟 | CUDA + 驱动 |
| **CPU 模式** | 无 GPU 或云服务器 | 中等，适合小模型 | 无特殊依赖 |

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

**GPU 模式**：
| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4 核 | 8+ 核 |
| 内存 | 16 GB | 32+ GB |
| GPU | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 3090/4090 (24GB) |
| 磁盘 | 50 GB SSD | 100+ GB NVMe SSD |

**CPU 模式**：
| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 8 核 | 16+ 核（支持 AVX2） |
| 内存 | 16 GB | 32+ GB |
| GPU | 不需要 | - |
| 磁盘 | 50 GB SSD | 100+ GB NVMe SSD |

> **CPU 模式说明**：推荐使用支持 AVX2/AVX-512 指令集的现代 CPU（Intel Haswell+/AMD Zen+）以获得最佳性能。

### 1.2 软件要求

**GPU 模式**：
| 组件 | 版本要求 | 说明 |
|------|----------|------|
| 操作系统 | Ubuntu 20.04/22.04 LTS 或 CentOS 7/8/Stream | |
| NVIDIA 驱动 | >= 525.x | 必需 |
| CUDA | >= 11.8（推荐 12.x） | 必需 |
| GCC | >= 9.0（推荐 11.x） | 必需 |
| CMake | >= 3.18 | 必需 |

**CPU 模式**：
| 组件 | 版本要求 | 说明 |
|------|----------|------|
| 操作系统 | Ubuntu 20.04/22.04 LTS 或 CentOS 7/8/Stream | |
| GCC | >= 9.0（推荐 11.x） | 必需 |
| CMake | >= 3.18 | 必需 |
| OpenBLAS | 最新版 | 推荐，加速矩阵运算 |

### 1.3 GPU 显存要求（GPU 模式）

| 模型大小 | 最低显存 | 推荐显存 |
|----------|----------|----------|
| 0.5B-1B  | 4 GB     | 8 GB     |
| 3B-7B    | 8 GB     | 16 GB    |
| 13B-14B  | 16 GB    | 24 GB    |
| 32B+     | 24 GB    | 48+ GB   |

### 1.4 内存要求（CPU 模式）

| 模型大小 | 量化格式 | 最低内存 | 推荐内存 |
|----------|----------|----------|----------|
| 0.5B-1B  | Q4_K_M   | 4 GB     | 8 GB     |
| 3B-7B    | Q4_K_M   | 8 GB     | 16 GB    |
| 7B-14B   | Q4_K_M   | 16 GB    | 32 GB    |
| 32B+     | Q4_K_M   | 32 GB    | 64+ GB   |

> **提示**：CPU 模式下推荐使用 Q4_K_M 或 Q4_K_S 量化模型以减少内存占用。

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

> **CPU 模式**：可跳过 3.1-3.3 节（NVIDIA 驱动、CUDA、cuDNN），直接到 3.4 节。

### 3.1 NVIDIA 驱动安装（GPU 模式）

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

### 3.2 CUDA 安装（GPU 模式）

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

### 4.2 编译 llama.cpp

根据部署模式选择编译选项：

#### GPU 模式（CUDA）

```bash
cd third_party/llama.cpp
mkdir -p build && cd build

# 配置（启用 CUDA）
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
cd ../../..
```

#### CPU 模式（纯 CPU）

```bash
cd third_party/llama.cpp
mkdir -p build && cd build

# 配置（纯 CPU，启用优化）
cmake .. \
    -DGGML_CUDA=OFF \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
cd ../../..
```

> **CPU 优化提示**：
> - `GGML_BLAS=ON` 启用 BLAS 加速矩阵运算
> - 确保已安装 OpenBLAS：`sudo apt install libopenblas-dev`（Ubuntu）
> - llama.cpp 会自动检测并使用 AVX2/AVX-512 指令集

**CUDA 架构代号说明**（GPU 模式）：

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

### 4.3 编译 cLLM

#### GPU 模式

```bash
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_TOKENIZERS_CPP=OFF \
    -DUSE_LIBTORCH=OFF

make -j$(nproc)
./bin/cllm_server --help
```

#### CPU 模式

```bash
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_TOKENIZERS_CPP=OFF \
    -DUSE_LIBTORCH=OFF

make -j$(nproc)
./bin/cllm_server --help
```

**编译选项说明**：

| 选项 | 值 | 说明 |
|------|-----|------|
| `USE_TOKENIZERS_CPP` | OFF | 禁用 tokenizers-cpp，使用 GGUF 内置 tokenizer |
| `USE_LIBTORCH` | OFF | 禁用 LibTorch 后端（实验性功能） |

> **注意**：
> - cLLM 本身不需要 CUDA，CUDA 依赖在 llama.cpp 编译时处理
> - 生产环境只使用 llama.cpp 后端，Kylin 和 LibTorch 后端仅供开发测试

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

### 5.2 模型文件结构

生产环境使用 GGUF 内置 Tokenizer，模型目录只需 GGUF 文件：

```
/opt/models/qwen2.5-7b/
└── qwen2.5-7b-instruct-q4_k_m.gguf    # GGUF 模型文件（仅需此文件）
```

> **说明**：GGUF 模型已内置完整的 tokenizer 信息，无需额外下载 `tokenizer.json`。

---

## 6. 配置文件

> **重要**：生产环境必须使用 `backend.type: "llama_cpp"`，这是唯一支持生产部署的后端。

### 6.1 GPU 模式配置

创建 `/opt/cllm/config/production_gpu.yaml`：

```yaml
# cLLM 生产环境配置 - GPU 模式
# 后端: llama.cpp (GGUF)
# Tokenizer: GGUF 内置

server:
  host: "0.0.0.0"
  port: 8080
  num_threads: 16

model:
  path: "/opt/models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf"
  vocab_size: 152064
  max_context_length: 32768
  default_max_tokens: 2048

backend:
  type: "llama_cpp"         # 🔥 生产环境唯一推荐后端
  
  llama_cpp:
    n_batch: 2048          # GPU 可以设置更大
    n_threads: 8           # CPU 线程（用于非 GPU 操作）
    n_gpu_layers: 99       # 🔥 关键：99 = 所有层放 GPU
    n_ctx: 32768
    n_seq_max: 8
    use_mmap: true
    use_mlock: true
    flash_attn: true       # Flash Attention（GPU）

# Tokenizer 使用 GGUF 内置，无需额外配置
# tokenizer:
#   type: "gguf"           # 自动从 GGUF 模型读取

scheduler:
  max_batch_size: 8
  request_timeout: 600.0
  default_max_tokens: 2048

resources:
  max_context_length: 32768
  kv_cache_max_size: 32

logging:
  level: "info"
  file: "/var/log/cllm/cllm.log"
```

### 6.2 CPU 模式配置

创建 `/opt/cllm/config/production_cpu.yaml`：

```yaml
# cLLM 生产环境配置 - CPU 模式
# 后端: llama.cpp (GGUF)
# Tokenizer: GGUF 内置

server:
  host: "0.0.0.0"
  port: 8080
  num_threads: 16          # 建议 = CPU 核心数

model:
  path: "/opt/models/qwen2.5-3b/qwen2.5-3b-instruct-q4_k_m.gguf"  # 推荐小模型
  vocab_size: 152064
  max_context_length: 8192   # CPU 模式建议减小
  default_max_tokens: 1024

backend:
  type: "llama_cpp"         # 🔥 生产环境唯一推荐后端
  
  llama_cpp:
    n_batch: 512           # CPU 模式建议较小值
    n_threads: 16          # 🔥 关键：设置为 CPU 核心数
    n_gpu_layers: 0        # 🔥 关键：0 = 纯 CPU 模式
    n_ctx: 8192            # CPU 模式建议减小
    n_seq_max: 2           # CPU 并发能力有限
    use_mmap: true
    use_mlock: false       # CPU 模式可关闭

# Tokenizer 使用 GGUF 内置，无需额外配置
# tokenizer:
#   type: "gguf"           # 自动从 GGUF 模型读取

scheduler:
  max_batch_size: 2        # CPU 模式建议减小
  request_timeout: 600.0
  default_max_tokens: 1024

resources:
  max_context_length: 8192
  kv_cache_max_size: 8     # CPU 模式减小

logging:
  level: "info"
  file: "/var/log/cllm/cllm.log"
```

**CPU 模式配置要点**：
| 参数 | GPU 模式 | CPU 模式 | 说明 |
|------|----------|----------|------|
| `n_gpu_layers` | 99 | **0** | CPU 模式必须为 0 |
| `n_threads` | 8 | **CPU 核心数** | 影响推理速度 |
| `n_batch` | 2048 | 512 | CPU 处理能力有限 |
| `n_ctx` | 32768 | 8192 | 减少内存占用 |
| `n_seq_max` | 8 | 2 | 减少并发压力 |
| `max_batch_size` | 8 | 2 | 减少调度压力 |

### 6.3 环境变量配置

**GPU 模式** (`/opt/cllm/env_gpu.sh`)：
```bash
#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/opt/cllm/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export LLAMA_LOG_LEVEL=warn
export OMP_NUM_THREADS=8
```

**CPU 模式** (`/opt/cllm/env_cpu.sh`)：
```bash
#!/bin/bash
export LD_LIBRARY_PATH=/opt/cllm/lib:$LD_LIBRARY_PATH
export LLAMA_LOG_LEVEL=warn

# 🔥 CPU 线程配置（设置为 CPU 核心数）
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# 禁用 NUMA 交错（单 NUMA 节点优化）
export GOMP_CPU_AFFINITY="0-15"
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

**GPU 模式** - 创建 `/etc/systemd/system/cllm.service`：

```ini
[Unit]
Description=cLLM Large Language Model Server (GPU)
After=network.target

[Service]
Type=simple
User=cllm
Group=cllm
WorkingDirectory=/opt/cllm

# GPU 模式环境变量
Environment="CUDA_HOME=/usr/local/cuda"
Environment="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/cllm/lib"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="OMP_NUM_THREADS=8"

ExecStart=/opt/cllm/bin/cllm_server --config /opt/cllm/config/production_gpu.yaml

Restart=always
RestartSec=10
LimitNOFILE=65535
LimitNPROC=65535

StandardOutput=append:/var/log/cllm/cllm.log
StandardError=append:/var/log/cllm/cllm.error.log

[Install]
WantedBy=multi-user.target
```

**CPU 模式** - 创建 `/etc/systemd/system/cllm.service`：

```ini
[Unit]
Description=cLLM Large Language Model Server (CPU)
After=network.target

[Service]
Type=simple
User=cllm
Group=cllm
WorkingDirectory=/opt/cllm

# CPU 模式环境变量
Environment="LD_LIBRARY_PATH=/opt/cllm/lib"
Environment="OMP_NUM_THREADS=16"
Environment="OPENBLAS_NUM_THREADS=16"

ExecStart=/opt/cllm/bin/cllm_server --config /opt/cllm/config/production_cpu.yaml

Restart=always
RestartSec=10
LimitNOFILE=65535
LimitNPROC=65535

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

### 9.2 CPU 优化

```bash
# 检查 CPU 支持的指令集
cat /proc/cpuinfo | grep -E "avx|avx2|avx512" | head -1

# 设置 CPU 性能模式（关闭节能）
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 禁用透明大页（可能影响延迟）
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# NUMA 优化（多 CPU 服务器）
# 绑定到单个 NUMA 节点
numactl --cpunodebind=0 --membind=0 /opt/cllm/bin/cllm_server --config ...
```

**CPU 线程配置建议**：
| CPU 核心数 | `n_threads` | `OMP_NUM_THREADS` | 说明 |
|------------|-------------|-------------------|------|
| 4 核 | 4 | 4 | 小型服务器 |
| 8 核 | 8 | 8 | 标准配置 |
| 16 核 | 16 | 16 | 推荐配置 |
| 32+ 核 | 16-24 | 16-24 | 过多线程可能降低效率 |

### 9.3 系统优化

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

### 9.4 配置调优建议

**GPU 模式**：
| 参数 | 小模型 (< 3B) | 中等模型 (3-14B) | 大模型 (> 14B) |
|------|---------------|------------------|----------------|
| `n_batch` | 512 | 1024-2048 | 512-1024 |
| `n_ctx` | 8192 | 16384-32768 | 8192-16384 |
| `n_seq_max` | 8-16 | 4-8 | 2-4 |
| `max_batch_size` | 16 | 8 | 4 |

**CPU 模式**：
| 参数 | 小模型 (< 3B) | 中等模型 (3-7B) | 说明 |
|------|---------------|-----------------|------|
| `n_batch` | 256-512 | 128-256 | CPU 处理能力有限 |
| `n_ctx` | 4096-8192 | 2048-4096 | 减少内存占用 |
| `n_seq_max` | 2-4 | 1-2 | 减少并发 |
| `max_batch_size` | 4 | 2 | 减少调度压力 |

> **CPU 模式建议**：推荐使用 3B 以下的小模型（如 Qwen2.5-3B），配合 Q4_K_M 量化获得最佳性价比。

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

## 附录 C: 生产环境架构说明

### 推荐配置

| 组件 | 生产环境配置 | 说明 |
|------|-------------|------|
| **Backend** | llama.cpp | 唯一推荐的生产后端 |
| **Tokenizer** | GGUF 内置 | 无需额外依赖 |
| **模型格式** | GGUF (量化) | 推荐 Q4_K_M 或 Q4_K_S |

### 禁用的组件

| 组件 | CMake 选项 | 说明 |
|------|-----------|------|
| tokenizers-cpp | `-DUSE_TOKENIZERS_CPP=OFF` | 使用 GGUF 内置 tokenizer |
| LibTorch | `-DUSE_LIBTORCH=OFF` | 实验性功能，生产不使用 |
| Kylin Backend | 默认不启用 | 实验性自研后端 |

### 最小依赖清单

```bash
# 必需依赖
- GCC >= 9.0
- CMake >= 3.18
- nlohmann-json
- yaml-cpp
- spdlog
- SentencePiece (llama.cpp 依赖)

# GPU 模式额外依赖
- NVIDIA 驱动 >= 525.x
- CUDA >= 11.8

# 可选依赖（加速）
- OpenBLAS (CPU BLAS 加速)
- OpenMP (并行计算)
```

---

*文档版本: 2.0*  
*最后更新: 2026-02-04*  
*支持后端: llama.cpp (GGUF)*  
*Tokenizer: GGUF 内置*
