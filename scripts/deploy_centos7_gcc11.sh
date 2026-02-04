#!/bin/bash
# ==============================================================================
# cLLM CentOS 7.9 部署脚本 - 使用 Conda GCC 11
# ==============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# ==============================================================================
# 配置
# ==============================================================================
ARCH=$(uname -m)
INSTALL_DIR="/opt/cllm"
MODEL_DIR="/opt/models"
LOG_DIR="/var/log/cllm"
BUILD_DIR="/tmp/cllm_build"
NPROC=$(nproc)
CONDA_DIR="/opt/miniforge"

log_info "系统架构: ${ARCH}"
log_info "CPU 核心数: ${NPROC}"

# 激活 conda 环境
source ${CONDA_DIR}/bin/activate

# 设置 GCC 11 环境变量
export CC=${CONDA_DIR}/bin/aarch64-conda-linux-gnu-gcc
export CXX=${CONDA_DIR}/bin/aarch64-conda-linux-gnu-g++
export PATH=${CONDA_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_DIR}/lib:$LD_LIBRARY_PATH

log_info "GCC 版本: $($CC --version | head -1)"
log_info "G++ 版本: $($CXX --version | head -1)"

# 配置库路径
echo "/usr/local/lib64" > /etc/ld.so.conf.d/local.conf
echo "/usr/local/lib" >> /etc/ld.so.conf.d/local.conf
echo "${CONDA_DIR}/lib" >> /etc/ld.so.conf.d/conda.conf
ldconfig

# ==============================================================================
# 第1步: 安装 nlohmann-json
# ==============================================================================
log_step "1/7 - 安装 nlohmann-json..."
if [ ! -f /usr/local/include/nlohmann/json.hpp ]; then
    cd /tmp
    rm -rf json
    git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
    cd json && mkdir -p build && cd build
    cmake .. -DJSON_BuildTests=OFF
    make -j${NPROC} && make install
    cd /tmp && rm -rf json
fi
log_info "nlohmann-json 安装完成"

# ==============================================================================
# 第2步: 安装 yaml-cpp
# ==============================================================================
log_step "2/7 - 安装 yaml-cpp..."
if [ ! -f /usr/local/include/yaml-cpp/yaml.h ]; then
    cd /tmp
    rm -rf yaml-cpp
    git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git
    cd yaml-cpp && mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DYAML_BUILD_SHARED_LIBS=ON \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX
    make -j${NPROC} && make install
    cd /tmp && rm -rf yaml-cpp
fi
ldconfig
log_info "yaml-cpp 安装完成"

# ==============================================================================
# 第3步: 安装 spdlog
# ==============================================================================
log_step "3/7 - 安装 spdlog..."
if [ ! -f /usr/local/include/spdlog/spdlog.h ]; then
    cd /tmp
    rm -rf spdlog
    git clone --depth 1 --branch v1.12.0 https://github.com/gabime/spdlog.git
    cd spdlog && mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DSPDLOG_BUILD_SHARED=ON \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX
    make -j${NPROC} && make install
    cd /tmp && rm -rf spdlog
fi
ldconfig
log_info "spdlog 安装完成"

# ==============================================================================
# 第4步: 安装 SentencePiece
# ==============================================================================
log_step "4/7 - 安装 SentencePiece..."
if [ ! -f /usr/local/include/sentencepiece_processor.h ]; then
    cd /tmp
    rm -rf sentencepiece
    git clone --depth 1 https://github.com/google/sentencepiece.git
    cd sentencepiece && mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_CXX_COMPILER=$CXX
    make -j${NPROC} && make install
    cd /tmp && rm -rf sentencepiece
fi
ldconfig
log_info "SentencePiece 安装完成"

# ==============================================================================
# 第5步: 安装 LibTorch
# ==============================================================================
log_step "5/7 - 安装 LibTorch..."

LIBTORCH_DIR="/opt/libtorch"
if [ ! -d "${LIBTORCH_DIR}" ] || [ ! -f "${LIBTORCH_DIR}/lib/libtorch.so" ]; then
    log_info "通过 pip 安装 PyTorch..."
    pip3 install --upgrade pip
    pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    
    TORCH_PATH=$(python3 -c "import torch; print(torch.__path__[0])")
    log_info "PyTorch 路径: ${TORCH_PATH}"
    
    mkdir -p ${LIBTORCH_DIR}
    cp -r ${TORCH_PATH}/lib ${LIBTORCH_DIR}/
    cp -r ${TORCH_PATH}/include ${LIBTORCH_DIR}/
    cp -r ${TORCH_PATH}/share ${LIBTORCH_DIR}/ 2>/dev/null || true
    
    mkdir -p ${LIBTORCH_DIR}/share/cmake/Torch
    cat > ${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake << 'EOFCMAKE'
set(TORCH_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/../../../include")
set(TORCH_LIBRARIES "${CMAKE_CURRENT_LIST_DIR}/../../../lib/libtorch.so")
include_directories(${TORCH_INCLUDE_DIRS})
EOFCMAKE
    
    echo "${LIBTORCH_DIR}/lib" > /etc/ld.so.conf.d/libtorch.conf
fi
ldconfig
log_info "LibTorch 安装完成"

# ==============================================================================
# 第6步: 克隆和编译项目
# ==============================================================================
log_step "6/7 - 克隆和编译 cLLM..."

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

git clone https://github.com/xdongp/cLLM.git
cd cLLM
git submodule update --init --recursive

# 编译 llama.cpp
log_info "编译 llama.cpp..."
cd third_party/llama.cpp
mkdir -p build && cd build
cmake .. \
    -DGGML_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
make -j${NPROC}
cd ../../..

# 编译 tokenizers-cpp
log_info "编译 tokenizers-cpp..."
source "$HOME/.cargo/env" 2>/dev/null || true
cd third_party/tokenizers-cpp
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX
make -j${NPROC} && make install
cd ../../..
ldconfig

# 编译 cLLM
log_info "编译 cLLM..."
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/opt/libtorch/lib:${CONDA_DIR}/lib:$LD_LIBRARY_PATH

mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/opt/libtorch;/usr/local" \
    -DUSE_TOKENIZERS_CPP=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX

make -j${NPROC}

log_info "cLLM 编译完成!"

# ==============================================================================
# 第7步: 部署
# ==============================================================================
log_step "7/7 - 部署 cLLM..."

mkdir -p ${INSTALL_DIR}/{bin,config,lib}
mkdir -p ${MODEL_DIR}
mkdir -p ${LOG_DIR}

# 复制文件
cp ${BUILD_DIR}/cLLM/build/bin/cllm_server ${INSTALL_DIR}/bin/
find ${BUILD_DIR}/cLLM/build -name "*.so*" -exec cp {} ${INSTALL_DIR}/lib/ \; 2>/dev/null || true
find ${BUILD_DIR}/cLLM/third_party/llama.cpp/build -name "*.so*" -exec cp {} ${INSTALL_DIR}/lib/ \; 2>/dev/null || true

# 复制 conda 运行时库
cp ${CONDA_DIR}/lib/libstdc++.so* ${INSTALL_DIR}/lib/ 2>/dev/null || true
cp ${CONDA_DIR}/lib/libgcc_s.so* ${INSTALL_DIR}/lib/ 2>/dev/null || true

# 创建配置文件
cat > ${INSTALL_DIR}/config/production_cpu.yaml << 'EOF'
# cLLM 生产环境配置 - CPU 模式

server:
  host: "0.0.0.0"
  port: 8080
  num_threads: 4

model:
  path: "/opt/models/qwen2.5-1.5b/qwen2.5-1.5b-instruct-q4_k_m.gguf"
  vocab_size: 152064
  max_context_length: 4096
  default_max_tokens: 512

backend:
  type: "llama_cpp"
  
  llama_cpp:
    n_batch: 256
    n_threads: 4
    n_gpu_layers: 0
    n_ctx: 4096
    n_seq_max: 2
    use_mmap: true
    use_mlock: false

tokenizer:
  type: "huggingface"
  path: "/opt/models/qwen2.5-1.5b/tokenizer.json"

scheduler:
  max_batch_size: 2
  request_timeout: 600.0
  default_max_tokens: 512

resources:
  max_context_length: 4096
  kv_cache_max_size: 4

logging:
  level: "info"
  file: "/var/log/cllm/cllm.log"
EOF

# 创建服务用户
useradd -r -s /bin/false -d ${INSTALL_DIR} cllm 2>/dev/null || true
chown -R cllm:cllm ${INSTALL_DIR} ${LOG_DIR}
chmod +x ${INSTALL_DIR}/bin/cllm_server

# 创建 systemd 服务
cat > /etc/systemd/system/cllm.service << EOF
[Unit]
Description=cLLM Large Language Model Server (CPU)
After=network.target

[Service]
Type=simple
User=cllm
Group=cllm
WorkingDirectory=${INSTALL_DIR}
Environment="LD_LIBRARY_PATH=${INSTALL_DIR}/lib:/opt/libtorch/lib:/usr/local/lib:/usr/local/lib64:${CONDA_DIR}/lib"
Environment="OMP_NUM_THREADS=${NPROC}"
ExecStart=${INSTALL_DIR}/bin/cllm_server --config ${INSTALL_DIR}/config/production_cpu.yaml
Restart=always
RestartSec=10
LimitNOFILE=65535

StandardOutput=append:${LOG_DIR}/cllm.log
StandardError=append:${LOG_DIR}/cllm.error.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "======================================"
echo -e "${GREEN}  cLLM 部署完成!${NC}"
echo "======================================"
echo ""
echo "安装目录: ${INSTALL_DIR}"
echo "模型目录: ${MODEL_DIR}"
echo ""
echo "下一步操作:"
echo ""
echo "1. 下载模型:"
echo "   pip3 install huggingface_hub"
echo "   huggingface-cli download \\"
echo "       Qwen/Qwen2.5-1.5B-Instruct-GGUF \\"
echo "       qwen2.5-1.5b-instruct-q4_k_m.gguf \\"
echo "       --local-dir ${MODEL_DIR}/qwen2.5-1.5b"
echo ""
echo "2. 下载 tokenizer.json:"
echo "   huggingface-cli download \\"
echo "       Qwen/Qwen2.5-1.5B-Instruct \\"
echo "       tokenizer.json tokenizer_config.json \\"
echo "       --local-dir ${MODEL_DIR}/qwen2.5-1.5b"
echo ""
echo "3. 启动服务:"
echo "   systemctl start cllm"
echo "   systemctl enable cllm"
echo ""
echo "4. 检查状态:"
echo "   curl http://localhost:8080/health"
echo ""
