#!/bin/bash
# ==============================================================================
# cLLM CentOS 7 生产环境部署脚本
# ==============================================================================
#
# 后端: llama.cpp (唯一支持的生产后端)
# Tokenizer: GGUF 内置 (无需 tokenizers-cpp)
# 模式: CPU / GPU
#
# 支持架构: x86_64 / aarch64 (ARM64)
#
# 用法:
#   全新部署:     ./deploy_centos7.sh [--gpu]
#   本地源码部署: ./deploy_centos7.sh --local [--gpu]
#   跳过依赖安装: ./deploy_centos7.sh --skip-deps [--gpu]
#
# 选项:
#   --gpu        启用 GPU 模式 (需要 CUDA)
#   --local      使用当前目录的源码 (适用于 make package 打包的源码)
#   --skip-deps  跳过依赖安装 (假设依赖已安装)
#   --help       显示帮助信息
#
# ==============================================================================

set -e

# ==============================================================================
# 颜色输出
# ==============================================================================
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
# 参数解析
# ==============================================================================
GPU_MODE=false
LOCAL_MODE=false
SKIP_DEPS=false

for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_MODE=true
            ;;
        --local)
            LOCAL_MODE=true
            ;;
        --skip-deps)
            SKIP_DEPS=true
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --gpu        启用 GPU 模式 (需要 CUDA)"
            echo "  --local      使用当前目录的源码"
            echo "  --skip-deps  跳过依赖安装 (假设依赖已安装)"
            echo "  --help       显示帮助信息"
            echo ""
            echo "示例:"
            echo "  全新部署 (CPU):  $0"
            echo "  全新部署 (GPU):  $0 --gpu"
            echo "  本地源码部署:    $0 --local"
            echo "  仅编译部署:      $0 --local --skip-deps"
            exit 0
            ;;
    esac
done

# ==============================================================================
# 检测系统架构
# ==============================================================================
ARCH=$(uname -m)
log_info "检测到系统架构: ${ARCH}"

if [[ "${ARCH}" == "aarch64" ]]; then
    IS_ARM64=true
    log_info "使用 ARM64 配置"
    if [[ "${GPU_MODE}" == true ]]; then
        log_error "ARM64 不支持 GPU 模式"
        exit 1
    fi
elif [[ "${ARCH}" == "x86_64" ]]; then
    IS_ARM64=false
    log_info "使用 x86_64 配置"
else
    log_error "不支持的架构: ${ARCH}"
    exit 1
fi

# ==============================================================================
# 配置变量
# ==============================================================================
INSTALL_DIR="/opt/cllm"
MODEL_DIR="/opt/models"
LOG_DIR="/var/log/cllm"
BUILD_DIR="/opt/cLLM_build"
NPROC=$(nproc)

# CMake 版本
CMAKE_VERSION="3.28.1"

# 检测脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 判断是否在源码目录中
if [[ -f "${SCRIPT_DIR}/../CMakeLists.txt" ]] && [[ -d "${SCRIPT_DIR}/../src" ]]; then
    SOURCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    IN_SOURCE_DIR=true
else
    SOURCE_DIR=""
    IN_SOURCE_DIR=false
fi

# 如果指定了 --local 但不在源码目录，报错
if [[ "${LOCAL_MODE}" == true ]] && [[ "${IN_SOURCE_DIR}" == false ]]; then
    log_error "使用 --local 选项时必须在 cLLM 源码目录中运行"
    log_error "当前目录: ${SCRIPT_DIR}"
    exit 1
fi

# 自动检测：如果在源码目录中，默认使用本地模式
if [[ "${IN_SOURCE_DIR}" == true ]] && [[ "${LOCAL_MODE}" == false ]]; then
    log_info "检测到在源码目录中运行，自动启用本地模式"
    LOCAL_MODE=true
fi

log_info "CPU 核心数: ${NPROC}"
log_info "GPU 模式: ${GPU_MODE}"
log_info "本地模式: ${LOCAL_MODE}"
if [[ "${LOCAL_MODE}" == true ]]; then
    log_info "源码目录: ${SOURCE_DIR}"
fi

# ==============================================================================
# 检查 root 权限
# ==============================================================================
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "请使用 root 权限运行此脚本"
        exit 1
    fi
}

# ==============================================================================
# 检查依赖
# ==============================================================================
check_dependencies() {
    log_step "检查依赖..."
    
    MISSING_DEPS=""
    
    if ! command -v cmake &> /dev/null; then
        MISSING_DEPS="${MISSING_DEPS} cmake"
    fi
    
    if [ ! -f /usr/local/include/nlohmann/json.hpp ] && [ ! -f /usr/include/nlohmann/json.hpp ]; then
        MISSING_DEPS="${MISSING_DEPS} nlohmann-json"
    fi
    
    if [ ! -f /usr/local/include/yaml-cpp/yaml.h ] && [ ! -f /usr/include/yaml-cpp/yaml.h ]; then
        MISSING_DEPS="${MISSING_DEPS} yaml-cpp"
    fi
    
    if [ ! -f /usr/local/include/spdlog/spdlog.h ] && [ ! -f /usr/include/spdlog/spdlog.h ]; then
        MISSING_DEPS="${MISSING_DEPS} spdlog"
    fi
    
    if [ ! -f /usr/local/include/sentencepiece_processor.h ] && [ ! -f /usr/include/sentencepiece_processor.h ]; then
        MISSING_DEPS="${MISSING_DEPS} sentencepiece"
    fi
    
    if [ -n "${MISSING_DEPS}" ]; then
        if [[ "${SKIP_DEPS}" == true ]]; then
            log_error "缺少依赖:${MISSING_DEPS}"
            log_error "请先安装依赖或不使用 --skip-deps 选项"
            exit 1
        else
            log_warn "缺少依赖:${MISSING_DEPS}，将自动安装"
            return 1
        fi
    fi
    
    log_info "依赖检查通过"
    return 0
}

# ==============================================================================
# 第1步: 系统更新和基础工具安装
# ==============================================================================
install_base_tools() {
    log_step "1/8 - 安装基础工具..."
    
    # 更新系统
    yum update -y
    
    # 安装 EPEL 仓库
    yum install -y epel-release
    
    # 安装基础工具
    yum install -y \
        git \
        wget \
        curl \
        vim \
        htop \
        unzip \
        bzip2 \
        patch \
        make \
        autoconf \
        automake \
        libtool \
        pkgconfig \
        zlib-devel \
        openssl-devel \
        libcurl-devel \
        which \
        tree \
        python3 \
        python3-pip \
        python3-devel
    
    log_info "基础工具安装完成"
}

# ==============================================================================
# 第2步: 安装 GCC
# ==============================================================================
install_gcc() {
    log_step "2/8 - 安装 GCC..."
    
    # 检查并启用已安装的 devtoolset
    enable_devtoolset() {
        for ver in 11 10 9; do
            if [ -f /opt/rh/devtoolset-${ver}/enable ]; then
                source /opt/rh/devtoolset-${ver}/enable
                log_info "已启用 devtoolset-${ver}"
                return 0
            fi
        done
        return 1
    }
    
    if [[ "${IS_ARM64}" == true ]]; then
        # ARM64: 检查 GCC 版本，必要时安装
        GCC_VER=$(gcc -dumpversion 2>/dev/null || echo "0")
        if [[ "${GCC_VER%%.*}" -lt 9 ]]; then
            log_info "ARM64: 需要 GCC >= 9，尝试安装..."
            # 尝试安装 devtoolset
            for ver in 11 10 9; do
                if yum list available devtoolset-${ver}-gcc &> /dev/null; then
                    yum install -y devtoolset-${ver}-gcc devtoolset-${ver}-gcc-c++ devtoolset-${ver}-make 2>/dev/null && break
                fi
            done
            
            if ! enable_devtoolset; then
                log_warn "无法安装新版 GCC，使用系统默认版本"
            fi
        else
            enable_devtoolset || true
        fi
    else
        # x86_64: 使用 SCL
        yum install -y centos-release-scl
        # 尝试安装最新可用的 devtoolset
        for ver in 11 10 9; do
            if yum install -y devtoolset-${ver}-gcc devtoolset-${ver}-gcc-c++ devtoolset-${ver}-make 2>/dev/null; then
                break
            fi
        done
        
        # 创建环境脚本
        cat > /etc/profile.d/devtoolset.sh << 'EOFSCRIPT'
#!/bin/bash
for ver in 11 10 9; do
    if [ -f /opt/rh/devtoolset-${ver}/enable ]; then
        source /opt/rh/devtoolset-${ver}/enable
        break
    fi
done
EOFSCRIPT
        chmod +x /etc/profile.d/devtoolset.sh
        enable_devtoolset
    fi
    
    log_info "GCC 版本: $(gcc --version | head -1)"
}

# ==============================================================================
# 第3步: 安装 CMake
# ==============================================================================
install_cmake() {
    log_step "3/8 - 安装 CMake ${CMAKE_VERSION}..."
    
    # 检查是否已安装
    if command -v cmake &> /dev/null; then
        CURRENT_CMAKE=$(cmake --version | head -1 | awk '{print $3}')
        if [[ "$(printf '%s\n' "3.18" "$CURRENT_CMAKE" | sort -V | head -n1)" == "3.18" ]]; then
            log_info "CMake 已安装: ${CURRENT_CMAKE}"
            return
        fi
    fi
    
    cd /tmp
    
    if [[ "${IS_ARM64}" == true ]]; then
        CMAKE_ARCH="aarch64"
    else
        CMAKE_ARCH="x86_64"
    fi
    
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
    tar -xzf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
    cp -r cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/bin/* /usr/local/bin/
    cp -r cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/share/* /usr/local/share/
    rm -rf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}*
    
    ln -sf /usr/local/bin/cmake /usr/bin/cmake
    
    log_info "CMake 版本: $(cmake --version | head -1)"
}

# ==============================================================================
# 第4步: 安装 OpenBLAS (CPU 加速)
# ==============================================================================
install_openblas() {
    log_step "4/8 - 安装 OpenBLAS..."
    
    if [ -f /usr/local/lib/libopenblas.so ] || [ -f /usr/lib64/libopenblas.so ]; then
        log_info "OpenBLAS 已安装"
        return
    fi
    
    if [[ "${IS_ARM64}" == true ]]; then
        yum install -y openblas-devel 2>/dev/null || {
            log_info "从源码编译 OpenBLAS (ARM64)..."
            cd /tmp
            git clone --depth 1 https://github.com/xianyi/OpenBLAS.git
            cd OpenBLAS
            make -j${NPROC} NO_LAPACK=1 NO_LAPACKE=1 NO_FORTRAN=1
            make PREFIX=/usr/local install
            cd /tmp && rm -rf OpenBLAS
        }
    else
        yum install -y openblas-devel
    fi
    
    ldconfig
    log_info "OpenBLAS 安装完成"
}

# ==============================================================================
# 第5步: 安装系统依赖
# ==============================================================================
install_dependencies() {
    log_step "5/8 - 安装系统依赖..."
    
    # 配置库路径
    echo "/usr/local/lib64" > /etc/ld.so.conf.d/local.conf
    echo "/usr/local/lib" >> /etc/ld.so.conf.d/local.conf
    
    # 安装 nlohmann-json
    if [ ! -f /usr/local/include/nlohmann/json.hpp ]; then
        log_info "安装 nlohmann-json..."
        cd /tmp
        git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
        cd json && mkdir -p build && cd build
        cmake .. -DJSON_BuildTests=OFF
        make -j${NPROC} && make install
        cd /tmp && rm -rf json
    fi
    
    # 安装 yaml-cpp
    if [ ! -f /usr/local/include/yaml-cpp/yaml.h ]; then
        log_info "安装 yaml-cpp..."
        cd /tmp
        git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git
        cd yaml-cpp && mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DYAML_BUILD_SHARED_LIBS=ON
        make -j${NPROC} && make install
        cd /tmp && rm -rf yaml-cpp
    fi
    
    # 安装 spdlog
    if [ ! -f /usr/local/include/spdlog/spdlog.h ]; then
        log_info "安装 spdlog..."
        cd /tmp
        git clone --depth 1 --branch v1.12.0 https://github.com/gabime/spdlog.git
        cd spdlog && mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DSPDLOG_BUILD_SHARED=ON
        make -j${NPROC} && make install
        cd /tmp && rm -rf spdlog
    fi
    
    # 安装 SentencePiece (llama.cpp 依赖)
    if [ ! -f /usr/local/include/sentencepiece_processor.h ]; then
        log_info "安装 SentencePiece..."
        cd /tmp
        git clone --depth 1 https://github.com/google/sentencepiece.git
        cd sentencepiece && mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j${NPROC} && make install
        cd /tmp && rm -rf sentencepiece
    fi
    
    ldconfig
    log_info "系统依赖安装完成"
}

# ==============================================================================
# 第6步: 准备源码
# ==============================================================================
prepare_source() {
    log_step "6/8 - 准备源码..."
    
    # 确保环境变量已设置（启用最新可用的 devtoolset）
    for ver in 11 10 9; do
        if [ -f /opt/rh/devtoolset-${ver}/enable ]; then
            source /opt/rh/devtoolset-${ver}/enable
            break
        fi
    done
    
    if [[ "${LOCAL_MODE}" == true ]]; then
        # 本地模式：使用当前目录的源码
        BUILD_DIR="${SOURCE_DIR}"
        log_info "使用本地源码: ${SOURCE_DIR}"

        # 创建符号链接，llama-config.cmake 和 ggml-config.cmake 需要这些目录存在
        # llama.cpp 安装到了 /usr/local，所以我们创建符号链接指向 /usr/local
        # llama-config.cmake 预期 ${SOURCE_DIR}/bin 和 ${SOURCE_DIR}/lib64
        # ggml-config.cmake 预期 ${SOURCE_DIR}/third_party/include
        if [ ! -e "${SOURCE_DIR}/bin" ]; then
            ln -sf /usr/local/bin "${SOURCE_DIR}/bin"
        fi
        if [ ! -e "${SOURCE_DIR}/lib64" ]; then
            ln -sf /usr/local/lib64 "${SOURCE_DIR}/lib64"
        fi
        if [ ! -e "${SOURCE_DIR}/third_party/include" ]; then
            ln -sf /usr/local/include "${SOURCE_DIR}/third_party/include"
        fi
        if [ ! -e "${SOURCE_DIR}/third_party/lib64" ]; then
            ln -sf /usr/local/lib64 "${SOURCE_DIR}/third_party/lib64"
        fi
    else
        # 远程模式：从 git 克隆
        rm -rf ${BUILD_DIR}
        mkdir -p ${BUILD_DIR}
        cd ${BUILD_DIR}
        
        git clone https://github.com/xdongp/cLLM.git
        cd cLLM
        git submodule update --init --recursive
        
        BUILD_DIR="${BUILD_DIR}/cLLM"
        log_info "项目克隆完成: ${BUILD_DIR}"
    fi
}

# ==============================================================================
# 第7步: 编译项目
# ==============================================================================
build_project() {
    log_step "7/8 - 编译项目..."
    
    cd ${BUILD_DIR}
    
    # 确保环境变量
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH
    export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:$PKG_CONFIG_PATH
    
    # 启用新版编译器 (优先使用更高版本)
    if [ -f /opt/rh/devtoolset-11/enable ]; then
        source /opt/rh/devtoolset-11/enable
        log_info "已启用 devtoolset-11"
    elif [ -f /opt/rh/devtoolset-10/enable ]; then
        source /opt/rh/devtoolset-10/enable
        log_info "已启用 devtoolset-10"
    fi
    
    # 检查 GCC 版本
    GCC_VER=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    if [[ "${GCC_VER}" -lt 9 ]]; then
        log_warn "GCC 版本较低 ($(gcc --version | head -1))，可能会有兼容性问题"
    fi
    
    # =========================================
    # 编译 llama.cpp (如果需要)
    # 注意：llama.cpp 已安装到 /usr/local，这里仅检查是否需要重新编译
    # =========================================
    log_info "编译 llama.cpp..."

    cd third_party/llama.cpp
    rm -rf build
    mkdir -p build && cd build

    if [[ "${GPU_MODE}" == true ]]; then
        # GPU 模式
        cmake .. \
            -DGGML_CUDA=ON \
            -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
            -DGGML_CUDA_F16=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF
    else
        # CPU 模式
        if ldconfig -p 2>/dev/null | grep -q libopenblas; then
            log_info "检测到 OpenBLAS，启用 BLAS 加速"
            cmake .. \
                -DGGML_CUDA=OFF \
                -DGGML_BLAS=ON \
                -DGGML_BLAS_VENDOR=OpenBLAS \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLAMA_BUILD_TESTS=OFF \
                -DLLAMA_BUILD_EXAMPLES=OFF
        else
            log_info "未检测到 OpenBLAS，使用纯 CPU 模式"
            cmake .. \
                -DGGML_CUDA=OFF \
                -DCMAKE_BUILD_TYPE=Release \
                -DLLAMA_BUILD_TESTS=OFF \
                -DLLAMA_BUILD_EXAMPLES=OFF
        fi
    fi

    make -j${NPROC}
    make install DESTDIR=/usr/local
    cd ${BUILD_DIR}
    
    # =========================================
    # 编译 cLLM (禁用 tokenizers-cpp 和 LibTorch)
    # =========================================
    log_info "编译 cLLM..."
    rm -rf build
    mkdir -p build && cd build
    
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_TOKENIZERS_CPP=OFF \
        -DUSE_LIBTORCH=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF
    
    make -j${NPROC}
    
    log_info "项目编译完成"
}

# ==============================================================================
# 第8步: 部署
# ==============================================================================
deploy() {
    log_step "8/8 - 部署 cLLM..."
    
    # 创建目录
    mkdir -p ${INSTALL_DIR}/{bin,config,lib}
    mkdir -p ${MODEL_DIR}
    mkdir -p ${LOG_DIR}
    
    # 复制可执行文件
    cp ${BUILD_DIR}/build/bin/cllm_server ${INSTALL_DIR}/bin/
    
    # 复制库文件
    find ${BUILD_DIR}/build -name "*.so*" -exec cp {} ${INSTALL_DIR}/lib/ \; 2>/dev/null || true
    find ${BUILD_DIR}/third_party/llama.cpp/build -name "*.so*" -exec cp {} ${INSTALL_DIR}/lib/ \; 2>/dev/null || true
    
    # 确定 n_threads 和 n_gpu_layers
    if [[ "${GPU_MODE}" == true ]]; then
        N_GPU_LAYERS=99
        N_THREADS=8
        N_BATCH=2048
        N_CTX=32768
        MAX_BATCH_SIZE=8
        CONFIG_FILE="config_llama_cpp_gpu.yaml"
    else
        N_GPU_LAYERS=0
        N_THREADS=${NPROC}
        N_BATCH=512
        N_CTX=8192
        MAX_BATCH_SIZE=2
        CONFIG_FILE="config_llama_cpp_cpu.yaml"
    fi
    
    # 创建配置文件
    cat > ${INSTALL_DIR}/config/${CONFIG_FILE} << EOF
# cLLM 生产环境配置
# 后端: llama.cpp (GGUF)
# Tokenizer: GGUF 内置
# 模式: $(if [[ "${GPU_MODE}" == true ]]; then echo "GPU"; else echo "CPU"; fi)

server:
  host: "0.0.0.0"
  port: 8080
  num_threads: ${N_THREADS}

http:
  max_input_tokens: 4096
  timeout_ms: 60000

logging:
  level: "info"
  file: "${LOG_DIR}/cllm.log"

model:
  # 请修改为实际的模型路径
  path: "/opt/models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m.gguf"
  vocab_size: 152064
  max_context_length: ${N_CTX}
  default_max_tokens: 2048

backend:
  type: "llama_cpp"
  
  llama_cpp:
    n_batch: ${N_BATCH}
    n_threads: ${N_THREADS}
    n_gpu_layers: ${N_GPU_LAYERS}
    n_ctx: ${N_CTX}
    n_seq_max: ${MAX_BATCH_SIZE}
    use_mmap: true
    use_mlock: $(if [[ "${GPU_MODE}" == true ]]; then echo "true"; else echo "false"; fi)

# Tokenizer 使用 GGUF 内置，无需额外配置

scheduler:
  max_batch_size: ${MAX_BATCH_SIZE}
  request_timeout: 600.0
  default_max_tokens: 2048

resources:
  max_context_length: ${N_CTX}
  kv_cache_max_size: $(if [[ "${GPU_MODE}" == true ]]; then echo "32"; else echo "8"; fi)

EOF
    
    # 创建环境变量脚本
    cat > ${INSTALL_DIR}/env.sh << EOF
#!/bin/bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:/usr/local/lib:/usr/local/lib64:\$LD_LIBRARY_PATH
export LLAMA_LOG_LEVEL=warn
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

# 启用新版编译器 (如果可用)
for ver in 11 10 9; do
    [ -f /opt/rh/devtoolset-\${ver}/enable ] && source /opt/rh/devtoolset-\${ver}/enable && break
done
EOF
    chmod +x ${INSTALL_DIR}/env.sh
    
    # 创建服务用户
    useradd -r -s /bin/false -d ${INSTALL_DIR} cllm 2>/dev/null || true
    
    # 设置权限
    chown -R cllm:cllm ${INSTALL_DIR} ${LOG_DIR}
    chmod +x ${INSTALL_DIR}/bin/cllm_server
    
    # GPU 模式添加 video 组
    if [[ "${GPU_MODE}" == true ]]; then
        usermod -aG video cllm 2>/dev/null || true
    fi
    
    # 创建 systemd 服务
    cat > /etc/systemd/system/cllm.service << EOF
[Unit]
Description=cLLM Large Language Model Server ($(if [[ "${GPU_MODE}" == true ]]; then echo "GPU"; else echo "CPU"; fi))
After=network.target

[Service]
Type=simple
User=cllm
Group=cllm
WorkingDirectory=${INSTALL_DIR}

Environment="LD_LIBRARY_PATH=${INSTALL_DIR}/lib:/usr/local/lib:/usr/local/lib64"
Environment="OMP_NUM_THREADS=${N_THREADS}"
Environment="OPENBLAS_NUM_THREADS=${N_THREADS}"

ExecStart=${INSTALL_DIR}/bin/cllm_server --config ${INSTALL_DIR}/config/${CONFIG_FILE}

Restart=always
RestartSec=10
LimitNOFILE=65535
LimitNPROC=65535

StandardOutput=append:${LOG_DIR}/cllm.log
StandardError=append:${LOG_DIR}/cllm.error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # 创建日志轮转配置
    cat > /etc/logrotate.d/cllm << EOF
${LOG_DIR}/*.log {
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
EOF
    
    systemctl daemon-reload
    
    log_info "部署完成"
}

# ==============================================================================
# 系统优化
# ==============================================================================
optimize_system() {
    log_info "系统优化..."
    
    if grep -q "cLLM 配置" /etc/security/limits.conf 2>/dev/null; then
        log_info "系统已优化"
        return
    fi
    
    cat >> /etc/security/limits.conf << EOF

# cLLM 配置
cllm soft nofile 65535
cllm hard nofile 65535
cllm soft nproc 65535
cllm hard nproc 65535
EOF

    cat >> /etc/sysctl.conf << EOF

# cLLM 网络优化
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
EOF
    sysctl -p 2>/dev/null || true
}

# ==============================================================================
# 打印部署信息
# ==============================================================================
print_summary() {
    echo ""
    echo "======================================"
    echo -e "${GREEN}  cLLM 部署完成!${NC}"
    echo "======================================"
    echo ""
    echo "系统架构: ${ARCH}"
    echo "部署模式: $(if [[ "${GPU_MODE}" == true ]]; then echo "GPU"; else echo "CPU"; fi)"
    echo "安装目录: ${INSTALL_DIR}"
    echo "模型目录: ${MODEL_DIR}"
    echo "日志目录: ${LOG_DIR}"
    echo ""
    echo "编译配置:"
    echo "  - Backend: llama.cpp"
    echo "  - Tokenizer: GGUF 内置"
    echo "  - USE_TOKENIZERS_CPP: OFF"
    echo "  - USE_LIBTORCH: OFF"
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 下载 GGUF 模型:"
    echo "   pip3 install huggingface_hub"
    echo "   huggingface-cli download \\"
    echo "       Qwen/Qwen2.5-7B-Instruct-GGUF \\"
    echo "       qwen2.5-7b-instruct-q4_k_m.gguf \\"
    echo "       --local-dir ${MODEL_DIR}/qwen2.5-7b"
    echo ""
    echo "2. 修改配置文件中的模型路径:"
    echo "   vim ${INSTALL_DIR}/config/$(if [[ "${GPU_MODE}" == true ]]; then echo "config_llama_cpp_gpu.yaml"; else echo "config_llama_cpp_cpu.yaml"; fi)"
    echo ""
    echo "3. 启动服务:"
    echo "   systemctl start cllm"
    echo "   systemctl enable cllm"
    echo ""
    echo "4. 检查服务状态:"
    echo "   systemctl status cllm"
    echo "   curl http://localhost:8080/health"
    echo ""
    echo "5. 查看日志:"
    echo "   tail -f ${LOG_DIR}/cllm.log"
    echo ""
}

# ==============================================================================
# 主函数
# ==============================================================================
main() {
    echo ""
    echo "======================================"
    echo "  cLLM CentOS 7 部署脚本"
    echo "  后端: llama.cpp (GGUF)"
    echo "  模式: $(if [[ "${GPU_MODE}" == true ]]; then echo "GPU"; else echo "CPU"; fi)"
    echo "  架构: ${ARCH}"
    if [[ "${LOCAL_MODE}" == true ]]; then
        echo "  源码: 本地 (${SOURCE_DIR})"
    else
        echo "  源码: Git 克隆"
    fi
    echo "======================================"
    echo ""
    
    check_root
    
    START_TIME=$(date +%s)
    
    # 检查依赖
    if [[ "${SKIP_DEPS}" == true ]]; then
        if ! check_dependencies; then
            exit 1
        fi
    else
        if ! check_dependencies; then
            install_base_tools
            install_gcc
            install_cmake
            install_openblas
            install_dependencies
        fi
    fi
    
    prepare_source
    build_project
    deploy
    optimize_system
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    log_info "总耗时: ${MINUTES}分${SECONDS}秒"
    
    print_summary
}

main "$@"
