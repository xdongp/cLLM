#!/bin/bash
set -e

# tokenizers-cpp安装脚本
# 用于快速安装HuggingFace tokenizers-cpp库

echo "======================================"
echo "  tokenizers-cpp安装脚本"
echo "======================================"
echo ""

# 检测操作系统
OS=$(uname -s)
echo "检测到操作系统: $OS"

# 检查Rust是否已安装
if command -v rustc &> /dev/null; then
    echo "✅ Rust已安装: $(rustc --version)"
else
    echo "❌ Rust未安装，开始安装..."
    if [[ "$OS" == "Darwin" ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install rust
        else
            echo "请先安装Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OS" == "Linux" ]]; then
        # Linux
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo "不支持的操作系统: $OS"
        exit 1
    fi
    echo "✅ Rust安装完成"
fi

# 设置安装前缀
if [[ "$OS" == "Darwin" ]]; then
    INSTALL_PREFIX="/opt/homebrew"
else
    INSTALL_PREFIX="/usr/local"
fi

echo ""
echo "安装前缀: $INSTALL_PREFIX"

# 检查是否已安装tokenizers-cpp
if [ -f "$INSTALL_PREFIX/include/tokenizers_cpp.h" ] || [ -f "$INSTALL_PREFIX/include/tokenizers/tokenizers_cpp.h" ]; then
    echo "⚠️  tokenizers-cpp似乎已安装"
    echo -n "是否重新安装? (y/n): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "跳过安装"
        exit 0
    fi
fi

# 下载并编译tokenizers-cpp
echo ""
echo "正在下载tokenizers-cpp..."
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

git clone --depth 1 https://github.com/mlc-ai/tokenizers-cpp.git
cd tokenizers-cpp

echo ""
echo "正在初始化子模块..."
git submodule update --init --recursive

echo ""
echo "正在编译tokenizers-cpp..."
mkdir -p build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON

# 使用所有CPU核心编译
if [[ "$OS" == "Darwin" ]]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

echo "使用 $NPROC 个CPU核心编译..."
make -j"$NPROC"

echo ""
echo "正在安装tokenizers-cpp..."
sudo make install

# 清理临时文件
cd ../../..
rm -rf "$TMP_DIR"

echo ""
echo "======================================"
echo "✅ tokenizers-cpp安装成功!"
echo "======================================"
echo ""
echo "头文件路径: $INSTALL_PREFIX/include/tokenizers/"
echo "库文件路径: $INSTALL_PREFIX/lib/"
echo ""
echo "现在您可以编译cLLM项目了:"
echo "  cd build"
echo "  cmake .. -DUSE_TOKENIZERS_CPP=ON"
echo "  make -j$NPROC"
echo ""
