# tokenizers-cpp安装指南

本文档说明如何安装tokenizers-cpp库,以支持HuggingFace格式的tokenizer。

## 快速安装

### macOS

```bash
# 方法1: 使用自动安装脚本(推荐)
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
./scripts/install_tokenizers_cpp.sh

# 方法2: 手动安装
brew install rust
git clone https://github.com/mlc-ai/tokenizers-cpp
cd tokenizers-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/homebrew
make -j8
sudo make install
```

### Linux (Ubuntu/Debian)

```bash
# 使用自动安装脚本(推荐)
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
./scripts/install_tokenizers_cpp.sh

# 或手动安装
sudo apt-get update
sudo apt-get install -y cargo rustc cmake g++
git clone https://github.com/mlc-ai/tokenizers-cpp
cd tokenizers-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

## 验证安装

```bash
# 检查头文件
ls /opt/homebrew/include/tokenizers/tokenizers_cpp.h  # macOS
ls /usr/local/include/tokenizers/tokenizers_cpp.h     # Linux

# 检查库文件
ls /opt/homebrew/lib/libtokenizers_cpp.*  # macOS
ls /usr/local/lib/libtokenizers_cpp.*     # Linux
```

## 编译cLLM启用HF支持

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
mkdir -p build && cd build

# tokenizers-cpp默认已启用 (USE_TOKENIZERS_CPP=ON)
cmake ..

# 或显式启用
cmake .. -DUSE_TOKENIZERS_CPP=ON

make -j8
```

## 禁用HF支持(仅使用NativeTokenizer)

```bash
cd build
cmake .. -DUSE_TOKENIZERS_CPP=OFF
make -j8
```

## 故障排查

### 问题1: 找不到tokenizers-cpp

**症状:**
```
CMake Warning: tokenizers-cpp not found, falling back to NativeTokenizer
```

**解决方案:**
1. 确认已安装tokenizers-cpp
2. 检查安装路径是否正确
3. 手动指定路径:
```bash
cmake .. \
  -DUSE_TOKENIZERS_CPP=ON \
  -DTOKENIZERS_INCLUDE_DIR=/opt/homebrew/include \
  -DTOKENIZERS_LIBRARY=/opt/homebrew/lib/libtokenizers_cpp.dylib
```

### 问题2: Rust未安装

**症状:**
```
error: cargo not found
```

**解决方案:**
```bash
# macOS
brew install rust

# Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 问题3: 编译失败

**症状:**
```
error: linking with `cc` failed
```

**解决方案:**
- macOS: 确保安装了Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```
- Linux: 安装构建工具
  ```bash
  sudo apt-get install build-essential
  ```

## 依赖关系

tokenizers-cpp需要以下依赖:
- **Rust工具链** (rustc 1.70+)
- **CMake** (3.16+)
- **C++编译器** (支持C++17)

## 支持的模型

启用tokenizers-cpp后支持的模型:
- ✅ Qwen/Qwen2/Qwen3系列
- ✅ DeepSeek/DeepSeek-V3系列
- ✅ GPT-2/GPT-J/GPT-NeoX
- ✅ BERT/RoBERTa/DeBERTa
- ✅ Mistral/Mixtral
- ✅ Gemma/Gemma-2
- ✅ Yi系列
- ✅ ChatGLM
- ✅ Baichuan
- ✅ 所有使用tokenizer.json格式的HuggingFace模型

## 参考资源

- tokenizers-cpp GitHub: https://github.com/mlc-ai/tokenizers-cpp
- HuggingFace tokenizers: https://github.com/huggingface/tokenizers
- cLLM项目文档: [README_TOKENIZER_MIGRATION.md](./analysis/README_TOKENIZER_MIGRATION.md)
