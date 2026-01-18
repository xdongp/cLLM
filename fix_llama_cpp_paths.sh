#!/bin/bash
# 修复 llama.cpp 路径配置

echo "修复 llama.cpp 路径配置..."

# 备份原始文件
cp third_party/llama.cpp/build/llama-config.cmake third_party/llama.cpp/build/llama-config.cmake.bak

# 修改路径：将 lib 改为 bin
sed -i '' 's|set_and_check(LLAMA_LIB_DIR.*|set_and_check(LLAMA_LIB_DIR     "${PACKAGE_PREFIX_DIR}/bin")|' third_party/llama.cpp/build/llama-config.cmake

echo "✅ 路径配置已修复"
echo "原路径: lib"
echo "新路径: bin"

# 验证修改
grep "LLAMA_LIB_DIR" third_party/llama.cpp/build/llama-config.cmake
