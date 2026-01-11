# .gitignore 配置说明

## 📋 概述

本文档说明 cLLM 项目的 `.gitignore` 配置，确保只有必要的源代码和配置文件上传到 GitHub，避免大文件和临时文件污染仓库。

---

## 🎯 忽略策略

### ✅ 纳入版本控制

**源代码**:
- `src/` - C++ 源代码
- `include/` - 头文件
- `examples/` - 示例代码
- `tests/` - 测试代码

**配置和构建**:
- `CMakeLists.txt` - CMake 配置
- `Makefile` - Make 配置
- `config/*.yaml` - 配置文件模板

**文档**:
- `docs/` - 所有文档
- `README*.md` - 说明文档
- `.codebuddy/rules/` - AI 开发规则

**脚本**:
- `scripts/` - 构建和工具脚本
- `model/*.py` - 模型导出脚本

---

### ❌ 忽略内容

#### 1. 编译产物 (自动生成)

```
build/              # CMake 构建目录
bin/               # 可执行文件
*.o, *.so, *.a     # 编译中间文件
*.exe, *.dll       # Windows 可执行文件
```

**原因**: 可以通过构建系统重新生成

#### 2. 第三方库 (通过依赖管理)

```
third_party/       # 第三方源码
llama.cpp/         # llama.cpp 子模块
_deps/             # CMake FetchContent
sentencepiece/     # SentencePiece 子模块
```

**原因**: 
- 体积庞大 (数百MB)
- 可通过 Git submodule 或 CMake 自动下载
- 避免重复存储

**如何恢复**:
```bash
# 初始化子模块
git submodule update --init --recursive

# CMake 会自动下载依赖
cmake -B build
```

#### 3. 模型文件 (大文件)

```
*.bin              # 二进制模型文件
*.safetensors      # SafeTensors 格式
*.gguf             # GGUF 格式
*.pt, *.pth        # PyTorch 模型
model/Qwen/        # Qwen 模型目录
```

**原因**: 
- 单个文件可达几GB
- GitHub 限制单文件 100MB
- 应使用 Git LFS 或外部存储

**替代方案**:
- 使用 Git LFS (Large File Storage)
- 使用 Hugging Face Model Hub
- 使用云存储 (S3, OSS 等)
- 在 README 中提供下载链接

#### 4. 日志和输出文件

```
*.log              # 日志文件
logs/              # 日志目录
test_results/      # 测试结果
*_results.json     # Benchmark 结果
```

**原因**: 
- 运行时生成
- 内容频繁变化
- 不需要版本控制

#### 5. IDE 配置 (个人设置)

```
.vscode/           # VS Code 配置
.idea/             # IntelliJ/CLion 配置
*.swp              # Vim 临时文件
.DS_Store          # macOS 元数据
```

**原因**: 
- 每个开发者配置不同
- 会产生合并冲突
- 应使用项目级配置

#### 6. CodeBuddy 缓存

```
.codebuddy/context/       # 上下文缓存
.codebuddy/memory/cache/  # 记忆缓存
```

**原因**: 
- 运行时缓存
- 个人特定内容
- 保留规则和配置即可

---

## 📊 忽略效果

### 仓库大小对比

| 场景 | 大小 | 说明 |
|------|------|------|
| **不使用 .gitignore** | ~2.5GB | 包含 build/, third_party/, 模型文件 |
| **使用 .gitignore** | ~50MB | 只包含源代码和文档 |
| **节省空间** | **98%** | 大幅减小仓库体积 |

### 忽略的主要内容

| 类型 | 大小 | 说明 |
|------|------|------|
| `build/` | ~500MB | CMake 构建产物 |
| `llama.cpp/` | ~300MB | 第三方库源码 |
| `third_party/` | ~200MB | GoogleTest, Eigen 等 |
| 模型文件 | ~1.5GB | 如果存在 |
| 其他 | ~50MB | 日志、临时文件等 |

---

## 🚀 使用指南

### 首次克隆仓库

```bash
# 1. 克隆仓库
git clone https://github.com/your-username/cLLM.git
cd cLLM

# 2. 初始化子模块 (如果使用)
git submodule update --init --recursive

# 3. 构建项目
cmake -B build -S cpp/cLLM
cmake --build build

# 4. 下载模型 (如果需要)
# 参考 README.md 中的模型下载说明
```

### 检查忽略状态

```bash
# 查看被忽略的文件
git status --ignored

# 检查特定文件是否被忽略
git check-ignore -v build/CMakeCache.txt

# 查看所有被忽略的文件
git ls-files --others --ignored --exclude-standard
```

### 添加例外

如果需要添加被忽略的文件，使用 `!` 前缀：

```gitignore
# 忽略所有 .bin 文件
*.bin

# 但保留特定的小文件
!tests/small_model.bin
```

---

## 🔧 特殊情况处理

### 1. 需要提交模型文件

**方案 A: 使用 Git LFS**

```bash
# 安装 Git LFS
git lfs install

# 跟踪模型文件
git lfs track "*.bin"
git lfs track "*.safetensors"

# 提交
git add .gitattributes
git add model/small_model.bin
git commit -m "Add model with Git LFS"
```

**方案 B: 使用外部存储**

```bash
# 不提交模型文件，在 README 中说明下载方式
echo "下载模型: https://huggingface.co/Qwen/Qwen3-0.6B" >> README.md
```

### 2. 需要分享 IDE 配置

创建 `.vscode/settings.json.example`:

```json
{
  "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools",
  "cmake.buildDirectory": "${workspaceFolder}/build"
}
```

提交示例文件，用户自行复制:
```bash
cp .vscode/settings.json.example .vscode/settings.json
```

### 3. 需要提交测试结果

将特定结果移到 `docs/` 目录:

```bash
# 将关键结果移到文档目录
cp test_results.json docs/benchmark_results/v1.0_results.json
git add docs/benchmark_results/v1.0_results.json
```

---

## 📋 .gitignore 规则说明

### 通配符

| 规则 | 说明 | 示例 |
|------|------|------|
| `*` | 匹配任意字符 | `*.o` 匹配所有 .o 文件 |
| `**` | 匹配任意路径 | `**/build` 匹配所有 build 目录 |
| `?` | 匹配单个字符 | `?.log` 匹配单字符名称的日志 |
| `[abc]` | 匹配括号内字符 | `*.[oa]` 匹配 .o 或 .a |
| `!` | 取反 (不忽略) | `!important.log` 保留该文件 |
| `#` | 注释 | `# This is a comment` |

### 目录匹配

```gitignore
# 忽略根目录的 build/
/build/

# 忽略所有 build/ 目录
build/

# 忽略 src/build/ 但不忽略 test/build/
src/build/
```

---

## ✅ 验证清单

提交前检查:

- [ ] 所有 `.cpp` 和 `.h` 文件已添加
- [ ] `CMakeLists.txt` 已添加
- [ ] `README.md` 和文档已添加
- [ ] 配置模板文件已添加
- [ ] `.codebuddy/rules/` 已添加
- [ ] **build/ 被忽略**
- [ ] **third_party/ 被忽略**
- [ ] **模型文件被忽略**
- [ ] **日志文件被忽略**
- [ ] **IDE 配置被忽略**

验证命令:

```bash
# 查看将要提交的文件
git status

# 检查是否有大文件
git ls-files | xargs du -h | sort -h | tail -20

# 确认被忽略的文件
git status --ignored
```

---

## 🔄 更新 .gitignore

### 已提交文件的处理

如果文件已被 Git 跟踪，添加到 `.gitignore` 后不会自动忽略。需要手动移除:

```bash
# 从 Git 移除但保留本地文件
git rm --cached build/CMakeCache.txt

# 从 Git 移除整个目录
git rm -r --cached build/

# 提交更改
git commit -m "Update .gitignore and remove cached files"
```

### 全局 .gitignore

为所有项目设置:

```bash
# 创建全局 .gitignore
vim ~/.gitignore_global

# 配置 Git 使用全局 .gitignore
git config --global core.excludesfile ~/.gitignore_global
```

内容示例:
```gitignore
# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# 编辑器
*.swp
*~
```

---

## 📞 常见问题

### Q1: 为什么 build/ 没有被忽略？

**A**: 可能已被 Git 跟踪，需要先移除:
```bash
git rm -r --cached build/
git commit -m "Remove build/ from tracking"
```

### Q2: 如何查看哪些文件被忽略？

**A**: 使用以下命令:
```bash
git status --ignored
git ls-files --others --ignored --exclude-standard
```

### Q3: 如何提交被忽略的文件？

**A**: 使用 `-f` 强制添加:
```bash
git add -f important_file.log
```

### Q4: .gitignore 不生效怎么办？

**A**: 清除 Git 缓存:
```bash
git rm -r --cached .
git add .
git commit -m "Reset .gitignore"
```

### Q5: 如何测试 .gitignore 规则？

**A**: 使用 `check-ignore`:
```bash
git check-ignore -v build/CMakeCache.txt
```

---

## 🎯 最佳实践

### 1. 提交前检查

```bash
# 查看文件大小
git ls-files | xargs du -h | sort -h | tail -10

# 查看将要提交的内容
git diff --cached --stat
```

### 2. 使用 .gitattributes

配合 `.gitignore` 使用:

```gitattributes
# 文本文件使用 LF 换行
*.cpp text eol=lf
*.h text eol=lf
*.md text eol=lf

# 二进制文件
*.bin binary
*.so binary
*.dylib binary
```

### 3. 定期清理

```bash
# 查看仓库大小
git count-objects -vH

# 清理历史中的大文件 (谨慎使用)
git filter-branch --tree-filter 'rm -f large_file.bin' HEAD
```

---

## 📚 参考资源

- [Git 官方文档 - .gitignore](https://git-scm.com/docs/gitignore)
- [GitHub .gitignore 模板](https://github.com/github/gitignore)
- [Git LFS 文档](https://git-lfs.github.com/)
- [.gitignore 在线生成器](https://www.toptal.com/developers/gitignore)

---

**最后更新**: 2026-01-11  
**版本**: v1.0  
**维护者**: cLLM Core Team
