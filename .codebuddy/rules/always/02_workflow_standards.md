# 🔄 cLLM 工作流程标准

> **优先级**: CRITICAL | AI必须严格遵守的操作流程

---

## 📋 标准工作流程

### 阶段1: 需求理解 (必须执行)

```markdown
收到用户请求后,首先:

1. ✅ 分析请求的核心意图
   - 是新增功能?
   - 是修复bug?
   - 是性能优化?
   - 是重构代码?

2. ✅ 识别影响范围
   - 涉及哪些模块?
   - 需要修改哪些文件?
   - 是否需要更新测试?
   - 是否需要更新文档?

3. ✅ 检查是否有现有设计文档
   - 搜索 docs/ 目录
   - 阅读相关模块设计
   - 理解现有架构约束
```

**示例**:

```
用户请求: "集成tokenizers-cpp作为默认分词"

分析:
- 核心意图: 集成新的分词库
- 影响范围: 
  * include/cllm/tokenizer/hf_tokenizer.h
  * src/tokenizer/hf_tokenizer.cpp
  * src/tokenizer/manager.cpp
  * CMakeLists.txt
- 需要文档: docs/analysis/README_TOKENIZER_MIGRATION.md
```

---

### 阶段2: 信息收集 (必须并行执行)

```markdown
使用工具收集必要信息:

并行工具调用 (一次性执行):
1. read_file() - 读取需要修改的文件
2. search_content() - 搜索相关代码
3. search_file() - 查找相关文件
4. list_files() - 了解目录结构
```

**✅ 正确示例** (并行调用):

```python
# 一次性并行执行
read_file("include/cllm/tokenizer/hf_tokenizer.h")
read_file("src/tokenizer/hf_tokenizer.cpp")
read_file("src/tokenizer/manager.cpp")
search_content("class.*Tokenizer", ".h,.cpp")
search_file("*tokenizer*.h", "include/cllm")
```

**❌ 错误示例** (串行调用):

```python
# ❌ 低效: 一个一个读取
read_file("file1.h")
# ... 分析 ...
read_file("file2.h")
# ... 分析 ...
```

---

### 阶段3: 任务规划 (复杂任务必须)

```markdown
对于复杂任务 (≥3步骤),使用 todo_write 创建任务列表:

条件:
- ✅ 需要修改3个以上文件
- ✅ 涉及多个模块
- ✅ 需要多个步骤完成
- ✅ 用户提供了任务列表

跳过条件:
- ❌ 单文件简单修改
- ❌ 纯信息查询
- ❌ 一步就能完成的操作
```

**✅ 应该创建TODO的场景**:

```
场景1: 集成新库
- [ ] 更新CMakeLists.txt
- [ ] 实现接口适配
- [ ] 更新工厂类
- [ ] 编写测试
- [ ] 更新文档

场景2: 重构模块
- [ ] 提取接口
- [ ] 实现新类
- [ ] 迁移调用点
- [ ] 删除旧代码
- [ ] 运行测试
```

**❌ 不应该创建TODO的场景**:

```
场景1: 简单修改
"给函数添加一行日志" → 直接执行

场景2: 信息查询
"解释这段代码的作用" → 直接回答

场景3: 单步操作
"格式化这个文件" → 直接执行
```

---

### 阶段4: 代码修改 (严格规范)

#### 4.1 修改前验证

```markdown
每次 replace_in_file 前必须:

1. ✅ 已用 read_file 读取了完整文件
2. ✅ old_str 完全匹配 (包括空白符)
3. ✅ 保留原始缩进格式
4. ✅ 检查是否需要添加 #include
5. ✅ 检查命名空间是否正确
```

#### 4.2 修改策略

**单文件修改**:

```cpp
// ✅ 正确: 精确替换
replace_in_file(
    "file.cpp",
    old_str="    int oldFunction() {\n        return 0;\n    }",
    new_str="    int newFunction() {\n        return calculate();\n    }"
)
```

**多处修改** (相邻20行内):

```cpp
// ✅ 正确: 合并修改
replace_in_file(
    "file.cpp",
    old_str="    // 大段代码包含多处修改\n    ...",
    new_str="    // 修改后的大段代码\n    ..."
)
```

**多处修改** (超过20行):

```cpp
// ✅ 正确: 分多次调用
replace_in_file("file.cpp", old_str1, new_str1)
replace_in_file("file.cpp", old_str2, new_str2)
```

**❌ 绝对禁止**:

```cpp
// ❌ 禁止: 重写整个文件
write_to_file("existing_file.cpp", "完整新内容")

// ❌ 禁止: 不精确的old_str
replace_in_file(
    "file.cpp",
    old_str="function()",  // ❌ 可能匹配多处
    new_str="newFunction()"
)
```

#### 4.3 修改后验证

```markdown
每次修改后立即:

1. ✅ 运行 read_lints() 检查语法错误
2. ✅ 检查编译是否通过
3. ✅ 更新TODO状态
4. ✅ 开始下一个任务前标记当前任务完成
```

---

### 阶段5: 测试验证 (关键操作必须)

```markdown
以下情况必须运行测试:

1. ✅ 修改了核心接口
2. ✅ 修改了关键逻辑
3. ✅ 集成了新库
4. ✅ 重构了模块

测试命令:
```bash
# 编译测试
cd build && cmake .. && make -j8

# 运行单元测试
./bin/test_tokenizer

# 运行集成测试
./bin/test_http_server_direct
```
```

---

## 🎯 具体场景工作流

### 场景1: 新增功能

```markdown
步骤:
1. ✅ 阅读设计文档
2. ✅ 搜索相关代码 (search_content)
3. ✅ 创建TODO列表
4. ✅ 实现功能 (分步骤)
5. ✅ 编写测试
6. ✅ 更新文档 (仅在用户要求时)
7. ✅ 运行验证

禁止事项:
- ❌ 不要创建临时脚本
- ❌ 不要生成超大文件
- ❌ 不要跳过测试
```

### 场景2: Bug修复

```markdown
步骤:
1. ✅ 重现问题 (如果有测试用例)
2. ✅ 定位问题代码 (search_content)
3. ✅ 阅读相关代码 (read_file)
4. ✅ 修复问题 (replace_in_file)
5. ✅ 验证修复 (read_lints)
6. ✅ 添加回归测试

禁止事项:
- ❌ 不要盲目修改
- ❌ 不要引入新问题
```

### 场景3: 性能优化

```markdown
步骤:
1. ✅ 分析性能瓶颈 (Profiling)
2. ✅ 查找优化点 (search_content)
3. ✅ 实施优化 (replace_in_file)
4. ✅ 性能测试对比
5. ✅ 验证功能正确性

优化策略:
- ✅ 使用并行处理 (BS::thread_pool)
- ✅ 预分配内存 (reserve)
- ✅ 避免拷贝 (std::move)
- ✅ 缓存重复计算

禁止事项:
- ❌ 不要过早优化
- ❌ 不要牺牲可读性
```

### 场景4: 代码重构

```markdown
步骤:
1. ✅ 理解现有设计
2. ✅ 制定重构计划 (TODO)
3. ✅ 分步骤执行
4. ✅ 每步后验证测试通过
5. ✅ 更新文档

重构原则:
- ✅ 保持接口兼容
- ✅ 小步快跑
- ✅ 每步都可编译
- ✅ 测试保持绿色

禁止事项:
- ❌ 不要大规模重写
- ❌ 不要改变接口
- ❌ 不要跳过测试
```

---

## 🔧 工具使用规范

### read_file

**使用时机**:
- ✅ 需要修改文件前
- ✅ 需要理解代码结构
- ✅ 需要验证old_str

**注意事项**:
- ✅ 优先并行读取多个文件
- ✅ 读取完整文件而非片段
- ❌ 避免重复读取同一文件

### replace_in_file

**使用规则**:
- ✅ old_str必须完全匹配
- ✅ 保留原始格式
- ✅ 一次修改不超过100行
- ❌ 禁止用于创建新文件

**错误处理**:
```markdown
如果replace失败:
1. ✅ 重新read_file验证内容
2. ✅ 检查old_str是否完全匹配
3. ✅ 检查是否有特殊字符
4. ✅ 分解为更小的替换
```

### search_content

**使用技巧**:
- ✅ 使用正则表达式精确匹配
- ✅ 指定文件类型 (fileTypes)
- ✅ 并行搜索多个模式
- ✅ 转义特殊字符

**示例**:
```python
# ✅ 精确搜索
search_content("class\\s+HFTokenizer", ".h,.cpp")
search_content("include.*<tokenizers", ".h,.cpp")

# ✅ 并行搜索
search_content("ITokenizer", ".h")
search_content("TokenizerManager", ".cpp")
```

### execute_command

**安全规则**:
- ✅ 非交互模式 (--yes)
- ✅ 禁用分页 (| cat)
- ⚠️  危险命令需确认 (requires_approval=true)
- ❌ 禁止修改git config
- ❌ 禁止force push

**示例**:
```bash
# ✅ 安全命令
execute_command("mkdir -p dir", requires_approval=false)
execute_command("cat file.txt | head -20", requires_approval=false)

# ⚠️  需要确认
execute_command("rm -rf dir/", requires_approval=true)
execute_command("git reset --hard", requires_approval=true)
```

---

## 📊 TODO管理规范

### 创建TODO

```python
# ✅ 首次创建 (merge=false)
todo_write(
    merge=false,
    todos='[
        {"id":"1","status":"in_progress","content":"实现功能A"},
        {"id":"2","status":"pending","content":"实现功能B"},
        {"id":"3","status":"pending","content":"编写测试"}
    ]'
)
```

### 更新TODO状态

```python
# ✅ 更新状态 (merge=true, 只包含变化的项)
todo_write(
    merge=true,
    todos='[
        {"id":"1","status":"completed","content":"实现功能A"},
        {"id":"2","status":"in_progress","content":"实现功能B"}
    ]'
)
```

### TODO状态管理

```markdown
规则:
1. ✅ 同时只有一个任务为 in_progress
2. ✅ 完成任务立即标记 completed
3. ✅ 开始新任务时批量更新状态
4. ✅ 所有TODO必须包含 id, status, content

状态流转:
pending → in_progress → completed
              ↓
          cancelled (不再需要)
```

### 批量更新示例

```python
# ✅ 推荐: 批量更新 (完成1, 开始2)
todo_write(
    merge=true,
    todos='[
        {"id":"1","status":"completed","content":"任务1"},
        {"id":"2","status":"in_progress","content":"任务2"}
    ]'
)

# ❌ 低效: 分两次更新
todo_write(merge=true, todos='[{"id":"1","status":"completed",...}]')
todo_write(merge=true, todos='[{"id":"2","status":"in_progress",...}]')
```

---

## 🚨 错误预防检查清单

### 修改前自检

```markdown
- [ ] 是否已read_file读取目标文件?
- [ ] old_str是否完全匹配 (包括空白)?
- [ ] 是否需要添加 #include?
- [ ] 命名空间是否正确?
- [ ] 条件编译宏是否完整?
- [ ] 是否需要同步修改 .h 和 .cpp?
- [ ] 是否需要更新 CMakeLists.txt?
```

### 修改后自检

```markdown
- [ ] 是否运行了 read_lints?
- [ ] 是否有编译错误?
- [ ] 是否需要更新测试?
- [ ] 是否需要更新文档?
- [ ] TODO状态是否已更新?
- [ ] 是否需要验证功能?
```

---

## 💡 最佳实践

### 1. 并行优先

```python
# ✅ 推荐: 并行执行
read_file("file1.h")
read_file("file2.h")
read_file("file3.h")
search_content("pattern1")
search_content("pattern2")

# ❌ 低效: 串行执行
read_file("file1.h")
# ... 等待 ...
read_file("file2.h")
```

### 2. 最小化上下文

```python
# ✅ 推荐: 精确替换小范围
replace_in_file(
    "file.cpp",
    old_str="    bool load() {\n        return false;\n    }",
    new_str="    bool load() {\n        return loadImpl();\n    }"
)

# ❌ 低效: 替换大范围
replace_in_file(
    "file.cpp",
    old_str="整个类的实现 (500行)",
    new_str="整个类的新实现 (500行)"
)
```

### 3. 渐进式修改

```python
# ✅ 推荐: 分步骤执行
# Step 1: 添加新接口
replace_in_file(...) 

# Step 2: 实现接口
replace_in_file(...)

# Step 3: 更新调用点
replace_in_file(...)

# Step 4: 删除旧代码
replace_in_file(...)
```

### 4. 及时验证

```python
# ✅ 推荐: 每步后验证
replace_in_file(...)
read_lints("file.cpp")

replace_in_file(...)
read_lints("file.cpp")
```

---

## 📚 常见错误与解决方案

### 错误1: replace_in_file 匹配失败

```markdown
症状: "old_str not found in file"

解决:
1. ✅ 重新 read_file 读取最新内容
2. ✅ 检查是否有空白符差异
3. ✅ 检查是否有特殊字符
4. ✅ 扩大 old_str 范围增加唯一性
```

### 错误2: 编译错误

```markdown
症状: read_lints 报告错误

解决:
1. ✅ 检查是否缺少 #include
2. ✅ 检查命名空间是否正确
3. ✅ 检查条件编译宏
4. ✅ 检查语法错误
```

### 错误3: 测试失败

```markdown
症状: 运行测试不通过

解决:
1. ✅ 检查是否破坏了接口
2. ✅ 检查是否改变了行为
3. ✅ 检查是否需要更新测试用例
4. ✅ 回滚到上一个工作状态
```

---

**最后更新**: 2026-01-11  
**维护者**: cLLM Workflow Team
