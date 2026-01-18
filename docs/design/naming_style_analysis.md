# runningRequests_ 命名风格分析

## 当前命名情况

### 数据结构类型
1. **RequestQueue** (类)
   - 成员变量：`requestQueue_` (下划线后缀)
   - 类型：`RequestQueue` (封装队列逻辑的类)
   - 作用：FIFO队列，管理等待处理的请求

2. **runningRequests_** (成员变量)
   - 类型：`std::map<size_t, RequestState>`
   - 作用：状态映射表，管理活跃请求（PENDING/PROCESSING状态）
   - 功能：状态管理、并发控制、超时检测

3. **completedRequests_** (成员变量)
   - 类型：`std::map<size_t, RequestState>`
   - 作用：历史记录映射表，存储已完成的请求

## 命名风格分析

### 选项1：保持现状（推荐）

**理由**：
- **语义准确性**：`runningRequests_` 不是队列，而是映射表（`std::map`）
- **命名约定一致性**：私有成员变量使用下划线后缀符合C++约定
- **清晰区分**：
  - `RequestQueue` = 队列（类）
  - `runningRequests_` = 映射表（成员变量）
  - `completedRequests_` = 映射表（成员变量）

**优点**：
- 不需要修改代码
- 命名准确反映数据结构类型
- 不会误导读者

**缺点**：
- 命名风格不完全统一（一个用Queue，两个用Requests_）

### 选项2：将 runningRequests_ 改为类似 Queue 的名称（不推荐）

**可能的名称**：
- `runningRequestQueue_` - 会误导，因为它不是队列
- `activeRequestQueue_` - 同样会误导
- `RequestStateMap` - 不够明确

**缺点**：
- 语义不准确：使用"Queue"命名但实际是`std::map`
- 容易误导：读者会期望FIFO行为，但实际是键值映射
- 需要大量代码修改

### 选项3：将所有都改为描述性命名（可选）

**可能的名称**：
- `runningRequests_` → `activeRequests_` - 更简洁，强调"活跃"状态
- `runningRequests_` → `processingRequests_` - 强调"处理中"
- `runningRequests_` → `concurrentRequests_` - 强调"并发"

**优点**：
- `activeRequests_` 更简洁，语义清晰（活跃的请求）
- 避免了"running"的歧义（running可以指"运行中"或"正在运行的进程"）

**缺点**：
- 需要修改代码（虽然改动范围可控）
- `activeRequests_` 可能不如 `runningRequests_` 直观

## 推荐方案

### 方案A：保持现状（最推荐）

**理由**：
1. **数据结构差异明确**：
   - `RequestQueue` = 队列类（FIFO）
   - `runningRequests_` = 映射表（键值对）
   - 命名已经准确反映了数据结构类型

2. **命名约定符合C++规范**：
   - 私有成员变量使用下划线后缀
   - 类名使用PascalCase
   - 变量名使用camelCase + 下划线

3. **语义清晰**：
   - `requestQueue_` = 请求队列
   - `runningRequests_` = 运行中的请求映射
   - `completedRequests_` = 已完成的请求映射

### 方案B：微调命名（如果必须统一）

**建议**：将 `runningRequests_` 改为 `activeRequests_`

**理由**：
- `activeRequests_` 更简洁，语义更清晰（活跃的请求）
- 避免了"running"可能的歧义
- 与 `completedRequests_` 形成更好的对比（active vs completed）

**修改范围**：
- 代码：`src/scheduler/*.cpp`, `include/cllm/scheduler/*.h`
- 文档：`docs/design/concurrent_request_management_design.md`

## 结论

**推荐：保持现状（方案A）**

原因：
1. 当前的命名已经准确反映了数据结构类型（队列 vs 映射表）
2. 符合C++命名约定（成员变量下划线后缀）
3. 语义清晰，不会误导读者
4. 无需修改代码，避免引入潜在问题

**如果一定要统一命名风格**，建议改为 `activeRequests_`（方案B），因为它：
- 更简洁
- 语义更准确（"活跃"比"运行中"更准确）
- 与 `completedRequests_` 形成清晰对比

---

**最终建议**：保持现状，因为命名已经准确反映了数据结构类型和用途。
