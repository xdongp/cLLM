# LibTorch推理乱码修复记录

## 现象
- 调用 `POST /generate`，当 `prompt` 超过约 16 个中文字符（或对应 token 数超过约 16）时，返回结果出现明显“乱码/随机字符”。
- 当输入较短（约 16 个字符以内）时表现正常。

复现示例：
- `curl -X POST http://localhost:18080/generate -H "Content-Type: application/json" -d '{"prompt":"介绍人工智能……(超过16字)","max_tokens":50,"temperature":0.7,"stream":false}'`

## 影响范围
- 仅在使用 **LibTorch traced（TorchScript trace）模型** 的推理路径下触发。
- 根因与 traced 模型的 **静态 `seq_len`** 强相关；当 traced 固化的 `seq_len` 约为 16 时，超长输入会触发该问题。

## 根因分析
### 关键事实
- traced 模型常会把输入的 `seq_len`（序列长度）固化成静态维度（例如 16）。
- 当输入 token 数 `len` 大于 traced 固化的 `tracedSeqLen_` 时，模型实际只能产出长度为 `tracedSeqLen_` 的 logits。

### 触发链路（概念）
1. 请求进入 `/generate`，`prompt` 被 tokenizer 转为 `input_ids`。
2. LibTorch 后端在 `forwardBatch()` 中推理，得到 logits。
3. 调度/采样逻辑通常会取“最后一个 token 位置”的 logits 来采样下一个 token。

### 具体错误点
在原实现里，当 `len > tracedSeqLen_` 时：
- 模型只能返回前 `tracedSeqLen_` 个位置 logits；
- 代码将这段 logits **从序列开头**拷贝到请求 logits 缓冲区；
- 导致请求的“最后一个 token 位置” logits **没有被有效写入**（保持为 0）。

当采样器在“全 0 logits”上进行采样时，softmax 会退化成近似均匀分布，结果等价于“从词表随机挑 token”，最终解码表现为“乱码/噪声文本”。

## 修复方案
修复集中在 `src/inference/libtorch_backend.cpp` 的 `LibTorchBackend::forwardBatch()`：

1. **超长输入滑动窗口**：当 `len > tracedSeqLen_` 时，不再截取“前缀”，改为保留 **尾部 `tracedSeqLen_` 个 tokens** 作为推理输入（tail window）。
2. **logits 对齐到请求末尾**：当输出长度 `actualOutputLen < len` 时，将可用 logits 拷贝到请求序列的**末尾区间**（而不是从 0 开始贴），保证用于采样的“最后位置 logits”始终来自有效输出。

修复后：即使 traced `seq_len` 仍是 16，超长输入也不会因为最后位置 logits 缺失而退化成随机乱码。

## 验证方式
- 重启服务，确保加载的是最新编译产物。
- 用明显超过 16 个中文字符的 `prompt` 进行请求，观察返回 `text` 不再出现随机乱码。
- 可对比：短输入、长输入均能得到稳定可读输出。

## 注意事项与后续建议
- 如果 traced 的 `seq_len` 确实只有 16，那么模型“有效上下文”仍然主要来自 **最后 16 个 tokens**（滑动窗口尾部），这属于 traced 静态维度的客观限制。
- 若要真正支持更长上下文：建议导出支持更大 `seq_len` 的模型，或改用支持动态 shape 的导出方式（例如 script/动态维度方案，视模型与导出链路而定）。
