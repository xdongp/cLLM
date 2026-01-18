## 文档信息

- **文档名称**: GGUF修复文档.md
- **适用范围**: cLLM（Kylin 后端）加载并推理 GGUF 模型（以 Qwen3 为代表）时，出现“输出乱码/不符合预期（例如 `1+1=` 不能得到 `2`）”的修复与验证。
- **目标读者**: 负责推理内核（Attention/RoPE/Norm/量化/采样）与 GGUF 加载链路的开发者。

---

## 背景与现象

在 `tests/test_hello_inference.cpp` 中输入 `1+1=`，输出出现乱码或随机文本，而不是期望的 `2`（或至少是稳定、语义一致的答案）。

### 重要澄清（避免误判）

- **这不是“算术模块算错”**：项目中并不存在把字符串 `1+1=` 解析成表达式并计算的逻辑。
- 该问题本质是：**Tokenizer → 前向计算（Attention/RoPE/Norm/FFN/量化）→ 采样/停止条件** 这条链路的实现与参考实现不一致，导致模型整体行为退化。

---

## 错误定位摘要（当前最可能的根因）

结合当前代码审查与 `third_party/llama.cpp` 的 Qwen3 参考实现对比，最可能的主因集中在以下三类（按优先级排序）：

1. **GQA（Grouped Query Attention）实现不正确**
   - Qwen3 通常满足 `num_attention_heads != num_key_value_heads`。
   - 我方实现当前将 K/V 也按 `num_attention_heads` 拆分，导致 KV 的 head 结构与真实模型不一致，attention 语义被破坏。

2. **缺失 Q/K 的额外归一化（`attn_q_norm` / `attn_k_norm`）**
   - llama.cpp 的 Qwen3 会在 RoPE 前分别对 Q 与 K 做 RMSNorm。
   - 我方目前只有 `norm1/norm2`（block pre-norm），缺少 Q/K 的独立 norm，属于“结构不一致”。

3. **RoPE 参数（尤其 `rope_theta` / `freq_base`）未正确传递**
   - Qwen3 的 `rope_theta` 往往为 `1000000`，而当前实现存在硬编码 `10000` 的风险（即使构造函数带了参数，也需要确认真正被用于 RoPE 表）。

---

## 与 `third_party/llama.cpp` 的对比结果（关键差异点）

下面列出会直接导致输出退化的“结构级差异”。这些差异比采样参数（temperature/top_k/top_p）影响更大，必须优先修复。

### 1) Head 维度与 head 数（GQA）

- **llama.cpp（Qwen3）**：
  - Q reshape 为 `[head_dim, n_head, n_tokens]`
  - K/V reshape 为 `[head_dim, n_head_kv, n_tokens]`
  - attention 计算中需要将 KV head 映射/重复到 Q head（
    \(gqa = n\_head / n\_head\_kv\)）。

- **我方现状（风险点）**：
  - `MultiHeadAttention` 只有一个 `numHeads_`，对 Q 和 KV 采用同一 head 数进行拆分。
  - 当 `n_head != n_head_kv` 时，KV 的 head_dim/head_count 会被“重解释”，造成不可恢复的语义偏差。

### 2) Q/K 额外 RMSNorm（Qwen3 特性）

- **llama.cpp（Qwen3）**：对 Q 与 K 各自做 RMSNorm（`attn_q_norm` / `attn_k_norm`），再做 RoPE。
- **我方现状**：仅对输入做 `norm1`，然后直接 `Wq/Wk/Wv` 投影 + RoPE + attention。

### 3) RoPE 参数来源

- **llama.cpp**：RoPE 参数来自 GGUF/hparams（`freq_base` 通常对应 `rope_theta`），并使用 `ggml_rope_ext` 支持扩展参数。
- **我方现状**：RoPE 为简化实现，需确保至少 `theta`（`freq_base`）正确读取并贯通；扩展参数可作为后续增强。

---

## 修复方案（可落地的实施清单）

### P0（必须优先完成，否则调参无意义）

#### P0.1 正确实现 GQA：区分 Q heads 与 KV heads，并实现 KV 映射/重复

- **目标**：让 attention 的张量语义与 Qwen3 参考实现一致。

- **建议改动点**：
  - **接口改造**：将 `MultiHeadAttention` 从“单 `numHeads_`”升级为：
    - `numQHeads`（= `num_attention_heads`）
    - `numKVHeads`（= `num_key_value_heads`）
    - `headDim`（通常 `hidden/numQHeads`）
  - **reshape 语义**：
    - Q: `[B, n_head, S, head_dim]`
    - K/V: `[B, n_head_kv, S, head_dim]`
  - **KV 映射策略**（任选其一，推荐索引映射，最小内存开销）：
    - **映射法**：在计算 attention 时，对每个 `q_head` 使用 `kv_head = q_head / gqa` 的 K/V。
    - **repeat_kv 法**：显式把 K/V 扩展到 `[B, n_head, S, head_dim]`（更直观但耗内存）。

- **验收标准**：
  - 对 Qwen3：`n_head=16, n_head_kv=8` 时，代码不会再走“强行 kvDim % numHeads” 的路径；
  - attention 的 K/V head_dim 与 Q head_dim 一致（均为 `head_dim`），不再出现 `minHeadDim` 截断计算。

#### P0.2 补齐 Q/K 的独立 RMSNorm（`attn_q_norm` / `attn_k_norm`）

- **目标**：对齐 Qwen3 结构，避免缺少关键归一化导致输出退化。

- **建议实现方式**：
  1. **GGUF 加载层**：从 GGUF 读取每层的 `attn_q_norm.weight` 与 `attn_k_norm.weight`（张量名以 GGUF 实际内容为准，可在 `third_party/llama.cpp` 或现有 GGUF 文档中核对）。
  2. **Kylin 层**：在 attention 内部，完成 `Qcur = rmsnorm(Qcur, q_norm_w)`、`Kcur = rmsnorm(Kcur, k_norm_w)`，然后再 RoPE。

- **注意事项**：
  - 归一化的位置应与参考实现一致：**reshape 后对每个 token/head 的向量做 RMSNorm**（等价于对最后一维做 RMSNorm）。
  - eps（`rms_norm_eps`）应来自模型配置（Qwen3 常见 `1e-6`）。

- **验收标准**：
  - 与 llama.cpp 的结构顺序一致：`attn_norm → Wq/Wk/Wv → reshape → q_norm/k_norm → RoPE → attention → Wo`。

#### P0.3 RoPE `theta`（`rope_theta`/`freq_base`）贯通：禁止硬编码

- **目标**：确保 RoPE 使用模型真实参数。

- **建议改动点**：
  - 确认 GGUF loader 或模型配置中能拿到：
    - `rope_theta`（常对应 GGUF 的 `rope.freq_base`）
    - `n_rot`（rotary dims）
  - 将 `RoPE` 初始化改为使用配置值（例如 `theta=config.ropeTheta`），并保证真正进入 `RoPE` 的 cache 生成逻辑。

- **验收标准**：
  - 对 Qwen3：`rope_theta=1000000` 时，日志/调试信息能确认 RoPE 使用的 theta 为 `1000000`，而不是 `10000`。

---

### P1（建议尽快做，提升稳定性与可诊断性）

#### P1.1 归一化 eps、head_dim、n_rot 等关键超参禁止默认值

- **目标**：避免“结构对了但数值仍偏”的隐性问题。
- **建议**：
  - `TransformerModel` 的 `rmsEps_` 不要固定 `1e-5`，应来自 `rms_norm_eps`。
  - head_dim/n_rot 统一从模型配置推导与校验，避免 mismatch。

#### P1.2 增加与 llama.cpp 的“可证伪”对齐测试

- **推荐验证项**：
  - 同一 prompt（`1+1=`）在相同采样策略（greedy）下：
    - **首个生成 token 的 top-5 logits** 与 llama.cpp 结果应“高度一致”（允许小浮点差）。
  - 当差异过大时，优先从 attention/rope/norm 的中间张量统计（min/max/NaN）定位。

#### P1.3 采样侧建议（在结构修复后再调整）

- **背景**：大词表下纯温度采样更容易出现乱码；你已在 `Sampler` 侧做了默认参数回退，这是有益的，但它不是根因。
- **建议默认策略**：
  - 对回归测试优先使用 greedy（`temperature=0` 或低于 greedy 阈值）。
  - 若使用采样：启用 `top_k`/`top_p`（例如 `top_k=40, top_p=0.9`）。

---

## 验证与回归步骤（建议按顺序执行）

### 1) 结构校验（必做）

- **GQA 形状检查**：
  - Q: `n_head`，K/V: `n_head_kv`，并满足 `n_head % n_head_kv == 0`。
- **RoPE 参数检查**：
  - theta 与模型配置一致（Qwen3 常见 `1000000`）。
- **Q/K Norm 检查**：
  - 每层都有 `attn_q_norm/attn_k_norm` 权重并参与计算。

### 2) 数值稳定性检查

- 打开 debug 日志（或临时统计）确认：
  - embedding 输出无 NaN/Inf
  - 每层 attention softmax 输入无 NaN/Inf
  - logits 不全零，且范围合理

### 3) 行为回归（端到端）

- **最小回归**：`test_hello_inference` 使用 greedy，限制 `max_new_tokens` 为 4~8。
- **期望结果**：对 `1+1=` 输出包含 `2`（或给出稳定的算术解释）。

> 备注：大模型输出可能受 prompt 模板影响（chat template / system prompt）。但在结构正确时，哪怕 prompt 不最优，也不应出现大量乱码/随机语种跳变。

---

## 风险与注意事项

- 当前 attention 实现为朴素 \(S\times S\) 计算，长上下文会非常慢且占用内存；但在修复正确性阶段可以接受（先确保“对”，再优化“快”）。
- 若修复结构后仍出现异常：
  - 优先排查量化反量化（Q4_K/Q6_K 等）是否与 ggml 一致，以及矩阵乘累加是否引入 NaN/Inf。

---

## 关联文档与参考

- `docs/design/GGUF规范.md`（关注 RoPE 字段，例如 `rope.freq_base`）
- `docs/design/GGUF格式支持详细设计.md`
- `docs/research/llama.cpp_code_analysis.md`
- `third_party/llama.cpp/src/models/qwen3.cpp`（Qwen3 参考结构：reshape + q/k norm + RoPE + attention）

---

## 附录：建议重点关注的代码文件

- **Kylin 前向**：`src/kylin/transformer_model.cpp`、`src/kylin/transformer_block.cpp`
- **Attention/RoPE**：`src/kylin/attention.cpp`、`src/kylin/rope.cpp`
- **GGUF 配置解析**：`src/model/gguf_loader_new.cpp`
- **采样器**：`src/sampler/sampler.cpp`
- **端到端回归用例**：`tests/test_hello_inference.cpp`
