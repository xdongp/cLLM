# Kylin对比llama.cpp实现

## 背景
针对 `1+1=` 推理输出异常的问题，本文对 `Kylin` 推理实现与 `llama.cpp` 的 `qwen3` 实现进行对比，梳理一致点、差异点与风险，并给出修复优先级建议与验证清单。

## 对比范围
- `src/kylin/attention.cpp`
- `src/kylin/transformer_block.cpp`
- `src/inference/kylin_backend.cpp`
- `third_party/llama.cpp/src/models/qwen3.cpp`

## 已对齐点（结构层面）
1. **Block 结构一致**：均为 `RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual`。
2. **Q/K Norm 位置一致**：均在 RoPE 之前应用（若权重存在）。
3. **GQA 框架存在**：均支持 `num_heads != num_kv_heads`。

## 关键差异与风险

### 1) 头维度一致性（head_dim）约束
- **llama.cpp**：强制断言 `Q/K/V head_dim` 与 `n_rot` 完全一致，否则直接失败。
- **Kylin**：出现不一致时进入“容错路径”（截断 RoPE、重复/补零 V、使用 `minHeadDim` 计算 attention）。
- **风险**：模型配置轻微偏差时仍可“跑通”，但输出严重偏离。

### 2) `numQHeads` 推断逻辑与扩展 head_dim 冲突
- **llama.cpp**：Qwen3 以 `n_embd_head` 为真实 head_dim，`n_head` 保持模型配置。
- **Kylin**：若 `qDim % standardHeadDim == 0`，会推断并覆盖 `numQHeads`，对 Qwen3 这类“扩展 head_dim”模型可能误判为“更多 heads”。
- **风险**：头数错误会导致 reshape、RoPE、QK Norm、attention 全链路错位。

### 3) RoPE 参数对齐不足
- **llama.cpp**：使用 `ggml_rope_ext`，包含 `freq_base`、`freq_scale`、`ext_factor`、`n_ctx_orig` 等。
- **Kylin**：仅使用 `ropeTheta`，且 `maxSeqLen` 固定为 `2048`。
- **风险**：长上下文/扩展 RoPE 模型结果偏差明显。

### 4) Q/K Norm 可能被静默跳过
- **llama.cpp**：默认启用 Q/K Norm（权重存在时）。
- **Kylin**：仅当权重长度 == `head_dim` 才应用，且 `head_dim` 由推断结果决定。
- **风险**：若 `head_dim` 推断错误，会直接跳过 Q/K Norm。

## 根因优先级判断（高 → 低）
1. **`numQHeads` 推断错误**（高概率导致全链路错位）。
2. **attention 容错路径导致偏离**（应对齐 llama.cpp 的强一致性）。
3. **RoPE 参数不完整**（对 Qwen3/长上下文影响显著）。
4. **Q/K Norm 参数与 epsilon 固定**（可能放大数值偏差）。

## 修复建议（按优先级）
1. **禁用“标准公式推断 heads”覆盖模型配置**：优先相信 GGUF 配置中的 `num_heads`，将 `qHeadDim = qDim / num_heads` 视为“扩展 head_dim”。
2. **移除/收紧 attention 的容错逻辑**：当 `qHeadDim != kvHeadDim` 或 `n_rot` 不一致时直接报错，避免“看似能跑”的错误输出。
3. **补齐 RoPE 扩展参数**：至少引入 `n_ctx_orig`、`freq_scale`、`rope_type`、`ext_factor` 等关键参数，避免固定 `maxSeqLen=2048`。
4. **读取并应用真实 `rms_norm_eps`**：不要在 Q/K Norm 固定为 `1e-6`。

## 验证清单
- [ ] 日志中 `numQHeads/numKVHeads/qHeadDim/kvHeadDim` 与 GGUF/llama.cpp 一致。
- [ ] 不再出现 `Head dimension mismatch` 日志。
- [ ] RoPE 初始化参数与 GGUF 配置一致（含 `n_ctx_orig`）。
- [ ] Q/K Norm 不再被跳过（权重长度与 head_dim 一致）。
- [ ] 在 `temperature=0` 下输出稳定，`1+1=` 可复现为合理结果。

## 备注
本文聚焦 Kylin 与 llama.cpp 的实现差异，未展开 tokenizer/vocab 对齐问题；若输出仍异常，需另行排查 tokenizer 与 vocab size 对齐情况。