# Tokenizer乱码修复记录

## 问题概述
- 现象：中文 prompt 生成结果出现乱码；并发场景下偶发 `total tokens cannot be zero`。
- 触发条件：使用 GGUF 模型，中文输入时 `GGUFTokenizer::encode` 大量 token 未命中词表。

## 根因定位
1. **GGUFTokenizer 编码逻辑过于简化**
   - 未进行 byte-level BPE 预编码，导致中文字符难以匹配词表。
   - `encode` 输出空 token 列表，进而触发 `LlamaCppBackend::forwardBatch: total tokens cannot be zero`。
2. **解码缺少 byte-level 还原**
   - 直接拼接 token 字符串，未将 byte-encoded 文本还原为 UTF‑8。

## 关键修复
### 1) 增加 GPT‑2/llama 风格 byte 编码表
- 使用标准 byte encoder/decoder（256 bytes 映射到 Unicode）。
- 保证与 GGUF 词表的 byte-level 预编码一致。

### 2) 编码流程调整
- 在 `GGUFTokenizer::encode` 中：
  - 先 `byteEncode` 后再走 BPE 合并。
  - 对未命中 token 增加 byte fallback。

### 3) 解码流程调整
- 在 `GGUFTokenizer::decode` 中：
  - 拼接 token 后执行 `byteDecode` 还原 UTF‑8。

### 4) 编码兜底
- 当编码结果为空时，兜底注入可用 token（UNK/BOS/EOS/0）避免 `total tokens=0`。

## 影响文件
- `src/tokenizer/gguf_tokenizer.cpp`

## 验证结果
- 中文 prompt 可正常生成。
- 不再出现 `Token not found` 导致的空编码。
- 不再报 `total tokens cannot be zero`。

## 经验总结
- GGUF 模型必须使用 **byte-level BPE** 编码/解码流程，否则中文与非 ASCII 字符容易失真。
- 编码为空时需要兜底，避免后端推理直接失败。
- 解码必须进行 byte-level 还原，否则输出呈现乱码。
