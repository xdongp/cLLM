"""Export Qwen3 HuggingFace weights to a flat .bin file for C++ ModelLoader.

当前约定与 C++ 侧 `cllm::inference::ModelLoader` 一致：
- fp32: 按顺序写入 float32，布局为：
  1. embedding: [vocabSize, hiddenSize]
  2. 对每一层 (0..numLayers-1)：
     - wq:    [hiddenSize, hiddenSize]
     - wk:    [hiddenSize, hiddenSize]
     - wv:    [hiddenSize, hiddenSize]
     - wo:    [hiddenSize, hiddenSize]
     - wGate: [hiddenSize, intermediateSize]
     - wUp:   [hiddenSize, intermediateSize]
     - wDown: [intermediateSize, hiddenSize]
     - norm1: [hiddenSize]
     - norm2: [hiddenSize]
  3. finalNorm: [hiddenSize]
  4. lmHead:    [hiddenSize, vocabSize]

注意：HuggingFace Linear 权重为 [out_features, in_features]，而 C++ 代码期望
的是 [in_features, out_features]，因此导出时需要对所有 Linear 权重做转置。

目前 C++ ModelLoader 只支持 fp32 文件：
- fp16 / int8 导出也支持，但要使用这些文件，需要后续扩展 C++ ModelLoader
  和 InferenceEngine 的解析逻辑（包括量化缩放参数），本脚本暂只负责生成。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _get_tensor(sd: Dict[str, torch.Tensor], *names: str) -> torch.Tensor:
    """从 state_dict 中按多个候选名字依次查找张量。

    便于兼容不同版本/命名差异；如果都找不到则抛出 KeyError。
    """

    for name in names:
        if name in sd:
            return sd[name]
    raise KeyError(f"None of tensor names found: {names}")


def export_qwen_flat_bin(
    model_dir: str,
    output_path: str,
    dtype: str = "fp32",
) -> None:
    model_dir = str(model_dir)
    output_path = str(output_path)

    if dtype not in {"fp32", "fp16", "int8"}:
        raise ValueError("dtype must be one of: fp32, fp16, int8")

    print(f"[export_qwen_bin] Loading config from: {model_dir}")
    config = AutoConfig.from_pretrained(model_dir)

    hidden_size: int = getattr(config, "hidden_size")
    vocab_size: int = getattr(config, "vocab_size")
    num_layers: int = getattr(config, "num_hidden_layers")
    intermediate_size: int = getattr(config, "intermediate_size")
    num_attention_heads: int = getattr(config, "num_attention_heads")
    num_key_value_heads: int = getattr(config, "num_key_value_heads", num_attention_heads)
    head_dim: int = hidden_size // num_attention_heads

    # 推导真实的 Q/K/V/O 投影维度
    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim

    print(
        f"[export_qwen_bin] Model config: hidden={hidden_size}, "
        f"vocab={vocab_size}, layers={num_layers}, inter={intermediate_size}\n"
        f"  Q heads={num_attention_heads}, KV heads={num_key_value_heads}, head_dim={head_dim}\n"
        f"  Q proj dim={q_dim}, KV proj dim={kv_dim}"
    )

    print(f"[export_qwen_bin] Loading model weights from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
    )
    model.eval()

    state_dict = model.state_dict()

    # 统一转换为 float32 cpu numpy
    for k, v in state_dict.items():
        state_dict[k] = v.to(torch.float32).cpu()

    # 1. embedding
    emb = _get_tensor(state_dict, "model.embed_tokens.weight", "transformer.embed_tokens.weight")
    assert emb.shape == (vocab_size, hidden_size), f"Unexpected embedding shape: {emb.shape}"
    emb_np = emb.cpu().numpy()  # [vocab, hidden]

    # 2. per-layer weights
    wq_list: List[np.ndarray] = []
    wk_list: List[np.ndarray] = []
    wv_list: List[np.ndarray] = []
    wo_list: List[np.ndarray] = []
    wgate_list: List[np.ndarray] = []
    wup_list: List[np.ndarray] = []
    wdown_list: List[np.ndarray] = []
    norm1_list: List[np.ndarray] = []
    norm2_list: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        prefix_candidates = [
            f"model.layers.{layer_idx}.",
            f"transformer.layers.{layer_idx}.",
        ]

        def name(*parts: str) -> List[str]:
            return [p + ".".join(parts) for p in prefix_candidates]

        # Attention Q/K/V/O
        q_w = _get_tensor(state_dict, *name("self_attn.q_proj.weight"))
        k_w = _get_tensor(state_dict, *name("self_attn.k_proj.weight"))
        v_w = _get_tensor(state_dict, *name("self_attn.v_proj.weight"))
        o_w = _get_tensor(state_dict, *name("self_attn.o_proj.weight"))

        # 自动检测实际的投影维度（Qwen3 使用 2x 扩展）
        actual_q_out = q_w.shape[0]
        actual_k_out = k_w.shape[0]
        actual_v_out = v_w.shape[0]
        actual_o_in = o_w.shape[1]

        # 转成 [in, out] 布局
        wq_np = q_w.cpu().numpy().T  # [hidden, actual_q_out]
        wk_np = k_w.cpu().numpy().T  # [hidden, actual_k_out]
        wv_np = v_w.cpu().numpy().T  # [hidden, actual_v_out]
        wo_np = o_w.cpu().numpy().T  # [actual_o_in, hidden]

        if layer_idx == 0:
            print(f"[export_qwen_bin] Layer 0 actual proj dims: Q={actual_q_out}, K={actual_k_out}, V={actual_v_out}, O_in={actual_o_in}")

        if wq_np.shape[0] != hidden_size:
            print(f"[ERROR] Layer {layer_idx} wq_np input dim mismatch: got {wq_np.shape[0]}, expected {hidden_size}")
            raise AssertionError(f"wq_np input dim mismatch at layer {layer_idx}")
        if wk_np.shape[0] != hidden_size:
            print(f"[ERROR] Layer {layer_idx} wk_np input dim mismatch: got {wk_np.shape[0]}, expected {hidden_size}")
            raise AssertionError(f"wk_np input dim mismatch at layer {layer_idx}")
        if wv_np.shape[0] != hidden_size:
            print(f"[ERROR] Layer {layer_idx} wv_np input dim mismatch: got {wv_np.shape[0]}, expected {hidden_size}")
            raise AssertionError(f"wv_np input dim mismatch at layer {layer_idx}")
        if wo_np.shape[1] != hidden_size:
            print(f"[ERROR] Layer {layer_idx} wo_np output dim mismatch: got {wo_np.shape[1]}, expected {hidden_size}")
            raise AssertionError(f"wo_np output dim mismatch at layer {layer_idx}")

        # MLP: gate / up / down
        gate_w = _get_tensor(state_dict, *name("mlp.gate_proj.weight"))
        up_w = _get_tensor(state_dict, *name("mlp.up_proj.weight"))
        down_w = _get_tensor(state_dict, *name("mlp.down_proj.weight"))

        wgate_np = gate_w.cpu().numpy().T  # [hidden, inter]
        wup_np = up_w.cpu().numpy().T      # [hidden, inter]
        wdown_np = down_w.cpu().numpy().T  # [inter, hidden]

        if wgate_np.shape != (hidden_size, intermediate_size):
            print(f"[ERROR] Layer {layer_idx} wgate_np shape mismatch: got {wgate_np.shape}, expected ({hidden_size}, {intermediate_size})")
            raise AssertionError(f"wgate_np shape mismatch at layer {layer_idx}")
        if wup_np.shape != (hidden_size, intermediate_size):
            print(f"[ERROR] Layer {layer_idx} wup_np shape mismatch: got {wup_np.shape}, expected ({hidden_size}, {intermediate_size})")
            raise AssertionError(f"wup_np shape mismatch at layer {layer_idx}")
        if wdown_np.shape != (intermediate_size, hidden_size):
            print(f"[ERROR] Layer {layer_idx} wdown_np shape mismatch: got {wdown_np.shape}, expected ({intermediate_size}, {hidden_size})")
            raise AssertionError(f"wdown_np shape mismatch at layer {layer_idx}")

        # LayerNorm / RMSNorm 权重
        norm1_w = _get_tensor(state_dict, *name("input_layernorm.weight"))
        norm2_w = _get_tensor(state_dict, *name("post_attention_layernorm.weight"))

        norm1_np = norm1_w.cpu().numpy().reshape(-1)
        norm2_np = norm2_w.cpu().numpy().reshape(-1)

        assert norm1_np.shape[0] == hidden_size
        assert norm2_np.shape[0] == hidden_size

        wq_list.append(wq_np)
        wk_list.append(wk_np)
        wv_list.append(wv_np)
        wo_list.append(wo_np)
        wgate_list.append(wgate_np)
        wup_list.append(wup_np)
        wdown_list.append(wdown_np)
        norm1_list.append(norm1_np)
        norm2_list.append(norm2_np)

    # 3. final norm
    final_norm = _get_tensor(
        state_dict,
        "model.norm.weight",
        "transformer.norm.weight",
    )
    final_norm_np = final_norm.cpu().numpy().reshape(-1)
    assert final_norm_np.shape[0] == hidden_size

    # 4. lm_head
    lm_head = _get_tensor(state_dict, "lm_head.weight")
    # lm_head: [vocab, hidden] -> 转为 [hidden, vocab]
    lm_head_np = lm_head.cpu().numpy().T
    assert lm_head_np.shape == (hidden_size, vocab_size)

    # 按 C++ 约定顺序拼接为一维数组
    flat_tensors: List[np.ndarray] = []

    flat_tensors.append(emb_np.reshape(-1))

    for i in range(num_layers):
        flat_tensors.append(wq_list[i].reshape(-1))
        flat_tensors.append(wk_list[i].reshape(-1))
        flat_tensors.append(wv_list[i].reshape(-1))
        flat_tensors.append(wo_list[i].reshape(-1))

        flat_tensors.append(wgate_list[i].reshape(-1))
        flat_tensors.append(wup_list[i].reshape(-1))
        flat_tensors.append(wdown_list[i].reshape(-1))

        flat_tensors.append(norm1_list[i])
        flat_tensors.append(norm2_list[i])

    flat_tensors.append(final_norm_np)
    flat_tensors.append(lm_head_np.reshape(-1))

    flat = np.concatenate(flat_tensors, axis=0)
    print(f"[export_qwen_bin] Total floats before dtype conversion: {flat.size}")

    meta: Dict[str, Dict[str, float]] = {}

    if dtype == "fp32":
        flat_out = flat.astype(np.float32)
    elif dtype == "fp16":
        flat_out = flat.astype(np.float16)
    else:  # int8
        # 简单对称量化: x -> int8, 按单个全局 scale
        max_abs = float(np.max(np.abs(flat)))
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
        flat_out = q
        meta["int8"] = {"scale": scale}
        print(f"[export_qwen_bin] int8 quantization scale={scale:.6e}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[export_qwen_bin] Writing binary to: {out_path} (dtype={dtype})")
    flat_out.tofile(str(out_path))

    # 生成元数据文件，记录实际投影维度
    meta_to_write: Dict[str, Any] = {
        "model_config": {
            "hidden_size": int(hidden_size),
            "vocab_size": int(vocab_size),
            "num_layers": int(num_layers),
            "intermediate_size": int(intermediate_size),
            "num_attention_heads": int(num_attention_heads),
            "num_key_value_heads": int(num_key_value_heads),
            "actual_q_proj_dim": int(actual_q_out),
            "actual_kv_proj_dim": int(actual_k_out),
        }
    }
    
    if dtype == "int8" and meta:
        meta_to_write.update(meta)

    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    print(f"[export_qwen_bin] Writing meta to: {meta_path}")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta_to_write, f, indent=2)

    print("[export_qwen_bin] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Qwen3 HF weights to flat .bin for C++ ModelLoader",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=False,
        default=str(
            Path(__file__).resolve().parents[1] / "model" / "Qwen" / "Qwen3-0.6B"
        ),
        help="HF 格式的 Qwen3 模型目录（包含 config.json / model.safetensors 等）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=str(
            Path(__file__).resolve().parents[1]
            / "model"
            / "Qwen"
            / "qwen3_0.6b_cllm_fp32.bin"
        ),
        help="导出的 .bin 文件路径",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "int8"],
        default="fp32",
        help="导出数据精度类型 (当前 C++ 仅兼容 fp32)",
    )

    args = parser.parse_args()
    export_qwen_flat_bin(args.model_dir, args.output, args.dtype)


if __name__ == "__main__":
    main()
