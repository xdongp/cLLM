#!/usr/bin/env python3
"""
将 HuggingFace Qwen3 模型导出为 TorchScript (.pt) 格式

使用方法：
    python export_qwen_torchscript.py --model-path ./Qwen/Qwen3-0.6B --output-path ./Qwen/qwen3_0.6b.pt
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


class Qwen3Wrapper(torch.nn.Module):
    """
    Qwen3 模型包装器，用于 TorchScript 导出
    
    将 HuggingFace 模型包装为简单的 forward 接口
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入 token IDs
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] 输出 logits
        """
        outputs = self.model(input_ids)
        return outputs.logits


def export_qwen_torchscript(
    model_path: str,
    output_path: str,
    use_fp16: bool = False,
    use_int8: bool = False
):
    """
    导出 Qwen3 模型为 TorchScript 格式
    
    Args:
        model_path: HuggingFace 模型路径
        output_path: 输出 .pt 文件路径
        use_fp16: 是否使用 FP16 量化
        use_int8: 是否使用 INT8 动态量化
    """
    print(f"[export_qwen_torchscript] Loading model from: {model_path}")
    
    # 加载模型和 tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if not use_fp16 else torch.float16,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"[export_qwen_torchscript] Model config:")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_layers: {model.config.num_hidden_layers}")
    print(f"  num_attention_heads: {model.config.num_attention_heads}")
    print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
    
    # 设置为评估模式
    model.eval()
    
    # 应用量化
    if use_int8:
        print("[export_qwen_torchscript] Applying INT8 dynamic quantization...")
        
        # 检查量化引擎支持
        supported_engines = torch.backends.quantized.supported_engines
        print(f"[export_qwen_torchscript] Available quantization engines: {supported_engines}")
        
        # 尝试设置量化引擎（优先使用 qnnpack，在 macOS ARM 上通常可用）
        try:
            if 'qnnpack' in supported_engines:
                torch.backends.quantized.engine = 'qnnpack'
                print("[export_qwen_torchscript] Using qnnpack quantization engine")
            elif 'fbgemm' in supported_engines:
                torch.backends.quantized.engine = 'fbgemm'
                print("[export_qwen_torchscript] Using fbgemm quantization engine")
            else:
                print("[export_qwen_torchscript] Warning: No quantization engine available!")
                print("[export_qwen_torchscript] Supported engines: " + str(supported_engines))
                raise RuntimeError(
                    "No quantization engine available. This is common on macOS/Apple Silicon. "
                    "Please use --fp16 instead, or install a PyTorch build with quantization support."
                )
        except Exception as e:
            print(f"[export_qwen_torchscript] Error setting quantization engine: {e}")
            raise
        
        # 确保模型在 CPU 上（量化需要 CPU）
        model = model.cpu()
        
        try:
            # 尝试使用新的 API (torch.ao.quantization) 如果可用
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except (AttributeError, ImportError):
                # 回退到旧 API
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "quantized::linear_prepack" in str(e):
                print("\n[export_qwen_torchscript] ❌ INT8 quantization failed!")
                print("[export_qwen_torchscript] Reason: Quantization engine not available")
                print("[export_qwen_torchscript] This is common on macOS, especially Apple Silicon.")
                print("\n[export_qwen_torchscript] 💡 Solution:")
                print("  1. Use FP16 instead: --fp16")
                print("  2. Or skip quantization: remove --int8 flag")
                print("\n[export_qwen_torchscript] To use INT8, you need:")
                print("  - PyTorch built with quantization support")
                print("  - Or quantize on Linux/x86 machine, then deploy to macOS")
                raise RuntimeError(
                    f"INT8 quantization failed: {e}\n"
                    "Tip: Use --fp16 instead, or remove --int8 flag to export FP32 model."
                ) from e
            else:
                raise
    elif use_fp16:
        print("[export_qwen_torchscript] Converting to FP16...")
        model = model.half()
    
    # 包装模型
    wrapped_model = Qwen3Wrapper(model)
    wrapped_model.eval()
    
    # 创建示例输入（用于 tracing）
    # NOTE: seq_len 决定了导出模型能处理的最大序列长度！
    # 之前用 (1, 8) 导致模型只能处理 8 token，现在改为 128
    example_input = torch.randint(0, model.config.vocab_size, (1, 128), dtype=torch.long)
    
    print(f"[export_qwen_torchscript] Tracing model with example input shape: {example_input.shape}")
    
    # 使用 torch.jit.trace 导出
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapped_model, example_input)
    
    # 保存 TorchScript 模型
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[export_qwen_torchscript] Saving TorchScript model to: {output_path}")
    torch.jit.save(traced_model, str(output_path))
    
    # 验证导出
    print("[export_qwen_torchscript] Verifying exported model...")
    loaded_model = torch.jit.load(str(output_path))
    test_output = loaded_model(example_input)
    print(f"  Test output shape: {test_output.shape}")
    print(f"  Expected shape: [1, 8, {model.config.vocab_size}]")
    
    # 保存模型配置
    config_path = output_path.with_suffix(".json")
    config_data = {
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "intermediate_size": model.config.intermediate_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "dtype": "int8" if use_int8 else ("fp16" if use_fp16 else "fp32")
    }
    
    import json
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    
    print(f"[export_qwen_torchscript] Model config saved to: {config_path}")
    print("[export_qwen_torchscript] Export completed successfully!")
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[export_qwen_torchscript] Model file size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3 model to TorchScript format")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./Qwen/Qwen3-0.6B",
        help="Path to HuggingFace Qwen3 model"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./Qwen/qwen3_0.6b_torchscript.pt",
        help="Output TorchScript model path (.pt file)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in FP16 format"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Apply INT8 dynamic quantization"
    )
    
    args = parser.parse_args()
    
    export_qwen_torchscript(
        model_path=args.model_path,
        output_path=args.output_path,
        use_fp16=args.fp16,
        use_int8=args.int8
    )


if __name__ == "__main__":
    main()
