#!/usr/bin/env python3
"""
Convert Qwen3-0.6B HuggingFace model to GGUF format.

This script wraps llama.cpp's convert_hf_to_gguf.py to convert
HuggingFace format models to GGUF format.

Usage:
    python convert_qwen_to_gguf.py [--model-dir MODEL_DIR] [--output OUTPUT] [--outtype OUTTYPE]

Examples:
    # Convert to F32 GGUF format
    python convert_qwen_to_gguf.py --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-f32.gguf --outtype f32

    # Convert to F16 GGUF format
    python convert_qwen_to_gguf.py --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-f16.gguf --outtype f16

    # Convert to Q8_0 quantized format
    python convert_qwen_to_gguf.py --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-q8_0.gguf --outtype q8_0

    # Convert to Q4_K_M quantized format (4-bit, medium quality)
    python convert_qwen_to_gguf.py --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-q4_k_m.gguf --outtype q4_k_m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_llama_cpp_convert_script() -> Path:
    """Find the llama.cpp convert_hf_to_gguf.py script."""
    # Try multiple possible locations
    script_name = "convert_hf_to_gguf.py"
    
    # 1. Check in third_party/llama.cpp
    project_root = Path(__file__).resolve().parent.parent
    llama_cpp_script = project_root / "third_party" / "llama.cpp" / script_name
    if llama_cpp_script.exists():
        return llama_cpp_script
    
    # 2. Check if llama.cpp is installed as a package
    try:
        import llama_cpp
        # If installed, the script might be in the package
        llama_cpp_path = Path(llama_cpp.__file__).parent.parent
        script = llama_cpp_path / script_name
        if script.exists():
            return script
    except ImportError:
        pass
    
    # 3. Check in current directory or PATH
    script = Path(script_name)
    if script.exists():
        return script.resolve()
    
    # 4. Try to find in common locations
    common_paths = [
        Path.home() / "llama.cpp" / script_name,
        Path("/usr/local/share/llama.cpp") / script_name,
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(
        f"Could not find {script_name}. Please ensure llama.cpp is available.\n"
        f"Options:\n"
        f"  1. Clone llama.cpp to third_party/llama.cpp/\n"
        f"  2. Install llama.cpp package\n"
        f"  3. Place {script_name} in PATH"
    )


def check_dependencies() -> None:
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Please install them with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def convert_to_gguf(
    model_dir: str,
    output_path: str,
    outtype: str = "f32",
    verbose: bool = False,
    vocab_only: bool = False,
) -> bool:
    """
    Convert HuggingFace model to GGUF format.
    
    Args:
        model_dir: Path to HuggingFace model directory
        output_path: Output GGUF file path
        outtype: Output type (f32, f16, q8_0, q4_k_m, etc.)
        verbose: Enable verbose output
        vocab_only: Only export vocabulary
    
    Returns:
        True if conversion successful, False otherwise
    """
    model_dir = Path(model_dir).resolve()
    output_path = Path(output_path).resolve()
    
    # Validate inputs
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        return False
    
    if not model_dir.is_dir():
        print(f"Error: Model path is not a directory: {model_dir}")
        return False
    
    # Check for required files
    required_files = ["config.json"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    if missing_files:
        print(f"Error: Missing required files in model directory: {', '.join(missing_files)}")
        return False
    
    # Check for model weight files
    weight_files = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]
    has_weights = any((model_dir / f).exists() for f in weight_files)
    if not has_weights and not vocab_only:
        print(f"Warning: No model weight files found in {model_dir}")
        print(f"  Looking for: {', '.join(weight_files)}")
        print(f"  If you only want to export vocabulary, use --vocab-only flag")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # Find conversion script
    try:
        convert_script = find_llama_cpp_convert_script()
        print(f"Using conversion script: {convert_script}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    
    # Prepare output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Note: Some quantization types (like q4_k_m) may not be directly supported
    # by convert_hf_to_gguf.py. If conversion fails, you may need to:
    # 1. First convert to f16: --outtype f16
    # 2. Then use llama.cpp quantization tools to convert f16 to q4_k_m
    
    # Build command
    cmd = [
        sys.executable,
        str(convert_script),
        str(model_dir),
        "--outtype", outtype,
        "--outfile", str(output_path),
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    if vocab_only:
        cmd.append("--vocab-only")
    
    # Print command for debugging
    print(f"\nConverting model to GGUF format...")
    print(f"  Model directory: {model_dir}")
    print(f"  Output file: {output_path}")
    print(f"  Output type: {outtype}")
    if outtype == "q4_k_m":
        print(f"  Note: If conversion fails, you may need to convert to f16 first,")
        print(f"        then use llama.cpp quantization tools for Q4_K_M")
    print(f"  Command: {' '.join(cmd)}\n")
    
    # Run conversion
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
        )
        
        if result.returncode == 0:
            print(f"\n✅ Successfully converted model to: {output_path}")
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"   Output file size: {size_mb:.2f} MB")
            return True
        else:
            print(f"\n❌ Conversion failed with return code: {result.returncode}")
            if outtype == "q4_k_m":
                print(f"\n💡 Tip: Q4_K_M may not be directly supported by convert_hf_to_gguf.py")
                print(f"   Try this two-step approach:")
                print(f"   1. Convert to F16: python {sys.argv[0]} --outtype f16 --output {output_path.parent / output_path.stem}_f16.gguf")
                print(f"   2. Use llama.cpp quantization tools to convert F16 to Q4_K_M")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion failed: {e}")
        if outtype == "q4_k_m":
            print(f"\n💡 Tip: Q4_K_M may not be directly supported by convert_hf_to_gguf.py")
            print(f"   Try converting to F16 first, then use quantization tools")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversion interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-0.6B HuggingFace model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to F32 format (full precision)
  %(prog)s --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-f32.gguf --outtype f32

  # Convert to F16 format (half precision, smaller file)
  %(prog)s --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-f16.gguf --outtype f16

  # Convert to Q8_0 quantized format (8-bit quantization)
  %(prog)s --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-q8_0.gguf --outtype q8_0

  # Convert to Q4_K_M quantized format (4-bit, medium quality, smaller file)
  %(prog)s --model-dir model/Qwen/Qwen3-0.6B --output model/Qwen/qwen3-0.6b-q4_k_m.gguf --outtype q4_k_m

Supported output types:
  - f32: Full precision (FP32)
  - f16: Half precision (FP16)
  - bf16: Brain float 16
  - q8_0: 8-bit quantization
  - q4_k_m: 4-bit K-quant (medium quality, recommended for 4-bit)
  - tq1_0: Tiny quant 1.0
  - tq2_0: Tiny quant 2.0
  - auto: Auto-detect best type
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "Qwen" / "Qwen3-0.6B"),
        help="Path to HuggingFace model directory (default: model/Qwen/Qwen3-0.6B)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GGUF file path (default: model/Qwen/qwen3-0.6b-{outtype}.gguf)",
    )
    
    parser.add_argument(
        "--outtype",
        type=str,
        choices=["f32", "f16", "bf16", "q8_0", "q4_k_m", "tq1_0", "tq2_0", "auto"],
        default="f32",
        help="Output quantization type (default: f32)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only export vocabulary (for testing)",
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        print("Checking dependencies...")
        check_dependencies()
        print("✅ All dependencies are available")
        return 0
    
    # Check dependencies
    check_dependencies()
    
    # Determine output path
    if args.output is None:
        model_name = Path(args.model_dir).name.lower().replace("-", "_").replace("_", "-")
        # Default output: model/Qwen/qwen3-0.6b-{outtype}.gguf
        output_path = Path(__file__).parent / "Qwen" / f"{model_name}-{args.outtype}.gguf"
    else:
        output_path = Path(args.output)
    
    # Ensure output has .gguf extension
    if output_path.suffix != ".gguf":
        output_path = output_path.with_suffix(".gguf")
    
    # Convert model
    success = convert_to_gguf(
        model_dir=args.model_dir,
        output_path=str(output_path),
        outtype=args.outtype,
        verbose=args.verbose,
        vocab_only=args.vocab_only,
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
