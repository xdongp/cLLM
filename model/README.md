# xLLM Model Directory

This directory is intended for storing language models used with the xLLM inference engine.

## Directory Structure

```
model/
├── qwen3-0.6b-int8/     # Qwen3 0.6B INT8 model (to be downloaded)
├── download_and_test.py  # Script to download and test models
└── test_model.py         # Simple model test script
```

## How to Use

1. **Download the Qwen3 0.6B INT8 Model**:
   - Visit: https://huggingface.co/Qwen/Qwen3-0.6B
   - Download the INT8 quantized version
   - Extract the files to the `qwen3-0.6b-int8` directory

2. **Test the Model**:
   ```bash
   python test_model.py
   ```

## Model Requirements

The Qwen3 0.6B INT8 model should include at least the following files:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`

## Testing with xLLM

After downloading the model, you can test it with xLLM:

```bash
# From the xllm directory
xllm serve --model-path ./model/qwen3-0.6b-int8 --port 8080 --quantization int8
```

Then use the API to interact with the model.