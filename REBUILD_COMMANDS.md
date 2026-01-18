# 重新编译命令

## ⚠️ 重要：需要设置日志级别为 debug

Makefile 中的 `start-gguf-q4k` 默认使用 `--log-level info`，这会导致 DEBUG 级别的日志不会输出。
**必须使用 `--log-level debug` 才能看到 `getLogitsForRequest` 的详细日志！**

## 方法 1: 完全清理并重新编译（推荐，确保所有文件重新编译）

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
make clean
make build
```

## 方法 2: 强制重新编译（touch 头文件强制依赖更新）

由于 `include/cllm/batch/output.h` 被以下文件包含：
- `src/scheduler/batch_processor.cpp`
- `src/model/executor.cpp`
- `src/batch/manager.cpp`
- `src/model/batch_processor.cpp`

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
# 强制更新头文件时间戳，触发重新编译
touch include/cllm/batch/output.h
cd build
make -j$(sysctl -n hw.ncpu)
```

## 方法 3: 直接重新编译 cllm_core 和 cllm_server（最快）

```bash
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
touch include/cllm/batch/output.h
cd build
make cllm_core -j$(sysctl -n hw.ncpu)
make cllm_server -j$(sysctl -n hw.ncpu)
```

## 启动服务器时使用 debug 日志级别

**重要**：默认的 `start-gguf-q4k` 使用 `--log-level info`，需要手动指定 debug：

```bash
# 停止旧服务器（如果正在运行）
make stop-gguf-q4k 2>/dev/null || pkill -f "cllm_server.*$(GGUF_Q4K_PORT)"

# 手动启动服务器，使用 debug 日志级别
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
./build/bin/cllm_server \
    --model-path /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf \
    --port 18082 \
    --host 0.0.0.0 \
    --log-level debug \
    --log-file logs/cllm_server_q4k_18082.log \
    --max-batch-size 8 \
    --max-context-length 2048
```

或者修改 Makefile 中的 `start-gguf-q4k` 目标，将 `--log-level info` 改为 `--log-level debug`

## 检查编译是否成功

编译完成后，检查二进制文件的修改时间：
```bash
ls -lh build/bin/cllm_server
stat -f "%Sm" build/bin/cllm_server
```

应该看到时间戳比 `include/cllm/batch/output.h` 新。
