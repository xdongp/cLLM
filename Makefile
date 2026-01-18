.PHONY: all build clean rebuild test run start stop help install-deps setup-env check-model test-gguf-e2e test-curl start-gguf-q4k start-gguf-q4k-bg test-gguf-q4k test-curl-q4k

# 激活虚拟环境
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# 如果存在虚拟环境，则激活它
ifneq (,$(wildcard ../../../.venv/bin/activate))
    VENV_ACTIVATE := source ../../../.venv/bin/activate &&
else
    VENV_ACTIVATE :=
endif

BUILD_DIR = build
BUILD_TYPE = Release
NUM_JOBS = $(shell sysctl -n hw.ncpu)
MODEL_PATH ?= /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3_0.6b_torchscript_fp32.pt
GGUF_MODEL_PATH ?= /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf
GGUF_Q4K_MODEL_PATH ?= /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf
PORT ?= 18080
GGUF_PORT ?= 18081
GGUF_Q4K_PORT ?= 18082
QUANTIZATION ?= fp16
MAX_BATCH_SIZE ?= 8
MAX_CONTEXT_LENGTH ?= 2048
USE_LIBTORCH ?= true
USE_GGUF ?= false

all: build

build:
	@echo "Building cLLM..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) make -j$(NUM_JOBS)
	@echo "Build completed successfully!"
	@echo "Note: Examples (http_server_example, basic_usage) are disabled."
	@echo "Use 'make start' to run the cLLM server."

build-debug:
	@echo "Building cLLM in Debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) make -j$(NUM_JOBS)
	@echo "Debug build completed successfully!"

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean completed!"

rebuild: clean build

test: build
	@echo "Running unit tests..."
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) ctest --output-on-failure -L unit_test
	@echo "Unit tests completed!"

integration-test: build
	@echo "Running integration tests..."
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) ctest --output-on-failure -L integration_test
	@echo "Integration tests completed!"

test-tokenizer: build
	@echo "Running tokenizer tests..."
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) ./tests/test_tokenizer
	@echo "Tokenizer tests completed!"

test-all: test integration-test

check-model:
	@echo "Checking model path..."
	@if [ ! -d "$(MODEL_PATH)" ]; then \
		echo "Error: Model not found at $(MODEL_PATH)"; \
		echo "Please set MODEL_PATH environment variable or use: make start MODEL_PATH=/path/to/model"; \
		exit 1; \
	fi
	@echo "Model found at: $(MODEL_PATH)"

start: build
	@echo "Starting cLLM server..."
	@echo "Configuration:";
	@if [ "$(USE_GGUF)" = "true" ]; then \
		MODEL_PATH_TO_USE=$(GGUF_MODEL_PATH); \
		PORT_TO_USE=$(GGUF_PORT); \
		echo "  Model Path: $$MODEL_PATH_TO_USE"; \
		echo "  Model Type: GGUF"; \
		echo "  Port: $$PORT_TO_USE"; \
	else \
		MODEL_PATH_TO_USE=$(MODEL_PATH); \
		PORT_TO_USE=$(PORT); \
		echo "  Model Path: $$MODEL_PATH_TO_USE"; \
		echo "  Model Type: $(if $(filter true,$(USE_LIBTORCH)),TorchScript,Kylin)"; \
		echo "  Port: $$PORT_TO_USE"; \
	fi;
	@echo "  Quantization: $(QUANTIZATION)";
	@echo "  Max Batch Size: $(MAX_BATCH_SIZE)";
	@echo "  Max Context Length: $(MAX_CONTEXT_LENGTH)";
	@echo "  Backend: $(if $(filter true,$(USE_LIBTORCH)),LibTorch,Kylin)";
	@echo "";
	@if [ -f $(BUILD_DIR)/bin/cllm_server ]; then \
		if [ "$(USE_GGUF)" = "true" ]; then \
			$(BUILD_DIR)/bin/cllm_server \
				--model-path $(GGUF_MODEL_PATH) \
				--port $(GGUF_PORT) \
				--quantization $(QUANTIZATION) \
				--max-batch-size $(MAX_BATCH_SIZE) \
				--max-context-length $(MAX_CONTEXT_LENGTH) \
				$(if $(filter true,$(USE_LIBTORCH)),--use-libtorch,); \
		else \
			$(BUILD_DIR)/bin/cllm_server \
				--model-path $(MODEL_PATH) \
				--port $(PORT) \
				--quantization $(QUANTIZATION) \
				--max-batch-size $(MAX_BATCH_SIZE) \
				--max-context-length $(MAX_CONTEXT_LENGTH) \
				$(if $(filter true,$(USE_LIBTORCH)),--use-libtorch,); \
		fi; \
	else \
		echo "Error: cllm_server executable not found.";
		echo "Note: The main server executable is not yet implemented.";
		echo "Current status: Only unit tests are available.";
		echo "Run 'make test' to execute unit tests.";
		exit 1; \
	fi

# 专门用于GGUF模型的启动目标
start-gguf: build
	@echo "Starting cLLM server with GGUF model..."
	@make start USE_GGUF=true USE_LIBTORCH=$(USE_LIBTORCH)

# 后台启动（不会阻塞当前终端/CI，适合脚本化 E2E 测试）
# 用法：make start-bg MODEL_PATH=... PORT=18080 QUANTIZATION=fp16
start-bg:
	@echo "Starting cLLM server in background..."
	@mkdir -p logs
	@if [ -f $(BUILD_DIR)/bin/cllm_server ]; then \
		if [ "$(USE_GGUF)" = "true" ]; then \
			MODEL_PATH_TO_USE=$(GGUF_MODEL_PATH); \
			PORT_TO_USE=$(GGUF_PORT); \
		else \
			MODEL_PATH_TO_USE=$(MODEL_PATH); \
			PORT_TO_USE=$(PORT); \
		fi; \
		echo "Configuration:"; \
		echo "  Model Path: $$MODEL_PATH_TO_USE"; \
		echo "  Model Type: $(if $(filter true,$(USE_GGUF)),GGUF,$(if $(filter true,$(USE_LIBTORCH)),TorchScript,Kylin))"; \
		echo "  Port: $$PORT_TO_USE"; \
		echo "  Log: logs/cllm_server_$$PORT_TO_USE.log"; \
		echo ""; \
		nohup $(BUILD_DIR)/bin/cllm_server \
			--model-path $$MODEL_PATH_TO_USE \
			--port $$PORT_TO_USE \
			--quantization $(QUANTIZATION) \
			--max-batch-size $(MAX_BATCH_SIZE) \
			--max-context-length $(MAX_CONTEXT_LENGTH) \
			$(if $(filter true,$(USE_LIBTORCH)),--use-libtorch,) \
			> logs/cllm_server_$$PORT_TO_USE.log 2>&1 & \
		echo $$! > /tmp/cllm_server_$$PORT_TO_USE.pid; \
		echo "PID: $$(cat /tmp/cllm_server_$$PORT_TO_USE.pid)"; \
		echo "Log: logs/cllm_server_$$PORT_TO_USE.log"; \
	else \
		echo "Error: cllm_server executable not found. Run 'make build' first."; \
		exit 1; \
	fi

# 专门用于GGUF模型的后台启动目标
start-gguf-bg:
	@echo "Starting cLLM server with GGUF model in background..."
	@make start-bg USE_GGUF=true USE_LIBTORCH=$(USE_LIBTORCH)

# GGUF 端到端测试 - 使用 Kylin Backend (方案1)
# 用法: make test-gguf-e2e
# 注意: 此目标使用 Kylin backend (不使用 --use-libtorch)
test-gguf-e2e: build
	@echo "=========================================="
	@echo "GGUF 格式端到端测试 (Kylin Backend)"
	@echo "=========================================="
	@echo "模型: $(GGUF_MODEL_PATH)"
	@echo "端口: $(GGUF_PORT)"
	@echo "后端: Kylin (不使用 LibTorch)"
	@echo ""
	@if [ ! -f "$(GGUF_MODEL_PATH)" ]; then \
		echo "❌ 错误: 模型文件不存在: $(GGUF_MODEL_PATH)"; \
		exit 1; \
	fi
	@if [ ! -f $(BUILD_DIR)/bin/cllm_server ]; then \
		echo "❌ 错误: cllm_server 可执行文件不存在"; \
		echo "请先运行: make build"; \
		exit 1; \
	fi
	@echo "✅ 启动服务器 (Kylin Backend)..."
	@mkdir -p logs
	@nohup $(BUILD_DIR)/bin/cllm_server \
		--model-path $(GGUF_MODEL_PATH) \
		--port $(GGUF_PORT) \
		--host 0.0.0.0 \
		--log-level info \
		> logs/cllm_server_gguf_test.log 2>&1 & \
	echo $$! > /tmp/cllm_server_$(GGUF_PORT).pid; \
	echo "服务器 PID: $$(cat /tmp/cllm_server_$(GGUF_PORT).pid)"; \
	echo "日志: logs/cllm_server_gguf_test.log"; \
	echo ""; \
	echo "等待服务器启动..."; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		sleep 1; \
		if curl -s http://localhost:$(GGUF_PORT)/health > /dev/null 2>&1; then \
			echo "✅ 服务器已启动 ($$i秒)"; \
			break; \
		fi; \
		if [ $$i -eq 10 ]; then \
			echo "❌ 服务器启动超时"; \
			echo "查看日志:"; \
			tail -30 logs/cllm_server_gguf_test.log; \
			kill $$(cat /tmp/cllm_server_$(GGUF_PORT).pid) 2>/dev/null || true; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="; \
	echo "测试 /generate 接口"; \
	echo "=========================================="; \
	echo "输入: hello"; \
	echo ""; \
	RESPONSE=$$(curl -s -X POST http://localhost:$(GGUF_PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "hello", "max_tokens": 50, "temperature": 0.7}'); \
	echo "响应:"; \
	echo "$$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$$RESPONSE"; \
	echo ""; \
	GENERATED_TEXT=$$(echo "$$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('text', data.get('generated_text', 'N/A')))" 2>/dev/null || echo "N/A"); \
	if [ "$$GENERATED_TEXT" != "N/A" ] && [ -n "$$GENERATED_TEXT" ]; then \
		echo "✅ 生成成功!"; \
		echo "生成的文本: $$GENERATED_TEXT"; \
	else \
		echo "⚠️  警告: 无法解析生成的文本"; \
	fi; \
	echo ""; \
	echo "停止服务器..."; \
	kill $$(cat /tmp/cllm_server_$(GGUF_PORT).pid) 2>/dev/null || true; \
	sleep 2; \
	echo "✅ 测试完成"; \
	echo "完整日志: logs/cllm_server_gguf_test.log"

# 快速测试 curl 命令（需要服务器已启动）
test-curl:
	@echo "测试 /generate 接口..."
	@curl -X POST http://localhost:$(GGUF_PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "hello", "max_tokens": 50, "temperature": 0.7}' \
		| python3 -m json.tool || echo "请求失败，请确保服务器已启动"

# ============================================
# GGUF Q4K 专用目标
# ============================================

# 启动使用 Q4K 模型的服务器（使用 Kylin 后端）
# 用法: make start-gguf-q4k
#       或: make start-gguf-q4k GGUF_Q4K_MODEL_PATH=/path/to/model.gguf GGUF_Q4K_PORT=18082
start-gguf-q4k: build
	@echo "=========================================="
	@echo "启动 cLLM 服务器 (GGUF Q4K 模型)"
	@echo "=========================================="
	@if [ ! -f "$(GGUF_Q4K_MODEL_PATH)" ]; then \
		echo "❌ 错误: 模型文件不存在: $(GGUF_Q4K_MODEL_PATH)"; \
		echo "请设置 GGUF_Q4K_MODEL_PATH 环境变量或修改 Makefile"; \
		exit 1; \
	fi
	@if [ ! -f $(BUILD_DIR)/bin/cllm_server ]; then \
		echo "❌ 错误: cllm_server 可执行文件不存在"; \
		echo "请先运行: make build"; \
		exit 1; \
	fi
	@echo "配置:"
	@echo "  模型路径: $(GGUF_Q4K_MODEL_PATH)"
	@echo "  端口: $(GGUF_Q4K_PORT)"
	@echo "  后端: Kylin (不使用 LibTorch)"
	@echo "  量化: Q4_K_M"
	@echo ""
	@echo "启动服务器..."
	@$(BUILD_DIR)/bin/cllm_server \
		--model-path $(GGUF_Q4K_MODEL_PATH) \
		--port $(GGUF_Q4K_PORT) \
		--host 0.0.0.0 \
		--log-level info \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--max-context-length $(MAX_CONTEXT_LENGTH)

# 后台启动 Q4K 服务器
start-gguf-q4k-bg: build
	@echo "=========================================="
	@echo "后台启动 cLLM 服务器 (GGUF Q4K 模型)"
	@echo "=========================================="
	@if [ ! -f "$(GGUF_Q4K_MODEL_PATH)" ]; then \
		echo "❌ 错误: 模型文件不存在: $(GGUF_Q4K_MODEL_PATH)"; \
		exit 1; \
	fi
	@if [ ! -f $(BUILD_DIR)/bin/cllm_server ]; then \
		echo "❌ 错误: cllm_server 可执行文件不存在"; \
		exit 1; \
	fi
	@mkdir -p logs
	@echo "配置:"
	@echo "  模型路径: $(GGUF_Q4K_MODEL_PATH)"
	@echo "  端口: $(GGUF_Q4K_PORT)"
	@echo "  后端: Kylin"
	@echo "  日志: logs/cllm_server_q4k_$(GGUF_Q4K_PORT).log"
	@echo ""
	@nohup $(BUILD_DIR)/bin/cllm_server \
		--model-path $(GGUF_Q4K_MODEL_PATH) \
		--port $(GGUF_Q4K_PORT) \
		--host 0.0.0.0 \
		--log-level debug \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--max-context-length $(MAX_CONTEXT_LENGTH) \
		> logs/cllm_server_q4k_$(GGUF_Q4K_PORT).log 2>&1 & \
	echo $$! > /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid; \
	echo "✅ 服务器已启动 (后台)"; \
	echo "  PID: $$(cat /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid)"; \
	echo "  日志: logs/cllm_server_q4k_$(GGUF_Q4K_PORT).log"; \
	echo "  端口: $(GGUF_Q4K_PORT)"; \
	echo ""; \
	echo "等待服务器就绪..."; \
	for i in 1 2 3 4 5 6 7 8 9 10; do \
		sleep 1; \
		if curl -s http://localhost:$(GGUF_Q4K_PORT)/health > /dev/null 2>&1; then \
			echo "✅ 服务器已就绪 ($$i秒)"; \
			break; \
		fi; \
		if [ $$i -eq 10 ]; then \
			echo "⚠️  警告: 服务器启动可能较慢，请检查日志"; \
			tail -20 logs/cllm_server_q4k_$(GGUF_Q4K_PORT).log; \
		fi; \
	done

# 测试 Q4K 模型的 /generate 接口（自动启动/停止服务器）
test-gguf-q4k: build
	@echo "=========================================="
	@echo "GGUF Q4K 模型端到端测试"
	@echo "=========================================="
	@echo "模型: $(GGUF_Q4K_MODEL_PATH)"
	@echo "端口: $(GGUF_Q4K_PORT)"
	@echo "后端: Kylin"
	@echo ""
	@if [ ! -f "$(GGUF_Q4K_MODEL_PATH)" ]; then \
		echo "❌ 错误: 模型文件不存在: $(GGUF_Q4K_MODEL_PATH)"; \
		exit 1; \
	fi
	@if [ ! -f $(BUILD_DIR)/bin/cllm_server ]; then \
		echo "❌ 错误: cllm_server 可执行文件不存在"; \
		exit 1; \
	fi
	@echo "✅ 启动服务器 (后台)..."
	@mkdir -p logs
	@nohup $(BUILD_DIR)/bin/cllm_server \
		--model-path $(GGUF_Q4K_MODEL_PATH) \
		--port $(GGUF_Q4K_PORT) \
		--host 0.0.0.0 \
		--log-level info \
		--max-batch-size $(MAX_BATCH_SIZE) \
		--max-context-length $(MAX_CONTEXT_LENGTH) \
		> logs/cllm_server_q4k_test.log 2>&1 & \
	echo $$! > /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid; \
	echo "服务器 PID: $$(cat /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid)"; \
	echo "日志: logs/cllm_server_q4k_test.log"; \
	echo ""; \
	echo "等待服务器启动..."; \
	for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		sleep 2; \
		if curl -s http://localhost:$(GGUF_Q4K_PORT)/health > /dev/null 2>&1; then \
			echo "✅ 服务器已启动 ($$(expr $$i \* 2)秒)"; \
			break; \
		fi; \
		if [ $$i -eq 15 ]; then \
			echo "❌ 服务器启动超时"; \
			echo "查看日志:"; \
			tail -50 logs/cllm_server_q4k_test.log; \
			kill $$(cat /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid) 2>/dev/null || true; \
			exit 1; \
		fi; \
	done; \
	echo ""; \
	echo "=========================================="; \
	echo "测试 /generate 接口"; \
	echo "=========================================="; \
	echo "测试 1: 简单生成"; \
	echo "输入: Hello, how are you?"; \
	echo ""; \
	RESPONSE1=$$(curl -s -X POST http://localhost:$(GGUF_Q4K_PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello, how are you?", "max_tokens": 50, "temperature": 0.7}'); \
	echo "响应:"; \
	echo "$$RESPONSE1" | python3 -m json.tool 2>/dev/null || echo "$$RESPONSE1"; \
	echo ""; \
	GENERATED_TEXT1=$$(echo "$$RESPONSE1" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('text', data.get('generated_text', 'N/A')))" 2>/dev/null || echo "N/A"); \
	if [ "$$GENERATED_TEXT1" != "N/A" ] && [ -n "$$GENERATED_TEXT1" ]; then \
		echo "✅ 生成成功!"; \
		echo "生成的文本: $$GENERATED_TEXT1"; \
	else \
		echo "⚠️  警告: 无法解析生成的文本"; \
	fi; \
	echo ""; \
	echo "测试 2: 中文生成"; \
	echo "输入: 你好，请介绍一下自己"; \
	echo ""; \
	RESPONSE2=$$(curl -s -X POST http://localhost:$(GGUF_Q4K_PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "你好，请介绍一下自己", "max_tokens": 50, "temperature": 0.7}'); \
	echo "响应:"; \
	echo "$$RESPONSE2" | python3 -m json.tool 2>/dev/null || echo "$$RESPONSE2"; \
	echo ""; \
	GENERATED_TEXT2=$$(echo "$$RESPONSE2" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('text', data.get('generated_text', 'N/A')))" 2>/dev/null || echo "N/A"); \
	if [ "$$GENERATED_TEXT2" != "N/A" ] && [ -n "$$GENERATED_TEXT2" ]; then \
		echo "✅ 生成成功!"; \
		echo "生成的文本: $$GENERATED_TEXT2"; \
	else \
		echo "⚠️  警告: 无法解析生成的文本"; \
	fi; \
	echo ""; \
	echo "停止服务器..."; \
	kill $$(cat /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid) 2>/dev/null || true; \
	sleep 2; \
	echo "✅ 测试完成"; \
	echo "完整日志: logs/cllm_server_q4k_test.log"

# 快速测试 Q4K 服务器的 /generate 接口（需要服务器已启动）
test-curl-q4k:
	@echo "测试 Q4K 服务器 /generate 接口..."
	@curl -X POST http://localhost:$(GGUF_Q4K_PORT)/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "介绍人工智能", "max_tokens": 8, "temperature": 0.7}' \
		| python3 -m json.tool || echo "请求失败，请确保服务器已启动 (端口: $(GGUF_Q4K_PORT))"

run: start

stop:
	@echo "Stopping cLLM server..."
	@if [ -f /tmp/cllm_server_$(PORT).pid ]; then \
		kill $$(cat /tmp/cllm_server_$(PORT).pid) 2>/dev/null || true; \
		rm -f /tmp/cllm_server_$(PORT).pid; \
		echo "Stopped server on port $(PORT)"; \
	fi
	@if [ -f /tmp/cllm_server_$(GGUF_PORT).pid ]; then \
		kill $$(cat /tmp/cllm_server_$(GGUF_PORT).pid) 2>/dev/null || true; \
		rm -f /tmp/cllm_server_$(GGUF_PORT).pid; \
		echo "Stopped server on port $(GGUF_PORT)"; \
	fi
	@if [ -f /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid ]; then \
		kill $$(cat /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid) 2>/dev/null || true; \
		rm -f /tmp/cllm_server_q4k_$(GGUF_Q4K_PORT).pid; \
		echo "Stopped Q4K server on port $(GGUF_Q4K_PORT)"; \
	fi
	@pkill -f cllm_server || echo "No other cllm_server processes found"
	@echo "Server stopped!"

install-deps:
	@echo "Installing dependencies..."
	@$(VENV_ACTIVATE) pip install cmake
	@echo "Dependencies installed!"

setup-env:
	@echo "Setting up build environment..."
	@mkdir -p third_party
	@if [ ! -f third_party/BS_thread_pool.hpp ]; then \
		echo "Downloading BS::thread_pool..."; \
		git clone --depth 1 https://github.com/bshoshany/thread-pool.git third_party/thread-pool-temp; \
		cp third_party/thread-pool-temp/include/BS_thread_pool.hpp third_party/; \
		rm -rf third_party/thread-pool-temp; \
	fi
	@if [ ! -d third_party/googletest ]; then \
		echo "Downloading Google Test..."; \
		git clone --depth 1 https://github.com/google/googletest.git third_party/googletest; \
	fi
	@echo "Environment setup completed!"

check-env:
	@echo "Checking build environment..."
	@echo "C++ Compiler:"
	@clang++ --version | head -1
	@echo "CMake:"
	@$(VENV_ACTIVATE) cmake --version | head -1
	@echo "Make:"
	@make --version | head -1
	@echo "BS::thread_pool:"
	@if [ -f third_party/BS_thread_pool.hpp ]; then \
		echo "  ✓ Installed"; \
	else \
		echo "  ✗ Not found"; \
	fi
	@echo "Google Test:"
	@if [ -d third_party/googletest ]; then \
		echo "  ✓ Downloaded"; \
	else \
		echo "  ✗ Not found"; \
	fi

help:
	@echo "cLLM Build System"
	@echo "  test-tokenizer    - Run tokenizer specific tests"
	@echo "  integration-test  - Run integration tests"
	@echo "  test-all          - Run all tests (unit + integration)"
	@echo "=================="
	@echo ""
	@echo "Available targets:";
	@echo "  all              - Build the project (default)";
	@echo "  build            - Build the project in Release mode";
	@echo "  build-debug      - Build the project in Debug mode";
	@echo "  clean            - Remove build directory";
	@echo "  rebuild          - Clean and build";
	@echo "  test             - Run unit tests";
	@echo "  start            - Start cLLM server (use MODEL_PATH to specify model)";
	@echo "  start-gguf       - Start cLLM server with GGUF model";
	@echo "  start-bg         - Start cLLM server in background";
	@echo "  start-gguf-bg    - Start cLLM server with GGUF model in background";
	@echo "  start-gguf-q4k   - Start cLLM server with GGUF Q4K model (Kylin backend)";
	@echo "  start-gguf-q4k-bg - Start cLLM server with GGUF Q4K model in background";
	@echo "  test-gguf-e2e    - Run GGUF end-to-end test (Kylin backend)";
	@echo "  test-gguf-q4k    - Run GGUF Q4K end-to-end test (auto start/stop)";
	@echo "  test-curl        - Test /generate endpoint with curl (server must be running)";
	@echo "  test-curl-q4k    - Test Q4K server /generate endpoint with curl";
	@echo "  run              - Alias for start";
	@echo "  stop             - Stop cLLM server";
	@echo "  check-model      - Verify model path exists";
	@echo "  install-deps     - Install required dependencies";
	@echo "  setup-env        - Setup build environment (download dependencies)";
	@echo "  check-env        - Check build environment status";
	@echo "  help             - Show this help message";
	@echo "";
	@echo "Configuration Variables:";
	@echo "  MODEL_PATH           - Path to model file (default: /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3_0.6b_torchscript_fp32.pt)";
	@echo "  GGUF_MODEL_PATH      - Path to GGUF model file (default: /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf)";
	@echo "  GGUF_Q4K_MODEL_PATH - Path to GGUF Q4K model file (default: /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf)";
	@echo "  PORT                 - Server port (default: 18080)";
	@echo "  GGUF_PORT            - Server port for GGUF model (default: 18081)";
	@echo "  GGUF_Q4K_PORT        - Server port for GGUF Q4K model (default: 18082)";
	@echo "  QUANTIZATION         - Quantization type: fp16, int8 (default: fp16)";
	@echo "  MAX_BATCH_SIZE       - Maximum batch size (default: 8)";
	@echo "  MAX_CONTEXT_LENGTH   - Maximum context length (default: 2048)";
	@echo "  USE_LIBTORCH         - Use LibTorch backend: true, false (default: true)";
	@echo "  USE_GGUF             - Use GGUF model: true, false (default: false)";
	@echo "";
	@echo "Examples:";
	@echo "  make                 - Build the project";
	@echo "  make clean           - Clean build files";
	@echo "  make rebuild         - Clean and rebuild";
	@echo "  make test            - Run all unit tests";
	@echo "  make start           - Start server with default configuration";
	@echo "  make start MODEL_PATH=/path/to/model PORT=9000";
	@echo "  make start-gguf      - Start server with GGUF model";
	@echo "  make start-gguf GGUF_MODEL_PATH=/path/to/model.gguf GGUF_PORT=9000";
	@echo "  make start-bg        - Start server in background";
	@echo "  make start-gguf-bg   - Start server with GGUF model in background";
	@echo "  make start-gguf-q4k  - Start server with GGUF Q4K model (Kylin backend)";
	@echo "  make start-gguf-q4k-bg - Start server with GGUF Q4K model in background";
	@echo "  make test-gguf-e2e   - Run GGUF end-to-end test (auto start/stop server)";
	@echo "  make test-gguf-q4k   - Run GGUF Q4K end-to-end test (auto start/stop server)";
	@echo "  make test-curl      - Test /generate endpoint (server must be running)";
	@echo "  make test-curl-q4k  - Test Q4K server /generate endpoint (server must be running)";
	@echo "  make start USE_LIBTORCH=false";
	@echo "  make stop            - Stop the server";
	@echo "";
	@echo "Note:";
	@echo "  - Default model uses LibTorch backend with TorchScript format (.pt)";
	@echo "  - Set USE_LIBTORCH=false to use Kylin backend with .bin format";
	@echo "  - Set USE_GGUF=true to use GGUF model format (.gguf)";
	@echo "  - Main server executable (cllm_server) is implemented and ready to use"
