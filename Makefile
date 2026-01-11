.PHONY: all build clean rebuild test run start stop help install-deps setup-env check-model

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
PORT ?= 18080
QUANTIZATION ?= fp16
MAX_BATCH_SIZE ?= 8
MAX_CONTEXT_LENGTH ?= 2048
USE_LIBTORCH ?= true

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
	@echo "Configuration:"
	@echo "  Model Path: $(MODEL_PATH)"
	@echo "  Port: $(PORT)"
	@echo "  Quantization: $(QUANTIZATION)"
	@echo "  Max Batch Size: $(MAX_BATCH_SIZE)"
	@echo "  Max Context Length: $(MAX_CONTEXT_LENGTH)"
	@echo "  Backend: $(if $(filter true,$(USE_LIBTORCH)),LibTorch,Kylin)"
	@echo ""
	@if [ -f $(BUILD_DIR)/bin/cllm_server ]; then \
		$(BUILD_DIR)/bin/cllm_server \
			--model-path $(MODEL_PATH) \
			--port $(PORT) \
			--quantization $(QUANTIZATION) \
			--max-batch-size $(MAX_BATCH_SIZE) \
			--max-context-length $(MAX_CONTEXT_LENGTH) \
			$(if $(filter true,$(USE_LIBTORCH)),--use-libtorch,); \
	else \
		echo "Error: cllm_server executable not found."; \
		echo "Note: The main server executable is not yet implemented."; \
		echo "Current status: Only unit tests are available."; \
		echo "Run 'make test' to execute unit tests."; \
		exit 1; \
	fi

# 后台启动（不会阻塞当前终端/CI，适合脚本化 E2E 测试）
# 用法：make start-bg MODEL_PATH=... PORT=18080 QUANTIZATION=fp16
start-bg:
	@echo "Starting cLLM server in background..."
	@mkdir -p logs
	@if [ -f $(BUILD_DIR)/bin/cllm_server ]; then \
		nohup $(BUILD_DIR)/bin/cllm_server \
			--model-path $(MODEL_PATH) \
			--port $(PORT) \
			--quantization $(QUANTIZATION) \
			--max-batch-size $(MAX_BATCH_SIZE) \
			--max-context-length $(MAX_CONTEXT_LENGTH) \
			$(if $(filter true,$(USE_LIBTORCH)),--use-libtorch,) \
			> logs/cllm_server_$(PORT).log 2>&1 & \
		echo $$! > /tmp/cllm_server_$(PORT).pid; \
		echo "PID: $$(cat /tmp/cllm_server_$(PORT).pid)"; \
		echo "Log: logs/cllm_server_$(PORT).log"; \
	else \
		echo "Error: cllm_server executable not found. Run 'make build' first."; \
		exit 1; \
	fi

run: start

stop:
	@echo "Stopping cLLM server..."
	@pkill -f cllm_server || echo "No cllm_server process found"
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
	@echo "Available targets:"
	@echo "  all              - Build the project (default)"
	@echo "  build            - Build the project in Release mode"
	@echo "  build-debug      - Build the project in Debug mode"
	@echo "  clean            - Remove build directory"
	@echo "  rebuild          - Clean and build"
	@echo "  test             - Run unit tests"
	@echo "  start            - Start cLLM server (use MODEL_PATH to specify model)"
	@echo "  run              - Alias for start"
	@echo "  stop             - Stop cLLM server"
	@echo "  check-model      - Verify model path exists"
	@echo "  install-deps     - Install required dependencies"
	@echo "  setup-env        - Setup build environment (download dependencies)"
	@echo "  check-env        - Check build environment status"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Configuration Variables:"
	@echo "  MODEL_PATH           - Path to model file (default: /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3_0.6b_torchscript_fp32.pt)"
	@echo "  PORT                 - Server port (default: 18080)"
	@echo "  QUANTIZATION         - Quantization type: fp16, int8 (default: fp16)"
	@echo "  MAX_BATCH_SIZE       - Maximum batch size (default: 8)"
	@echo "  MAX_CONTEXT_LENGTH   - Maximum context length (default: 2048)"
	@echo "  USE_LIBTORCH         - Use LibTorch backend: true, false (default: true)"
	@echo ""
	@echo "Examples:"
	@echo "  make                 - Build the project"
	@echo "  make clean           - Clean build files"
	@echo "  make rebuild         - Clean and rebuild"
	@echo "  make test            - Run all unit tests"
	@echo "  make start           - Start server with default configuration"
	@echo "  make start MODEL_PATH=/path/to/model PORT=9000"
	@echo "  make start USE_LIBTORCH=false"
	@echo "  make stop            - Stop the server"
	@echo ""
	@echo "Note:"
	@echo "  - Default model uses LibTorch backend with TorchScript format (.pt)"
	@echo "  - Set USE_LIBTORCH=false to use Kylin backend with .bin format"
	@echo "  - Main server executable (cllm_server) is implemented and ready to use"
