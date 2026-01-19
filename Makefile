.PHONY: all build build-debug clean rebuild test integration-test test-all test-tokenizer run start start-gpu start-cpu start-bg stop status tail-logs setup-env check-env help

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

# 日志目录
LOG_DIR = logs
LOG_FILE = $(LOG_DIR)/cllm_server.log
ERROR_LOG_FILE = $(LOG_DIR)/cllm_server_error.log

# 配置文件
CONFIG_GPU = config/config_gpu.yaml
CONFIG_CPU = config/config_cpu.yaml
CONFIG_DEFAULT = $(CONFIG_GPU)

all: build

build:
	@echo "Building cLLM..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) make -j$(NUM_JOBS)
	@echo "Build completed successfully!"
	@echo "Note: Examples (http_server_example, basic_usage) are disabled."
	@echo "Use 'make start' to run the cLLM server."

build-debug:
	@echo "Building cLLM in Debug mode..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(VENV_ACTIVATE) cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
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

start: build start-gpu

start-gpu: build
	@echo "========================================"
	@echo "Starting cLLM server (GPU Mode)"
	@echo "========================================"
	@mkdir -p $(LOG_DIR);
	@echo "Configuration:"
	@echo "  Config File: $(CONFIG_GPU)"
	@echo "  Log File: $(LOG_FILE)"
	@echo "  Error Log: $(ERROR_LOG_FILE)"
	@echo "  Port: 18085"
	@echo "  Mode: GPU (n_gpu_layers: 99)"
	@echo ""
	@echo "Server starting... Press Ctrl+C to stop."
	@echo "Check logs: tail -f $(LOG_FILE)"
	@echo ""
	@$(BUILD_DIR)/bin/cllm_server --config $(CONFIG_GPU) 2>&1 | tee -a $(LOG_FILE);

start-cpu: build
	@echo "========================================"
	@echo "Starting cLLM server (CPU Mode)"
	@echo "========================================"
	@mkdir -p $(LOG_DIR);
	@echo "Configuration:"
	@echo "  Config File: $(CONFIG_CPU)"
	@echo "  Log File: $(LOG_DIR)/cllm_server_cpu.log"
	@echo "  Error Log: $(LOG_DIR)/cllm_server_cpu_error.log"
	@echo "  Port: 18085"
	@echo "  Mode: CPU (n_gpu_layers: 0)"
	@echo ""
	@echo "Server starting... Press Ctrl+C to stop."
	@echo "Check logs: tail -f $(LOG_DIR)/cllm_server_cpu.log"
	@echo ""
	@$(BUILD_DIR)/bin/cllm_server --config $(CONFIG_CPU) 2>&1 | tee -a $(LOG_DIR)/cllm_server_cpu.log;

start-bg: build
	@echo "========================================"
	@echo "Starting cLLM server in background (GPU Mode)"
	@echo "========================================"
	@mkdir -p $(LOG_DIR);
	@echo "Configuration:"
	@echo "  Config File: $(CONFIG_GPU)"
	@echo "  Log File: $(LOG_FILE)"
	@echo "  Error Log: $(ERROR_LOG_FILE)"
	@echo "  Port: 18085"
	@echo "  Mode: Background (GPU)"
	@echo ""
	@if pgrep -f "cllm_server.*$(CONFIG_GPU)" > /dev/null; then \
		echo "Warning: cLLM server is already running."; \
		echo "Use 'make stop' to stop it first."; \
		exit 1; \
	fi;
	@nohup $(BUILD_DIR)/bin/cllm_server --config $(CONFIG_GPU) > $(LOG_FILE) 2> $(ERROR_LOG_FILE) < /dev/null &
	@SERVER_PID=$$!;
	@echo "Server started with PID: $$SERVER_PID";
	@echo $$SERVER_PID > $(LOG_DIR)/cllm_server.pid;
	@echo "Check logs: tail -f $(LOG_FILE)";
	@echo "Stop server: make stop";
	@echo "";

tail-logs:
	@echo "========================================"
	@echo "Tailing cLLM server logs"
	@echo "========================================"
	@tail -f $(LOG_FILE);

stop:
	@echo "========================================"
	@echo "Stopping cLLM server"
	@echo "========================================"
	@if [ -f $(LOG_DIR)/cllm_server.pid ]; then \
		SERVER_PID=$$(cat $(LOG_DIR)/cllm_server.pid); \
		echo "Stopping server with PID: $$SERVER_PID"; \
		kill $$SERVER_PID 2>/dev/null || true; \
		rm -f $(LOG_DIR)/cllm_server.pid; \
		echo "Server stopped."; \
	elif pgrep -f "cllm_server" > /dev/null; then \
		echo "Found running cLLM server(s):"; \
		pgrep -f "cllm_server" | while read PID; do \
			echo "  Stopping PID: $$PID"; \
			kill $$PID 2>/dev/null || true; \
		done; \
		echo "All cLLM servers stopped."; \
	else \
		echo "No running cLLM server found."; \
	fi;
	@echo "";

status:
	@echo "========================================"
	@echo "cLLM server status"
	@echo "========================================"
	@if [ -f $(LOG_DIR)/cllm_server.pid ]; then \
		SERVER_PID=$$(cat $(LOG_DIR)/cllm_server.pid); \
		if ps -p $$SERVER_PID > /dev/null 2>&1; then \
			echo "Status: Running (PID: $$SERVER_PID)"; \
			echo "Config: $(CONFIG_GPU)"; \
			echo "Port: 18085"; \
			echo "Log: $(LOG_FILE)"; \
		else \
			echo "Status: Stopped (Stale PID file)"; \
			echo "Removing stale PID file..."; \
			rm -f $(LOG_DIR)/cllm_server.pid; \
		fi; \
	elif pgrep -f "cllm_server" > /dev/null; then \
		echo "Status: Running (PID: $$(pgrep -f cllm_server))"; \
		echo "Config: Unknown (started without make)"; \
	else \
		echo "Status: Stopped"; \
	fi; \
	@echo "";
	@echo "Quick commands:";
	@echo "  make start-gpu    - Start server (GPU mode, foreground)";
	@echo "  make start-cpu    - Start server (CPU mode, foreground)";
	@echo "  make start-bg     - Start server (background)";
	@echo "  make stop         - Stop server";
	@echo "  make tail-logs    - View server logs";
	@echo "  make status       - Check server status";
	@echo "";

run: start

setup-env:
	@echo "Setting up development environment..."
	@mkdir -p third_party
	@if [ ! -f third_party/BS_thread_pool.hpp ]; then \
		echo "Downloading BS::thread_pool...";
		git clone --depth 1 https://github.com/bshoshany/thread-pool.git third_party/thread-pool-temp;
		cp third_party/thread-pool-temp/include/BS_thread_pool.hpp third_party/;
		rm -rf third_party/thread-pool-temp;
	fi
	@if [ ! -d third_party/googletest ]; then \
		echo "Downloading Google Test...";
		git clone --depth 1 https://github.com/google/googletest.git third_party/googletest;
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
	@echo "Python:"
	@python3 --version
	@echo "CUDNN (if available):"
	@if [ -f /usr/local/cuda/include/cudnn_version.h ]; then \
		echo "CUDA/CUDNN installed";
	else \
		echo "CUDA/CUDNN not found (optional)";
	fi
	@echo "";
	@echo "Build environment check completed!"

help:
	@echo "========================================"
	@echo "cLLM Makefile Help"
	@echo "========================================"
	@echo "";
	@echo "Build targets:"
	@echo "  make build          - Build the project (Release mode)"
	@echo "  make build-debug    - Build the project (Debug mode)"
	@echo "  make clean          - Clean build files"
	@echo "  make rebuild        - Rebuild the project"
	@echo "";
	@echo "Server targets:"
	@echo "  make start          - Start server (GPU mode)"
	@echo "  make start-gpu      - Start server (GPU mode)"
	@echo "  make start-cpu      - Start server (CPU mode)"
	@echo "  make start-bg       - Start server in background"
	@echo "  make stop           - Stop server"
	@echo "  make status         - Check server status"
	@echo "  make tail-logs      - View server logs"
	@echo "";
	@echo "Test targets:"
	@echo "  make test           - Run unit tests"
	@echo "  make integration-test - Run integration tests"
	@echo "  make test-all       - Run all tests"
	@echo "  make test-tokenizer - Run tokenizer tests"
	@echo "";
	@echo "Environment targets:"
	@echo "  make setup-env      - Setup development environment"
	@echo "  make check-env      - Check build environment"
	@echo "";
	@echo "Other targets:"
	@echo "  make help           - Show this help message"
	@echo "";
