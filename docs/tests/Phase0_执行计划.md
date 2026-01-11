# Phase 0: 准备阶段 执行计划

**负责Agent**: Agent-0  
**预计耗时**: 2小时5分钟  
**依赖**: 无  
**执行时间**: T+0h ~ T+2h5m  

---

## 📋 阶段目标

准备完整的测试环境，包括模型下载、测试数据生成、编译测试程序和环境验证。

---

## 📊 任务清单

| 任务ID | 任务名称 | 耗时 | 优先级 | 状态 |
|--------|---------|------|--------|------|
| P0.1 | 验证 Qwen3 模型 | 5min | 高 | ⏳ 待执行 |
| P0.2 | 生成测试数据 | 30min | 高 | ⏳ 待执行 |
| P0.3 | 配置环境变量 | 15min | 高 | ⏳ 待执行 |
| P0.4 | 编译所有测试程序 | 60min | 高 | ⏳ 待执行 |
| P0.5 | 验证环境就绪 | 15min | 高 | ⏳ 待执行 |

**总计**: 5个任务，125分钟（2小时5分钟）

---

## 📝 详细任务说明

### P0.1: 验证 Qwen3 模型 (5分钟)

**目标**: 验证本地已有的 Qwen3-0.6B 模型完整性

**说明**: 本地已存在完整模型在 `model/Qwen/Qwen3-0.6B/` 目录，无需重新下载

**执行命令**:
```bash
# 检查模型文件完整性
MODEL_PATH="model/Qwen/Qwen3-0.6B"

# 验证必要文件
test -f "${MODEL_PATH}/tokenizer.json" && echo "✅ tokenizer.json"
test -f "${MODEL_PATH}/tokenizer_config.json" && echo "✅ tokenizer_config.json"
test -f "${MODEL_PATH}/config.json" && echo "✅ config.json"
test -f "${MODEL_PATH}/model.safetensors" && echo "✅ model.safetensors"
test -f "${MODEL_PATH}/vocab.json" && echo "✅ vocab.json"
test -f "${MODEL_PATH}/merges.txt" && echo "✅ merges.txt"

# 检查模型大小
du -sh "${MODEL_PATH}"
```

**验证标准**:
```bash
# 所有必需文件必须存在
# 模型大小约 1.5GB (model.safetensors 约 1.5GB)
```

**输出**:
- `model/Qwen/Qwen3-0.6B/` 目录包含完整模型文件
- 模型大小约 1.5GB

---

### P0.2: 生成测试数据 (30分钟)

**目标**: 生成所有测试所需的数据文件

**执行命令**:
```bash
# 运行测试数据生成脚本
python3 scripts/generate_test_data.py

# 脚本会生成以下数据：
# - test_data/tokenizer_test_data.json
# - test_data/inference_test_data.json
# - test_data/performance_test_data.json
# - test_data/stress_test_data.json
# - test_data/e2e_scenarios.json
```

**生成的测试数据**:

1. **Tokenizer 测试数据** (`tokenizer_test_data.json`):
```json
{
  "english_texts": [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial Intelligence is transforming our world."
  ],
  "chinese_texts": [
    "你好，世界！",
    "人工智能正在改变我们的世界。",
    "自然语言处理是人工智能的重要分支。"
  ],
  "mixed_texts": [
    "Hello 世界！",
    "AI人工智能 and Machine Learning机器学习"
  ],
  "special_chars": [
    "😀🎉🚀",
    "Symbol: @#$%^&*()",
    "Unicode: \u4e2d\u6587"
  ]
}
```

2. **推理测试数据** (`inference_test_data.json`):
```json
{
  "prompts": [
    "What is the capital of China?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate factorial."
  ],
  "expected_keywords": [
    ["Beijing", "capital"],
    ["quantum", "computing", "bits"],
    ["def", "factorial", "return"]
  ]
}
```

3. **性能测试数据** (`performance_test_data.json`):
```json
{
  "batch_sizes": [1, 4, 8, 16],
  "sequence_lengths": [10, 50, 100, 500, 1000],
  "test_iterations": 100
}
```

**验证标准**:
```bash
# 检查数据文件
ls -lh test_data/*.json
wc -l test_data/*.json

# 验证JSON格式
python3 -m json.tool test_data/tokenizer_test_data.json > /dev/null && echo "✅ Valid JSON"
```

**输出**:
- `test_data/` 目录包含5个测试数据文件
- 总大小约 10MB

---

### P0.3: 配置环境变量 (15分钟)

**目标**: 设置测试所需的所有环境变量

**执行命令**:
```bash
# 创建环境配置文件
cat > test_env.sh << 'EOF'
#!/bin/bash

# 项目根目录
export CLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型路径 (使用本地已有的 Qwen3-0.6B 模型)
export CLLM_TEST_MODEL_PATH="${CLLM_ROOT}/model/Qwen/Qwen3-0.6B"

# 测试数据路径
export CLLM_TEST_DATA_PATH="${CLLM_ROOT}/tests/data"

# 测试报告路径
export CLLM_TEST_REPORTS="${CLLM_ROOT}/test_reports"

# 日志路径
export CLLM_LOG_DIR="${CLLM_ROOT}/logs"

# 线程数
export CLLM_NUM_THREADS=8

# 设备
export CLLM_DEVICE="cpu"  # 或 "cuda:0"

# 日志级别
export CLLM_LOG_LEVEL="INFO"

echo "✅ Environment configured:"
echo "  MODEL_PATH: ${CLLM_TEST_MODEL_PATH}"
echo "  DATA_PATH: ${CLLM_TEST_DATA_PATH}"
echo "  REPORTS: ${CLLM_TEST_REPORTS}"
echo "  LOG_DIR: ${CLLM_LOG_DIR}"
EOF

chmod +x test_env.sh

# 加载环境变量
source test_env.sh

# 创建必要目录
mkdir -p "${CLLM_TEST_REPORTS}"
mkdir -p "${CLLM_LOG_DIR}"
```

**验证标准**:
```bash
# 验证环境变量
echo "MODEL_PATH: ${CLLM_TEST_MODEL_PATH}"
echo "DATA_PATH: ${CLLM_TEST_DATA_PATH}"

# 验证目录存在
test -d "${CLLM_TEST_MODEL_PATH}" && echo "✅ Model directory exists"
test -d "${CLLM_TEST_DATA_PATH}" && echo "✅ Data directory exists"
test -d "${CLLM_TEST_REPORTS}" && echo "✅ Reports directory exists"
```

**输出**:
- `test_env.sh` 配置文件
- 所有必要目录已创建

---

### P0.4: 编译所有测试程序 (60分钟)

**目标**: 编译所有测试二进制文件

**执行命令**:
```bash
# 进入构建目录
cd build

# 配置 CMake（启用测试和 tokenizers-cpp）
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_TOKENIZERS_CPP=ON \
  -DBUILD_TESTS=ON \
  -DCMAKE_CXX_STANDARD=17

# 编译所有测试（使用8个并行任务）
make -j8 all_tests

# 或分别编译各个测试
make -j8 test_http_server
make -j8 test_hf_tokenizer
make -j8 test_model_executor
make -j8 test_libtorch_backend
make -j8 test_qwen3_model
make -j8 test_http_tokenizer_integration
make -j8 test_tokenizer_executor_integration
make -j8 test_executor_backend_integration
make -j8 test_backend_qwen3_integration
make -j8 test_frontend_subsystem
make -j8 test_inference_subsystem
make -j8 test_e2e_subsystem
make -j8 test_system_functionality
make -j8 test_performance_benchmark
make -j8 test_stress_stability
make -j8 test_e2e_scenarios
```

**编译目标列表**:

| 测试程序 | 说明 | 大小估计 |
|---------|------|----------|
| `test_http_server` | HTTP Server 单元测试 | ~5MB |
| `test_hf_tokenizer` | HFTokenizer 单元测试 | ~12MB |
| `test_model_executor` | ModelExecutor 单元测试 | ~8MB |
| `test_libtorch_backend` | LibTorch Backend 单元测试 | ~15MB |
| `test_qwen3_model` | Qwen3 模型测试 | ~10MB |
| `test_http_tokenizer_integration` | HTTP+Tokenizer 集成 | ~10MB |
| `test_tokenizer_executor_integration` | Tokenizer+Executor 集成 | ~12MB |
| `test_executor_backend_integration` | Executor+Backend 集成 | ~15MB |
| `test_backend_qwen3_integration` | Backend+Qwen3 集成 | ~18MB |
| `test_frontend_subsystem` | 前端子系统测试 | ~12MB |
| `test_inference_subsystem` | 推理子系统测试 | ~20MB |
| `test_e2e_subsystem` | E2E 子系统测试 | ~22MB |
| `test_system_functionality` | 系统功能测试 | ~25MB |
| `test_performance_benchmark` | 性能基准测试 | ~20MB |
| `test_stress_stability` | 压力稳定性测试 | ~20MB |
| `test_e2e_scenarios` | E2E 场景测试 | ~25MB |

**验证标准**:
```bash
# 检查所有测试二进制是否存在
cd build/bin
for test in test_*; do
  if [ -f "$test" ]; then
    echo "✅ $test ($(du -h $test | cut -f1))"
  else
    echo "❌ $test NOT FOUND"
  fi
done

# 测试是否可执行
./test_http_server --help > /dev/null 2>&1 && echo "✅ Executable"
```

**输出**:
- `build/bin/` 目录包含16个测试二进制
- 总大小约 250MB

---

### P0.5: 验证环境就绪 (15分钟)

**目标**: 确认所有准备工作完成，环境可用

**执行脚本**:
```bash
#!/bin/bash
# verify_environment.sh

echo "========================================="
echo "环境验证开始"
echo "========================================="
echo

# 1. 检查模型文件
echo "1. 检查模型文件..."
MODEL_PATH="${CLLM_TEST_MODEL_PATH}"
if [ -f "${MODEL_PATH}/tokenizer.json" ]; then
  echo "  ✅ tokenizer.json"
else
  echo "  ❌ tokenizer.json NOT FOUND"
  exit 1
fi

if [ -f "${MODEL_PATH}/config.json" ]; then
  echo "  ✅ config.json"
else
  echo "  ❌ config.json NOT FOUND"
  exit 1
fi

# 2. 检查测试数据
echo
echo "2. 检查测试数据..."
DATA_PATH="${CLLM_TEST_DATA_PATH}"
for data_file in tokenizer_test_data.json inference_test_data.json performance_test_data.json; do
  if [ -f "${DATA_PATH}/${data_file}" ]; then
    echo "  ✅ ${data_file}"
  else
    echo "  ❌ ${data_file} NOT FOUND"
    exit 1
  fi
done

# 3. 检查环境变量
echo
echo "3. 检查环境变量..."
for var in CLLM_TEST_MODEL_PATH CLLM_TEST_DATA_PATH CLLM_TEST_REPORTS; do
  if [ -n "${!var}" ]; then
    echo "  ✅ ${var}=${!var}"
  else
    echo "  ❌ ${var} NOT SET"
    exit 1
  fi
done

# 4. 检查编译产物
echo
echo "4. 检查编译产物..."
cd build/bin
TEST_COUNT=0
for test in test_*; do
  if [ -f "$test" ] && [ -x "$test" ]; then
    TEST_COUNT=$((TEST_COUNT + 1))
  fi
done
echo "  ✅ 找到 ${TEST_COUNT} 个测试程序"

if [ ${TEST_COUNT} -lt 10 ]; then
  echo "  ⚠️  警告: 测试程序数量少于预期"
fi

# 5. 运行快速健康检查
echo
echo "5. 运行健康检查..."

# 测试 HFTokenizer 是否可用
./test_hf_tokenizer --gtest_list_tests > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "  ✅ HFTokenizer 测试可执行"
else
  echo "  ❌ HFTokenizer 测试执行失败"
  exit 1
fi

# 6. 检查磁盘空间
echo
echo "6. 检查磁盘空间..."
AVAILABLE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
echo "  可用空间: ${AVAILABLE_SPACE}"

# 7. 生成验证报告
echo
echo "========================================="
echo "环境验证完成 ✅"
echo "========================================="
echo
echo "验证报告:" > "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 模型文件: ✅" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 测试数据: ✅" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 环境变量: ✅" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 测试程序: ✅ (${TEST_COUNT}个)" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 健康检查: ✅" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"
echo "  - 验证时间: $(date)" >> "${CLLM_TEST_REPORTS}/phase0_verification.txt"

echo "验证报告已保存: ${CLLM_TEST_REPORTS}/phase0_verification.txt"
```

**验证标准**:
- ✅ 所有模型文件存在
- ✅ 所有测试数据存在
- ✅ 环境变量正确配置
- ✅ 至少10个测试程序编译成功
- ✅ 健康检查通过

**输出**:
- `test_reports/phase0_verification.txt` 验证报告
- 环境就绪标志

---

## ✅ 验收标准

### 必须完成

- [ ] Qwen3-0.6B 模型完整性验证（包含 tokenizer.json、config.json、weights）
- [ ] 5个测试数据文件生成
- [ ] 环境变量正确配置
- [ ] 至少16个测试程序编译成功
- [ ] 环境验证通过

### 质量检查

- [ ] 模型文件完整性校验
- [ ] 测试数据 JSON 格式正确
- [ ] 所有测试程序可执行
- [ ] 磁盘空间充足（> 10GB）

---

## 📊 执行报告

**执行时间**: ________

**完成情况**:
- P0.1: ☐ 完成 / ☐ 失败 (模型验证)
- P0.2: ☐ 完成 / ☐ 失败
- P0.3: ☐ 完成 / ☐ 失败
- P0.4: ☐ 完成 / ☐ 失败
- P0.5: ☐ 完成 / ☐ 失败

**总体状态**: ☐ 成功 / ☐ 部分成功 / ☐ 失败

**问题记录**:
```
（记录遇到的问题和解决方案）
```

---

## 🔄 下一步

Phase 0 完成后，创建完成标志并通知 Agent-1 启动 Phase 1:

```bash
# 创建完成标志
touch /tmp/cllm_test_locks/phase0.done

# 生成交接报告
cat > test_reports/phase0_handoff.txt << EOF
Phase 0 完成
完成时间: $(date)
模型路径: ${CLLM_TEST_MODEL_PATH}
数据路径: ${CLLM_TEST_DATA_PATH}
测试程序数: 16个
状态: 就绪 ✅

Agent-1 可以开始执行 Phase 1 单元测试
EOF

echo "✅ Phase 0 完成，Agent-1 可以启动"
```

---

**Agent-0 准备阶段执行计划完成**
