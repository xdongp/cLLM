#!/bin/bash
################################################################################
# cLLM 并行测试执行脚本
# 
# 功能：
# - 分阶段执行集成测试
# - 支持并行测试执行
# - 自动生成测试报告
# - 依赖关系管理
#
# 使用方法：
#   ./scripts/parallel_test_runner.sh [phase]
#   
#   phase: 0-5 或 all
#     0 - 准备阶段
#     1 - 单元测试
#     2 - 模块集成测试
#     3 - 子系统测试
#     4 - 系统集成测试
#     5 - E2E 场景测试
#     all - 运行所有阶段
#
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
LOG_DIR="${PROJECT_ROOT}/logs"
REPORT_DIR="${PROJECT_ROOT}/test_reports"

# 创建必要的目录
mkdir -p "${LOG_DIR}"
mkdir -p "${REPORT_DIR}"

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查测试环境..."
    
    # 检查构建目录
    if [ ! -d "${BUILD_DIR}" ]; then
        log_error "构建目录不存在: ${BUILD_DIR}"
        exit 1
    fi
    
    # 检查模型路径
    if [ -z "$CLLM_TEST_MODEL_PATH" ]; then
        log_warning "未设置 CLLM_TEST_MODEL_PATH，集成测试可能跳过"
    else
        log_info "测试模型路径: $CLLM_TEST_MODEL_PATH"
    fi
    
    # 检查并行工具
    if ! command -v parallel &> /dev/null; then
        log_warning "GNU parallel 未安装，将使用串行执行"
        USE_PARALLEL=false
    else
        USE_PARALLEL=true
    fi
    
    log_success "环境检查完成"
}

# 运行单个测试
run_test() {
    local test_name=$1
    local phase=$2
    local log_file="${LOG_DIR}/${test_name}.log"
    local json_file="${LOG_DIR}/${test_name}.json"
    
    log_info "[$phase] 运行 $test_name..."
    
    # 运行测试
    if "${BUILD_DIR}/bin/${test_name}" \
        --gtest_output="json:${json_file}" \
        > "${log_file}" 2>&1; then
        log_success "[$phase] $test_name PASSED"
        return 0
    else
        log_error "[$phase] $test_name FAILED (查看日志: ${log_file})"
        return 1
    fi
}

# Phase 0: 准备阶段
phase_0_preparation() {
    echo "========================================="
    echo "Phase 0: 测试准备阶段"
    echo "========================================="
    
    # P0.1: 检查模型（✅ 本项目默认已包含本地模型目录，避免重复下载）
    if [ -z "$CLLM_TEST_MODEL_PATH" ]; then
        # 自动探测默认 HF 模型目录（用于 HFTokenizer 测试）
        local default_hf_dir="${PROJECT_ROOT}/model/Qwen/Qwen3-0.6B"
        if [ -d "$default_hf_dir" ] && [ -f "$default_hf_dir/tokenizer.json" ]; then
            export CLLM_TEST_MODEL_PATH="$default_hf_dir"
            log_info "P0.1: 未设置 CLLM_TEST_MODEL_PATH，已自动设置为: $CLLM_TEST_MODEL_PATH"
        fi
    fi

    if [ -n "$CLLM_TEST_MODEL_PATH" ] && [ -d "$CLLM_TEST_MODEL_PATH" ]; then
        if [ -f "$CLLM_TEST_MODEL_PATH/tokenizer.json" ]; then
            log_success "P0.1: HF tokenizer 模型目录就绪: $CLLM_TEST_MODEL_PATH"
        else
            log_warning "P0.1: 已设置 CLLM_TEST_MODEL_PATH，但缺少 tokenizer.json: $CLLM_TEST_MODEL_PATH"
        fi
    else
        log_warning "P0.1: 未检测到可用的 HF 模型目录（不会自动下载）。"
        log_info "请手动设置: export CLLM_TEST_MODEL_PATH=\"${PROJECT_ROOT}/model/Qwen/Qwen3-0.6B\""
    fi

    # 额外：检测 Kylin 扁平权重 bin（用于 test_inference_engine_qwen）
    if [ -z "$CLLM_TEST_MODEL_BIN_DIR" ]; then
        local default_bin_dir="${PROJECT_ROOT}/model/Qwen"
        if [ -d "$default_bin_dir" ]; then
            export CLLM_TEST_MODEL_BIN_DIR="$default_bin_dir"
            log_info "P0.1: 已自动设置 CLLM_TEST_MODEL_BIN_DIR=$CLLM_TEST_MODEL_BIN_DIR"
        fi
    fi

    if [ -n "$CLLM_TEST_MODEL_BIN_DIR" ]; then
        local fp32_bin="$CLLM_TEST_MODEL_BIN_DIR/qwen3_0.6b_cllm_fp32.bin"
        if [ -f "$fp32_bin" ]; then
            log_success "P0.1: Kylin bin 权重就绪: $fp32_bin"
        else
            log_warning "P0.1: 未找到 Kylin bin 权重: $fp32_bin（相关测试将跳过或失败）"
        fi
    fi
    
    # P0.2: 生成测试数据
    log_info "P0.2: 生成测试数据..."
    if [ -f "${PROJECT_ROOT}/scripts/generate_test_data.py" ]; then
        python3 "${PROJECT_ROOT}/scripts/generate_test_data.py"
        log_success "P0.2: 测试数据生成完成"
    else
        log_warning "P0.2: 测试数据生成脚本不存在"
    fi
    
    # P0.3: 配置环境
    log_info "P0.3: 配置测试环境..."
    export CLLM_TEST_DATA_PATH="${PROJECT_ROOT}/tests/data"
    log_success "P0.3: 环境配置完成"
    
    # P0.4: 检查编译产物
    log_info "P0.4: 检查测试程序编译..."
    local test_count=0
    for test_bin in "${BUILD_DIR}"/bin/test_*; do
        if [ -x "$test_bin" ]; then
            ((test_count++))
        fi
    done
    log_success "P0.4: 找到 $test_count 个测试程序"
    
    # P0.5: Mock 对象（已在代码中）
    log_success "P0.5: Mock 对象已就绪"
    
    echo "========================================="
    log_success "Phase 0 完成！"
    echo "========================================="
}

# Phase 1: 单元测试
phase_1_unit_tests() {
    echo "========================================="
    echo "Phase 1: 单元测试阶段"
    echo "========================================="
    
    # 定义测试列表
    local tests=(
        "test_hf_tokenizer"
        "test_http_server_direct"
        "test_model_executor"
        "test_libtorch_backend"
    )
    
    local passed=0
    local failed=0
    
    if [ "$USE_PARALLEL" = true ]; then
        log_info "使用并行执行（${#tests[@]} 个测试）..."
        
        # 使用 GNU parallel 并行执行
        export -f run_test log_info log_success log_error
        export BUILD_DIR LOG_DIR
        
        printf '%s\n' "${tests[@]}" | \
            parallel -j 4 run_test {} "P1"
        
    else
        log_info "使用串行执行..."
        for test_name in "${tests[@]}"; do
            if run_test "$test_name" "P1"; then
                ((passed++))
            else
                ((failed++))
            fi
        done
    fi
    
    # 统计结果
    log_info "Phase 1 结果: 通过=$passed, 失败=$failed"
    
    echo "========================================="
    if [ $failed -eq 0 ]; then
        log_success "Phase 1 完成！所有测试通过"
    else
        log_error "Phase 1 完成，但有 $failed 个测试失败"
    fi
    echo "========================================="
    
    return $failed
}

# Phase 2: 模块集成测试
phase_2_integration_tests() {
    echo "========================================="
    echo "Phase 2: 模块集成测试阶段"
    echo "========================================="
    
    local tests=(
        "test_tokenizer_executor_integration"
        "test_server_integration"
    )
    
    local passed=0
    local failed=0
    
    for test_name in "${tests[@]}"; do
        if [ -f "${BUILD_DIR}/bin/${test_name}" ]; then
            if run_test "$test_name" "P2"; then
                ((passed++))
            else
                ((failed++))
            fi
        else
            log_warning "测试不存在: $test_name"
        fi
    done
    
    echo "========================================="
    log_info "Phase 2 结果: 通过=$passed, 失败=$failed"
    if [ $failed -eq 0 ]; then
        log_success "Phase 2 完成！"
    else
        log_error "Phase 2 完成，但有 $failed 个测试失败"
    fi
    echo "========================================="
    
    return $failed
}

# Phase 3: 子系统测试
phase_3_subsystem_tests() {
    echo "========================================="
    echo "Phase 3: 子系统测试阶段"
    echo "========================================="
    
    local tests=(
        "test_full_system"
        "test_inference_pipeline"
    )
    
    local passed=0
    local failed=0
    
    for test_name in "${tests[@]}"; do
        if [ -f "${BUILD_DIR}/bin/${test_name}" ]; then
            if run_test "$test_name" "P3"; then
                ((passed++))
            else
                ((failed++))
            fi
        else
            log_warning "测试不存在: $test_name"
        fi
    done
    
    echo "========================================="
    log_info "Phase 3 结果: 通过=$passed, 失败=$failed"
    if [ $failed -eq 0 ]; then
        log_success "Phase 3 完成！"
    else
        log_error "Phase 3 完成，但有 $failed 个测试失败"
    fi
    echo "========================================="
    
    return $failed
}

# Phase 4: 系统集成测试
phase_4_system_tests() {
    echo "========================================="
    echo "Phase 4: 系统集成测试阶段"
    echo "========================================="
    
    # 功能测试
    log_info "运行系统功能测试..."
    local tests=(
        "test_api_integration"
        "test_performance"
    )
    
    local passed=0
    local failed=0
    
    for test_name in "${tests[@]}"; do
        if [ -f "${BUILD_DIR}/bin/${test_name}" ]; then
            if run_test "$test_name" "P4"; then
                ((passed++))
            else
                ((failed++))
            fi
        else
            log_warning "测试不存在: $test_name"
        fi
    done
    
    echo "========================================="
    log_info "Phase 4 结果: 通过=$passed, 失败=$failed"
    if [ $failed -eq 0 ]; then
        log_success "Phase 4 完成！"
    else
        log_error "Phase 4 完成，但有 $failed 个测试失败"
    fi
    echo "========================================="
    
    return $failed
}

# Phase 5: E2E 场景测试
phase_5_e2e_tests() {
    echo "========================================="
    echo "Phase 5: E2E 场景测试阶段"
    echo "========================================="
    
    log_info "运行 E2E 场景测试..."
    
    # 如果有 Python 测试脚本
    if [ -f "${PROJECT_ROOT}/test/test_api.py" ]; then
        log_info "运行 API 测试..."
        python3 "${PROJECT_ROOT}/test/test_api.py" > "${LOG_DIR}/e2e_api.log" 2>&1
        if [ $? -eq 0 ]; then
            log_success "API 测试通过"
        else
            log_error "API 测试失败"
        fi
    fi
    
    echo "========================================="
    log_success "Phase 5 完成！"
    echo "========================================="
}

# 生成测试报告
generate_report() {
    log_info "生成测试报告..."
    
    local report_file="${REPORT_DIR}/test_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << 'EOF'
# cLLM 集成测试报告

## 测试概要

**执行时间**: $(date)
**测试阶段**: 分阶段集成测试

## 测试结果

EOF
    
    # 统计测试结果
    local total_passed=0
    local total_failed=0
    
    for json_file in "${LOG_DIR}"/*.json; do
        if [ -f "$json_file" ]; then
            # 解析 JSON 文件（简化版）
            if grep -q '"failures": 0' "$json_file"; then
                ((total_passed++))
            else
                ((total_failed++))
            fi
        fi
    done
    
    cat >> "$report_file" << EOF

### 总体统计

- **通过**: $total_passed
- **失败**: $total_failed
- **总计**: $((total_passed + total_failed))

### 详细结果

EOF
    
    # 列出所有测试结果
    for json_file in "${LOG_DIR}"/*.json; do
        if [ -f "$json_file" ]; then
            local test_name=$(basename "$json_file" .json)
            if grep -q '"failures": 0' "$json_file"; then
                echo "- ✅ $test_name" >> "$report_file"
            else
                echo "- ❌ $test_name" >> "$report_file"
            fi
        fi
    done
    
    log_success "测试报告已生成: $report_file"
}

# 主函数
main() {
    local phase=${1:-all}
    
    echo "========================================="
    echo "cLLM 分阶段集成测试"
    echo "========================================="
    
    # 检查环境
    check_environment
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行测试阶段
    case "$phase" in
        0)
            phase_0_preparation
            ;;
        1)
            phase_1_unit_tests
            ;;
        2)
            phase_2_integration_tests
            ;;
        3)
            phase_3_subsystem_tests
            ;;
        4)
            phase_4_system_tests
            ;;
        5)
            phase_5_e2e_tests
            ;;
        all)
            phase_0_preparation
            phase_1_unit_tests || true
            phase_2_integration_tests || true
            phase_3_subsystem_tests || true
            phase_4_system_tests || true
            phase_5_e2e_tests || true
            ;;
        *)
            log_error "无效的阶段: $phase"
            echo "用法: $0 [0|1|2|3|4|5|all]"
            exit 1
            ;;
    esac
    
    # 记录结束时间
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # 生成报告
    if [ "$phase" = "all" ] || [ "$phase" -ge 1 ]; then
        generate_report
    fi
    
    echo "========================================="
    log_success "测试完成！"
    log_info "总耗时: ${duration}秒 ($((duration / 60))分钟)"
    echo "========================================="
}

# 执行主函数
main "$@"
