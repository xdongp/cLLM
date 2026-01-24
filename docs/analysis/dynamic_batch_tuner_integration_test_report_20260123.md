## 动态 Batch Size 调谐器集成测试报告 (2026-01-23)

### 目标
- 完成调谐器与调度器集成后，验证构建与配置校验流程正常。

### 变更范围
- 接入动态调谐器：`Scheduler`、`BatchManager`、`Config` 与构建系统。
- 新增调谐器头文件：`include/cllm/scheduler/dynamic_batch_tuner.h`。
- 更新配置示例：`config/config.yaml`、`config/config_gpu.yaml`。

### 测试环境
- OS: macOS 15.x (darwin 24.5.0)
- 构建系统: CMake

### 测试项与结果
1. **构建**
   - 命令:
     - `cmake -S . -B build`
     - `cmake --build build -j4`
   - 结果: ✅ 通过
   - 备注: 链接阶段存在 tokenizers-cpp 静态库版本警告（15.5 > 15.0），未影响构建产物生成。

2. **配置校验**
   - 命令:
     - `python3 scripts/validate_configs.py`
   - 结果: ✅ 通过（含 5 条告警）
   - 告警摘要:
     - scheduler_config.yaml 内存在冗余配置项（default_temperature / default_top_k / default_top_p）
     - scheduler 与 server 同时定义 max_batch_size
     - server_config.yaml host=0.0.0.0 安全提示

3. **运行时基准对比（吞吐/延迟）**
   - 静态基线（`config/config.yaml`，端口 `8080`）
     - 命令:
       - `python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8080 --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output-file /tmp/cllm_static_20260123.json`
     - 结果（关键指标）:
       - 平均响应: `3.45s`
       - 平均吞吐: `114.07 tokens/s`
       - 平均单请求 tokens/s: `16.18`
       - 最大响应: `6.28s`
       - 总耗时: `31.56s`

   - 动态/混合（`config/config_llama_cpp.yaml`，端口 `8081`）
     - 命令:
       - `python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8081 --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output-file /tmp/cllm_dynamic_20260123.json`
     - 结果（关键指标）:
       - 平均响应: `3.86s`
       - 平均吞吐: `101.74 tokens/s`
       - 平均单请求 tokens/s: `14.15`
       - 最大响应: `7.18s`
       - 总耗时: `35.38s`

   - 对比结论（动态/混合 vs 静态）:
     - 吞吐下降约 `10.8%`
     - 平均响应变慢约 `11.9%`
     - 当前配置下混合策略未优于静态基线

4. **运行时基准对比（第二轮动态策略）**
   - 动态策略 v2（`config/config_llama_cpp.yaml`，端口 `8082`）
     - 命令:
       - `python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8082 --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output-file /tmp/cllm_dynamic_20260123_v2.json`
     - 结果（关键指标）:
       - 平均响应: `3.92s`
       - 平均吞吐: `100.44 tokens/s`
       - 平均单请求 tokens/s: `14.10`
       - 最大响应: `7.02s`
       - 总耗时: `35.84s`

   - 对比结论（动态 v2 vs 静态）:
     - 吞吐下降约 `12.0%`
     - 平均响应变慢约 `13.7%`
     - 当前配置下动态策略仍未超过静态基线

5. **运行时基准对比（第三轮混合策略）**
   - 混合策略 v3（`config/config_llama_cpp.yaml`，端口 `8083`）
     - 命令:
       - `python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8083 --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output-file /tmp/cllm_hybird_20260123_v3.json`
     - 结果（关键指标）:
       - 平均响应: `4.40s`
       - 平均吞吐: `89.67 tokens/s`
       - 平均单请求 tokens/s: `12.73`
       - 最大响应: `8.63s`
       - 总耗时: `40.15s`

   - 对比结论（混合 v3 vs 静态）:
     - 吞吐下降约 `21.4%`
     - 平均响应变慢约 `27.4%`
     - 当前配置下混合策略明显劣于静态基线

6. **运行时基准对比（固定 batch_size=64）**
   - 固定 batch size（`config/config_llama_cpp.yaml`，端口 `8084`）
     - 命令:
       - `python3 tools/unified_benchmark.py --server-type cllm --server-url http://localhost:8084 --test-type api-concurrent --requests 72 --concurrency 8 --max-tokens 50 --output-file /tmp/cllm_static_fixed64_20260123.json`
     - 结果（关键指标）:
       - 平均响应: `4.94s`
       - 平均吞吐: `79.65 tokens/s`
       - 平均单请求 tokens/s: `11.39`
       - 最大响应: `11.82s`
       - 总耗时: `45.20s`

   - 对比结论（fixed=64 vs 静态）:
     - 吞吐下降约 `30.2%`
     - 平均响应变慢约 `43.1%`
     - 固定 batch_size=64 明显不适合当前负载

### 结论
- 集成代码已成功编译通过，配置校验正常运行。
- 运行时对比显示：当前动态/混合配置与第二轮动态策略均未超过静态基线，需要继续调参或策略优化。
  - 建议方向：继续使用现有静态基线（batch size 较小），固定大 batch size 对当前负载不友好。

### 后续建议
- 若需严格消除告警，可统一整理 scheduler/server 配置并设置更安全的 host。
