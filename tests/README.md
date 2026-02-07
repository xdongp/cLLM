# cLLM 测试套件

## 概述

cLLM 测试套件采用模块化、分层的结构设计，提供完整的测试覆盖，包括单元测试、集成测试、性能测试和压力测试。

## 目录结构

```
tests/
├── utils/                          # 测试工具库（公共测试基类和辅助函数）
│   ├── test_base.h                # 测试基类（TestBase、SchedulerTestBase等）
│   ├── mock_tokenizer.h           # Mock Tokenizer实现
│   ├── http_test_helpers.h        # HTTP测试辅助工具
│   ├── performance_test_helpers.h # 性能测试辅助工具
│   └── test_data_helpers.h        # 测试数据生成工具
│
├── unit/                           # 单元测试
│   ├── tokenizer/                 # Tokenizer相关单元测试
│   │   └── tokenizer_interface_test.cpp
│   ├── scheduler/                 # 调度器相关单元测试
│   ├── cache/                     # 缓存相关单元测试
│   ├── backend/                   # 后端相关单元测试
│   └── http/                      # HTTP相关单元测试
│
├── integration/                    # 集成测试
│   ├── api/                       # API集成测试
│   │   └── generate_api_integration_test.cpp
│   ├── pipeline/                  # 流水线集成测试
│   ├── backend/                   # 后端集成测试
│   └── system/                    # 系统集成测试
│
├── performance/                    # 性能测试
│   ├── benchmark_inference.cpp    # 推理性能基准测试
│   ├── benchmark_tokenizer.cpp    # Tokenizer性能基准测试
│   └── ...
│
├── stress/                        # 压力测试
│   ├── concurrency_stress_test.cpp # 并发压力测试
│   └── ...
│
├── data/                          # 测试数据
│   ├── inference_test_data.json
│   ├── performance_test_data.json
│   └── ...
│
├── kylin_test_suite/              # Kylin专用测试套件
│
├── CMakeLists.txt                 # CMake配置
├── CMakeLists.txt.new            # 新的CMake配置模板
├── README.md                      # 本文件
├── TEST_GUIDE.md                  # 测试编写指南
├── TEST_REFACTORING_PLAN.md      # 测试重构方案
└── MIGRATION_GUIDE.md             # 测试迁移指南
```

## 快速开始

### 编译测试

```bash
cd build
cmake ..
make -j$(nproc)
```

### 运行所有测试

```bash
ctest --verbose
```

### 运行特定类别的测试

```bash
# 单元测试
ctest -R "unit_.*" --verbose

# 集成测试
ctest -R "integration_.*" --verbose

# 性能测试
ctest -R "benchmark_.*" --verbose

# 压力测试
ctest -R "stress_.*" --verbose
```

### 运行单个测试

```bash
./bin/tests/unit_tokenizer_interface_test
./bin/tests/integration_generate_api_test
```

## 测试分类

### 单元测试 (Unit Tests)
- **位置**：`tests/unit/`
- **特点**：快速、隔离、使用Mock对象
- **用途**：测试单个类或函数的功能
- **执行时间**：< 1秒

### 集成测试 (Integration Tests)
- **位置**：`tests/integration/`
- **特点**：测试真实交互、可能涉及I/O
- **用途**：测试多个模块的协作
- **执行时间**：1-10秒

### 性能测试 (Performance Tests)
- **位置**：`tests/performance/`
- **特点**：测量性能指标、生成报告
- **用途**：性能回归检测、性能优化验证
- **执行时间**：10-60秒

### 压力测试 (Stress Tests)
- **位置**：`tests/stress/`
- **特点**：测试极限情况、长时间运行
- **用途**：验证系统稳定性、检测内存泄漏
- **执行时间**：1-30分钟

## 测试工具库

### 1. TestBase - 测试基类
提供通用的测试环境设置和清理功能。

```cpp
#include "utils/test_base.h"

class MyTest : public cllm::test::TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        // 自动创建临时目录
    }
};
```

### 2. MockTokenizer - Mock Tokenizer
提供完整的Mock Tokenizer实现，无需真实模型。

```cpp
#include "utils/mock_tokenizer.h"

auto tokenizer = std::make_unique<cllm::test::MockTokenizer>();
auto tokens = tokenizer->encode("Hello", true);
```

### 3. HttpTestHelpers - HTTP测试工具
提供HTTP请求创建和响应验证的便捷方法。

```cpp
#include "utils/http_test_helpers.h"

auto request = HttpTestHelpers::createGenerateRequest("Hello", 10);
auto jsonResponse = HttpTestHelpers::verifySuccessResponse(response);
```

### 4. PerformanceTestHelpers - 性能测试工具
提供性能测量和基准测试功能。

```cpp
#include "utils/performance_test_helpers.h"

auto stats = PerformanceTestHelpers::benchmark(testFunc, 1000, "Test");
stats.print("Test");
```

### 5. TestDataHelpers - 测试数据生成工具
提供各种测试数据的生成方法。

```cpp
#include "utils/test_data_helpers.h"

auto prompts = TestDataHelpers::generateTestPrompts();
auto randomText = TestDataHelpers::generateRandomString(100);
```

## 编写新测试

### 步骤1：选择测试类型和位置

根据测试目的选择合适的目录：
- 单元测试 → `tests/unit/<模块>/`
- 集成测试 → `tests/integration/<分类>/`
- 性能测试 → `tests/performance/`
- 压力测试 → `tests/stress/`

### 步骤2：创建测试文件

遵循命名规范：
- 单元测试：`<模块>_test.cpp`
- 集成测试：`<功能>_integration_test.cpp`
- 性能测试：`benchmark_<功能>.cpp`
- 压力测试：`<类型>_stress_test.cpp`

### 步骤3：编写测试代码

使用测试工具库简化测试编写：

```cpp
#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "utils/mock_tokenizer.h"

using namespace cllm;
using namespace cllm::test;

class MyTest : public TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        tokenizer_ = std::make_unique<MockTokenizer>();
    }
    
    std::unique_ptr<MockTokenizer> tokenizer_;
};

TEST_F(MyTest, TestName_Condition_ExpectedResult) {
    // Arrange - 准备
    std::string text = "Hello";
    
    // Act - 执行
    auto tokens = tokenizer_->encode(text, true);
    
    // Assert - 验证
    EXPECT_FALSE(tokens.empty());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 步骤4：更新CMakeLists.txt

```cmake
add_cllm_test(my_test tests/unit/my_module/my_test.cpp)
```

### 步骤5：编译和运行

```bash
cd build
make my_test
./bin/tests/my_test
```

## 示例测试

### 示例1：单元测试
查看 `tests/unit/tokenizer/tokenizer_interface_test.cpp`

### 示例2：集成测试
查看 `tests/integration/api/generate_api_integration_test.cpp`

## 测试最佳实践

1. **使用AAA模式**：Arrange-Act-Assert
2. **描述性命名**：`TestClass_Condition_ExpectedResult`
3. **独立测试**：每个测试应独立运行
4. **使用Mock对象**：单元测试隔离依赖
5. **适当的断言**：使用正确的EXPECT/ASSERT宏
6. **文档注释**：为复杂测试添加注释
7. **测试覆盖**：覆盖正常、边界和错误情况

## 文档资源

- **[TEST_GUIDE.md](TEST_GUIDE.md)** - 详细的测试编写指南
- **[TEST_REFACTORING_PLAN.md](TEST_REFACTORING_PLAN.md)** - 测试重构方案和旧测试映射表
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - 从旧测试迁移到新测试的指南

## 测试覆盖率

查看测试覆盖率：

```bash
# 使用gcov/lcov生成覆盖率报告
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON ..
make
make test
make coverage
```

## 持续集成

测试在CI/CD流程中自动运行：
- **Pull Request**：运行单元测试和集成测试
- **合并到主分支**：运行完整测试套件
- **夜间构建**：运行压力测试和性能测试

## 贡献指南

添加新测试时：
1. 遵循测试分类和命名规范
2. 使用公共测试工具库
3. 添加必要的文档注释
4. 确保测试通过
5. 更新CMakeLists.txt
6. 提交代码审查

## 故障排查

### 编译失败
- 检查头文件路径
- 确保包含了必要的依赖

### 测试失败
- 查看详细错误信息
- 检查SetUp/TearDown逻辑
- 验证测试数据

### 性能问题
- 使用性能分析工具（valgrind、perf）
- 检查是否有资源泄漏

## 联系方式

如有问题或建议：
- 查看文档
- 参考示例测试
- 联系测试负责人

## 更新日志

### 2024-02-05
- ✅ 创建新的测试工具库
- ✅ 添加MockTokenizer和测试辅助工具
- ✅ 创建示例单元测试和集成测试
- ✅ 编写完整的测试文档
- 📝 开始测试迁移工作

## 许可证

与cLLM项目相同的许可证。
