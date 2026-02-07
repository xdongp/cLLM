/**
 * @file test_softmax_comparison.cpp
 * @brief CPU vs GPU Softmax 对比测试
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"

// 注意：HFTransformerModel 的完整定义在 kylin_test_main.cpp 中包含

namespace kylin_test {

// ============================================================================
// 测试：Softmax 数值精度对比
// ============================================================================
class SoftmaxComparisonTest : public TestCase {
public:
    SoftmaxComparisonTest() : TestCase("softmax_comparison", 
        "对比 CPU 和 GPU 的 softmax 实现") {}
    
    void execute() override;
};

// 注册测试
inline void registerSoftmaxComparisonTests(TestSuite& suite) {
    suite.addTest(std::make_shared<SoftmaxComparisonTest>());
}

} // namespace kylin_test
