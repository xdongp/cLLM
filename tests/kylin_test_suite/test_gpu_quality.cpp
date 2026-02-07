/**
 * @file test_gpu_quality.cpp
 * @brief 测试 GPU 生成质量
 *
 * 验证 GPU 生成的文本是否正确/合理
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"

namespace kylin_test {

// ============================================================================
// 测试：GPU 生成质量验证
// ============================================================================
class GPUQualityTest : public TestCase {
public:
    GPUQualityTest() : TestCase("gpu_quality_check", 
        "验证 GPU 生成的文本是否正确合理") {}
    
    void execute() override;
};

// 注册测试
inline void registerGPUQualityTests(TestSuite& suite) {
    suite.addTest(std::make_shared<GPUQualityTest>());
}

} // namespace kylin_test
