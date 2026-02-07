#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <filesystem>
#include <chrono>

namespace cllm {
namespace test {

/**
 * @brief 测试基类
 * 提供测试所需的通用功能和资源管理
 */
class TestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时测试目录
        tempTestDir_ = createTempTestDirectory();
    }
    
    void TearDown() override {
        // 清理临时测试目录
        cleanupTempDirectory();
    }
    
    /**
     * @brief 创建临时测试目录
     * @return 临时目录路径
     */
    std::filesystem::path createTempTestDirectory() {
        auto tempDir = std::filesystem::temp_directory_path() / "cllm_test" / 
                       std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(tempDir);
        return tempDir;
    }
    
    /**
     * @brief 清理临时目录
     */
    void cleanupTempDirectory() {
        if (std::filesystem::exists(tempTestDir_)) {
            std::filesystem::remove_all(tempTestDir_);
        }
    }
    
    /**
     * @brief 获取临时测试目录路径
     */
    std::filesystem::path getTempTestDir() const {
        return tempTestDir_;
    }
    
    /**
     * @brief 解析配置文件路径
     * @param configFile 配置文件名
     * @return 配置文件绝对路径
     */
    std::string resolveConfigPath(const std::string& configFile) {
        const std::vector<std::string> candidates = {
            "config/" + configFile,
            "../config/" + configFile,
            "../../config/" + configFile
        };
        
        for (const auto& path : candidates) {
            if (std::filesystem::exists(path)) {
                return std::filesystem::absolute(path).string();
            }
        }
        return "config/" + configFile;
    }
    
private:
    std::filesystem::path tempTestDir_;
};

// 注意：SchedulerTestBase 和 HttpEndpointTestBase 需要包含额外的头文件
// 如果你需要使用它们，请在你的测试文件中自己定义或包含：
// #include <cllm/scheduler/scheduler.h>
// #include <cllm/http/generate_endpoint.h>
// #include <cllm/tokenizer/i_tokenizer.h>
//
// class MySchedulerTest : public TestBase {
// protected:
//     void SetUp() override {
//         TestBase::SetUp();
//         scheduler_ = new Scheduler(...);
//         scheduler_->start();
//     }
//     
//     void TearDown() override {
//         if (scheduler_) {
//             scheduler_->stop();
//             delete scheduler_;
//         }
//         TestBase::TearDown();
//     }
//     
//     Scheduler* scheduler_;
// };

} // namespace test
} // namespace cllm
