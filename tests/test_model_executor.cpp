#include <gtest/gtest.h>
#include <cllm/model/executor.h>
#include <fstream>
#include <cstdio>

using namespace cllm;

class ModelExecutorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试模型文件
        createTestModelFile();
    }

    void TearDown() override {
        // 清理测试文件
        std::remove("test_model_fp32.bin");
        std::remove("test_model_int8.bin");
        std::remove("test_model_int4.bin");
    }

    void createTestModelFile() {
        // 创建简单的FP32测试模型文件
        const size_t modelSize = 1024;
        float* weights = new float[modelSize];
        for (size_t i = 0; i < modelSize; ++i) {
            weights[i] = static_cast<float>(i) / modelSize;
        }
        
        std::ofstream fp32File("test_model_fp32.bin", std::ios::binary);
        fp32File.write(reinterpret_cast<char*>(weights), modelSize * sizeof(float));
        fp32File.close();
        
        // 创建简单的INT8测试模型文件
        int8_t* int8Weights = new int8_t[modelSize];
        for (size_t i = 0; i < modelSize; ++i) {
            int8Weights[i] = static_cast<int8_t>(i % 256 - 128);
        }
        
        std::ofstream int8File("test_model_int8.bin", std::ios::binary);
        int8File.write(reinterpret_cast<char*>(int8Weights), modelSize);
        int8File.close();
        
        // 创建简单的INT4测试模型文件
        uint8_t* int4Weights = new uint8_t[modelSize / 2];
        for (size_t i = 0; i < modelSize / 2; ++i) {
            int4Weights[i] = (i % 16) | ((i % 16) << 4);
        }
        
        std::ofstream int4File("test_model_int4.bin", std::ios::binary);
        int4File.write(reinterpret_cast<char*>(int4Weights), modelSize / 2);
        int4File.close();
        
        delete[] weights;
        delete[] int8Weights;
        delete[] int4Weights;
    }
};

TEST_F(ModelExecutorTest, FP32ModelLoading) {
    EXPECT_NO_THROW({
        ModelExecutor executor("test_model_fp32.bin");
        executor.loadModel();
        EXPECT_TRUE(executor.isLoaded());
        executor.unloadModel();
    });
}

TEST_F(ModelExecutorTest, Int8ModelLoading) {
    EXPECT_NO_THROW({
        ModelExecutor executor("test_model_int8.bin", "int8");
        executor.loadModel();
        EXPECT_TRUE(executor.isLoaded());
        executor.unloadModel();
    });
}

TEST_F(ModelExecutorTest, Int4ModelLoading) {
    EXPECT_NO_THROW({
        ModelExecutor executor("test_model_int4.bin", "int4");
        executor.loadModel();
        EXPECT_TRUE(executor.isLoaded());
        executor.unloadModel();
    });
}

TEST_F(ModelExecutorTest, UnloadModelSafety) {
    ModelExecutor executor("test_model_fp32.bin");
    executor.loadModel();
    EXPECT_TRUE(executor.isLoaded());
    
    // 多次卸载应该安全
    executor.unloadModel();
    executor.unloadModel();
    EXPECT_FALSE(executor.isLoaded());
}

TEST_F(ModelExecutorTest, DoubleFreePrevention) {
    ModelExecutor executor("test_model_int8.bin", "int8");
    executor.loadModel();
    
    // 测试量化模型的内存安全
    executor.unloadModel();
    executor.unloadModel();  // 不应该崩溃
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}