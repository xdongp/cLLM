#include <gtest/gtest.h>
#include <cllm/sampler.h>
#include <cllm/memory/float_array.h>
#include <cllm/common/config.h>
#include <vector>
#include <algorithm>

using namespace cllm;

class SamplerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("config/sampler_config.yaml");
        sampler_ = std::make_unique<Sampler>();
    }

    void TearDown() override {
        sampler_.reset();
    }

    FloatArray createLogits(const std::vector<float>& values) {
        FloatArray logits(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            logits[i] = values[i];
        }
        return logits;
    }

    std::unique_ptr<Sampler> sampler_;
};

TEST_F(SamplerTest, Constructor) {
    EXPECT_NE(sampler_, nullptr);
}

TEST_F(SamplerTest, SampleGreedySingleMax) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleGreedyMultipleMax) {
    std::vector<float> values = {1.0f, 3.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_TRUE(sampled == 1 || sampled == 2);
}

TEST_F(SamplerTest, SampleGreedyAllSame) {
    std::vector<float> values = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 5);
}

TEST_F(SamplerTest, SampleGreedyNegativeValues) {
    std::vector<float> values = {-5.0f, -3.0f, -1.0f, -2.0f, -4.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleTemperatureLow) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 0.1f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleTemperatureHigh) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 10.0f);
    
    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 5);
}

TEST_F(SamplerTest, SampleTemperatureZero) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 0.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleTemperatureOne) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleEmptyLogits) {
    std::vector<float> values;
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 0);
}

TEST_F(SamplerTest, SampleSingleElement) {
    std::vector<float> values = {5.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 0);
}

TEST_F(SamplerTest, SampleTwoElements) {
    std::vector<float> values = {1.0f, 2.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 1);
}

TEST_F(SamplerTest, SampleLargeVocabulary) {
    std::vector<float> values(1000, 0.0f);
    values[500] = 10.0f;
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 500);
}

TEST_F(SamplerTest, SampleTemperatureDistribution) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    std::vector<int> counts(5, 0);
    const int numSamples = 1000;
    
    for (int i = 0; i < numSamples; ++i) {
        int sampled = sampler_->sample(logits, 2.0f);
        if (sampled >= 0 && sampled < 5) {
            counts[sampled]++;
        }
    }
    
    EXPECT_GT(counts[2], 0);
}

TEST_F(SamplerTest, SampleVeryLowTemperature) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 0.01f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleVeryHighTemperature) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 100.0f);
    
    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 5);
}

TEST_F(SamplerTest, SampleUniformDistribution) {
    std::vector<float> values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    FloatArray logits = createLogits(values);
    
    std::vector<int> counts(5, 0);
    const int numSamples = 1000;
    
    for (int i = 0; i < numSamples; ++i) {
        int sampled = sampler_->sample(logits, 1.0f);
        if (sampled >= 0 && sampled < 5) {
            counts[sampled]++;
        }
    }
    
    for (int count : counts) {
        EXPECT_GT(count, 0);
    }
}

TEST_F(SamplerTest, SampleSkewedDistribution) {
    std::vector<float> values = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f};
    FloatArray logits = createLogits(values);
    
    std::vector<int> counts(5, 0);
    const int numSamples = 1000;
    
    for (int i = 0; i < numSamples; ++i) {
        int sampled = sampler_->sample(logits, 1.0f);
        if (sampled >= 0 && sampled < 5) {
            counts[sampled]++;
        }
    }
    
    EXPECT_GT(counts[2], counts[0]);
    EXPECT_GT(counts[2], counts[4]);
}

TEST_F(SamplerTest, SampleExtremeValues) {
    std::vector<float> values = {-1000.0f, -500.0f, 0.0f, 500.0f, 1000.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 4);
}

TEST_F(SamplerTest, SampleMixedPositiveNegative) {
    std::vector<float> values = {-1.0f, 0.0f, 1.0f, 0.0f, -1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleConsistencyWithTemperature) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int greedySample = sampler_->sample(logits, 0.0f);
    int lowTempSample = sampler_->sample(logits, 0.1f);
    
    EXPECT_EQ(greedySample, lowTempSample);
}

TEST_F(SamplerTest, SampleRandomnessWithTemperature) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sample1 = sampler_->sample(logits, 5.0f);
    int sample2 = sampler_->sample(logits, 5.0f);
    int sample3 = sampler_->sample(logits, 5.0f);
    
    bool allSame = (sample1 == sample2) && (sample2 == sample3);
    
    EXPECT_FALSE(allSame);
}

TEST_F(SamplerTest, SampleTemperatureEffect) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    std::vector<int> lowTempCounts(5, 0);
    std::vector<int> highTempCounts(5, 0);
    
    const int numSamples = 100;
    
    for (int i = 0; i < numSamples; ++i) {
        int lowSample = sampler_->sample(logits, 0.5f);
        int highSample = sampler_->sample(logits, 5.0f);
        
        if (lowSample >= 0 && lowSample < 5) {
            lowTempCounts[lowSample]++;
        }
        if (highSample >= 0 && highSample < 5) {
            highTempCounts[highSample]++;
        }
    }
    
    EXPECT_GT(lowTempCounts[2], highTempCounts[2]);
}

TEST_F(SamplerTest, SampleWithVerySmallDifferences) {
    std::vector<float> values = {1.0f, 1.0001f, 1.0002f, 1.0001f, 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_EQ(sampled, 2);
}

TEST_F(SamplerTest, SampleStressTest) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    const int numSamples = 10000;
    
    for (int i = 0; i < numSamples; ++i) {
        int sampled = sampler_->sample(logits, 1.0f);
        EXPECT_GE(sampled, 0);
        EXPECT_LT(sampled, 5);
    }
}

TEST_F(SamplerTest, SampleDifferentTemperatures) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    FloatArray logits = createLogits(values);
    
    std::vector<float> temperatures = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
    
    for (float temp : temperatures) {
        int sampled = sampler_->sample(logits, temp);
        EXPECT_GE(sampled, 0);
        EXPECT_LT(sampled, 5);
    }
}

TEST_F(SamplerTest, SampleBoundaryValues) {
    std::vector<float> values = {std::numeric_limits<float>::min(), 
                                  std::numeric_limits<float>::max(),
                                  0.0f,
                                  -std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::epsilon()};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_GE(sampled, 0);
    EXPECT_LT(sampled, 5);
}

TEST_F(SamplerTest, SampleZeroProbability) {
    std::vector<float> values = {1.0f, -std::numeric_limits<float>::infinity(), 
                                  2.0f, -std::numeric_limits<float>::infinity(), 1.0f};
    FloatArray logits = createLogits(values);
    
    int sampled = sampler_->sample(logits, 1.0f);
    
    EXPECT_TRUE(sampled == 0 || sampled == 2 || sampled == 4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
