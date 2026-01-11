#pragma once

#include "../tokenizer/i_tokenizer.h"
#include "performance_monitor.h"
#include "performance_config.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

// Forward declaration since we can't include llama.h directly here due to path issues
using llama_token = int32_t;

namespace cllm {

class CTokenizer {
public:
    virtual ~CTokenizer() = default;
    
    // 核心功能
    virtual std::vector<llama_token> encode(
        const std::string& text, 
        bool addSpecialTokens = true
    ) = 0;
    
    virtual std::string decode(
        const std::vector<llama_token>& ids,
        bool skipSpecialTokens = true
    ) = 0;
    
    // 词汇表操作
    virtual int getVocabSize() const = 0;
    virtual std::string idToToken(llama_token id) const = 0;
    virtual llama_token tokenToId(const std::string& token) const = 0;
    
    // 特殊Token
    virtual llama_token getBosId() const = 0;
    virtual llama_token getEosId() const = 0;
    virtual llama_token getPadId() const = 0;
    virtual llama_token getUnkId() const = 0;
    
    // 模型类型
    virtual cllm::ModelType getModelType() const = 0;
    
    // 加载模型
    virtual bool load(const std::string& modelPath) = 0;
    
    // 性能监控接口
    virtual void enablePerformanceMonitor(bool enable = true) {
        if (enable && !perfMonitor_) {
            perfMonitor_ = std::make_unique<PerformanceMonitor>();
        } else if (!enable) {
            perfMonitor_.reset();
        }
    }
    
    virtual bool isPerformanceMonitorEnabled() const {
        return perfMonitor_ != nullptr;
    }
    
    virtual TokenizerPerformanceStats getPerformanceStats() const {
        if (perfMonitor_) {
            return perfMonitor_->getStats();
        }
        return TokenizerPerformanceStats{};
    }
    
    virtual void resetPerformanceStats() {
        if (perfMonitor_) {
            perfMonitor_->reset();
        }
    }
    
    // 性能配置接口
    virtual void setPerformanceConfig(const TokenizerPerformanceConfig& config) {
        perfConfig_ = config;
        applyPerformanceConfig();
    }
    
    virtual TokenizerPerformanceConfig getPerformanceConfig() const {
        return perfConfig_;
    }
    
protected:
    std::unique_ptr<IPerformanceMonitor> perfMonitor_;
    TokenizerPerformanceConfig perfConfig_ = TokenizerPerformanceConfig::getDefault();
    
    // 子类需要实现这个方法来应用性能配置
    virtual void applyPerformanceConfig() {}
};

} // namespace cllm