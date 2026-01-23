/**
 * @file gguf_loader.h
 * @brief GGUF 模型加载器
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 从 GGUF 文件加载模型配置和权重
 */
#pragma once

#include "cllm/kylin/gguf/context.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief GGUF 模型配置
 * 
 * 从 GGUF 文件的元数据中解析出的模型参数
 */
struct GGUFModelConfig {
    // 基本信息
    std::string architecture;       ///< 架构名称 (如 "qwen2", "llama")
    std::string name;               ///< 模型名称
    
    // 结构参数
    uint32_t contextLength = 0;     ///< 最大上下文长度
    uint32_t embeddingLength = 0;   ///< 隐藏层维度
    uint32_t blockCount = 0;        ///< Transformer 层数
    uint32_t headCount = 0;         ///< 注意力头数
    uint32_t headCountKV = 0;       ///< KV 头数（GQA）
    uint32_t feedForwardLength = 0; ///< FFN 中间层维度
    uint32_t vocabSize = 0;         ///< 词表大小
    uint32_t keyLength = 0;         ///< K/V 每头维度（从 GGUF 读取）
    
    // 归一化参数
    float rmsNormEps = 1e-6f;       ///< RMS Norm epsilon
    float ropeFreqBase = 10000.0f;  ///< RoPE 频率基数
    
    // RoPE 参数
    int ropeType = 0;               ///< RoPE 类型 (0=NORMAL, 2=NEOX for Qwen/LLaMA)
    
    // 量化类型
    ggml_type quantType = GGML_TYPE_F32;  ///< 主要量化类型
    
    /**
     * @brief 检查配置是否有效
     */
    bool isValid() const {
        return !architecture.empty() && 
               embeddingLength > 0 && 
               blockCount > 0 && 
               headCount > 0 && 
               vocabSize > 0;
    }
    
    /**
     * @brief 获取每个注意力头的维度
     * 优先使用 GGUF 中指定的 keyLength，否则从 embeddingLength 计算
     */
    uint32_t headDim() const {
        if (keyLength > 0) return keyLength;
        return (headCount > 0) ? (embeddingLength / headCount) : 0;
    }
};

/**
 * @brief Tokenizer 信息（从 GGUF 提取）
 */
struct TokenizerInfo {
    std::string model;              ///< Tokenizer 类型 (如 "gpt2", "llama")
    std::vector<std::string> tokens;///< 词表
    std::vector<float> scores;      ///< Token 分数
    
    // 特殊 token ID
    int32_t bosId = -1;             ///< BOS token ID
    int32_t eosId = -1;             ///< EOS token ID
    int32_t padId = -1;             ///< PAD token ID
    int32_t unkId = -1;             ///< UNK token ID
    
    bool isValid() const {
        return !model.empty() && !tokens.empty();
    }
};

/**
 * @brief GGUF 模型加载器
 * 
 * 职责：
 * - 解析 GGUF 文件格式
 * - 读取模型配置（从元数据）
 * - 加载量化权重张量
 * - 提取 Tokenizer 信息
 */
class GGUFLoader {
public:
    /**
     * @brief 构造函数
     * @param path GGUF 文件路径
     */
    explicit GGUFLoader(const std::string& path);
    
    /**
     * @brief 析构函数
     */
    ~GGUFLoader();
    
    // 禁止拷贝
    GGUFLoader(const GGUFLoader&) = delete;
    GGUFLoader& operator=(const GGUFLoader&) = delete;
    
    // ========== 文件操作 ==========
    
    /**
     * @brief 检查文件是否有效
     */
    bool isValid() const;
    
    /**
     * @brief 获取文件路径
     */
    const std::string& getPath() const { return path_; }
    
    /**
     * @brief 获取 GGUF 版本
     */
    uint32_t getVersion() const;
    
    // ========== 配置加载 ==========
    
    /**
     * @brief 加载模型配置
     * @return 模型配置结构
     */
    GGUFModelConfig loadConfig();
    
    /**
     * @brief 获取 Tokenizer 信息
     * @return Tokenizer 信息（如果存在）
     */
    std::optional<TokenizerInfo> getTokenizerInfo();
    
    // ========== 张量加载 ==========
    
    /**
     * @brief 获取张量数量
     */
    int64_t getTensorCount() const;
    
    /**
     * @brief 获取张量名称列表
     */
    std::vector<std::string> getTensorNames() const;
    
    /**
     * @brief 获取张量类型
     * @param name 张量名称
     * @return GGML 数据类型
     */
    ggml_type getTensorType(const std::string& name) const;
    
    /**
     * @brief 获取张量形状
     * @param name 张量名称
     * @return 形状向量
     */
    std::vector<int64_t> getTensorShape(const std::string& name) const;
    
    /**
     * @brief 加载张量到 GGML 上下文
     * @param ctx GGML 上下文
     * @param tensors 输出：张量名称到指针的映射
     */
    void loadTensors(GGMLContext* ctx, std::map<std::string, ggml_tensor*>& tensors);
    
    /**
     * @brief 加载单个张量
     * @param ctx GGML 上下文
     * @param name 张量名称
     * @return 张量指针（如果存在）
     */
    ggml_tensor* loadTensor(GGMLContext* ctx, const std::string& name);
    
    /**
     * @brief 加载模型权重到 kylin::Tensor（前向声明）
     * 
     * 此方法从 GGUF 文件加载量化权重，反量化为 FP32，
     * 并填充到 kylin::Tensor 容器中。
     * 
     * @param embedding Token embedding 张量
     * @param wq Query 权重（每层）
     * @param wk Key 权重（每层）
     * @param wv Value 权重（每层）
     * @param wo Output 权重（每层）
     * @param wGate FFN Gate 权重（每层）
     * @param wUp FFN Up 权重（每层）
     * @param wDown FFN Down 权重（每层）
     * @param norm1 Attention Norm（每层）
     * @param norm2 FFN Norm（每层）
     * @param finalNorm Final Norm
     * @param lmHead LM Head 权重
     * @return true 成功，false 失败
     */
    template<typename Tensor>
    bool loadInto(
        Tensor& embedding,
        std::vector<Tensor>& wq,
        std::vector<Tensor>& wk,
        std::vector<Tensor>& wv,
        std::vector<Tensor>& wo,
        std::vector<Tensor>& wGate,
        std::vector<Tensor>& wUp,
        std::vector<Tensor>& wDown,
        std::vector<Tensor>& norm1,
        std::vector<Tensor>& norm2,
        Tensor& finalNorm,
        Tensor& lmHead
    );
    
    // ========== 元数据访问 ==========
    
    /**
     * @brief 获取字符串类型的元数据
     * @param key 键名
     * @return 值（如果存在）
     */
    std::optional<std::string> getMetaString(const std::string& key) const;
    
    /**
     * @brief 获取整数类型的元数据
     * @param key 键名
     * @return 值（如果存在）
     */
    std::optional<int64_t> getMetaInt(const std::string& key) const;
    
    /**
     * @brief 获取浮点类型的元数据
     * @param key 键名
     * @return 值（如果存在）
     */
    std::optional<float> getMetaFloat(const std::string& key) const;
    
    /**
     * @brief 获取原始 GGUF 上下文
     */
    gguf_context* raw() { return ggufCtx_; }
    const gguf_context* raw() const { return ggufCtx_; }

private:
    std::string path_;              ///< 文件路径
    gguf_context* ggufCtx_;         ///< GGUF 上下文
    ggml_context* dataCtx_;         ///< 数据 GGML 上下文
    
    /**
     * @brief 从架构前缀获取键名
     * @param arch 架构名称
     * @param suffix 键后缀
     * @return 完整键名
     */
    std::string getArchKey(const std::string& arch, const std::string& suffix) const;
    
    /**
     * @brief 解析架构特定的配置
     * @param config 配置结构（输出）
     * @param arch 架构名称
     */
    void parseArchConfig(GGUFModelConfig& config, const std::string& arch);
};

} // namespace kylin
} // namespace cllm
