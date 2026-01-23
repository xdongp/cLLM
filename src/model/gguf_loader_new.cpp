#include "cllm/model/gguf_loader_new.h"
#include "cllm/model/gguf_dequantization.h"
#include "cllm/kylin/core/quantization.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <limits>
#include <string>
#include <cmath>

namespace cllm {

GGUFLoader::GGUFLoader(const std::string& filepath, bool useMemoryMap, bool allowMultipleMappings)
    : filepath_(filepath),
      file_(nullptr),
      useMemoryMap_(useMemoryMap),
      currentPosition_(0),
      needByteOrderSwap_(false),
      allowMultipleMappings_(allowMultipleMappings),
      ggufVersion_(0),
      alignment_(32) // 默认对齐值32
#ifdef _WIN32
    , fileHandle_(INVALID_HANDLE_VALUE),
      mapHandle_(nullptr),
      mappedMemory_(nullptr)
#else
    , fileDescriptor_(-1),
      mappedMemory_(nullptr)
#endif
{
    // 打开文件
    file_ = fopen(filepath.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("无法打开GGUF文件: " + filepath);
    }
    
    // 获取文件大小
    fseek(file_, 0, SEEK_END);
    fileSize_ = ftell(file_);
    fseek(file_, 0, SEEK_SET);
    
    // 检查文件大小是否足够大
    if (fileSize_ < sizeof(GGUFHeader)) {
        fclose(file_);
        file_ = nullptr;
        throw std::runtime_error("GGUF文件太小，无法包含完整的文件头");
    }
    
    // 检查系统字节序
    needByteOrderSwap_ = !isSystemLittleEndian();
    CLLM_INFO(needByteOrderSwap_ ? "系统是大端字节序，需要进行字节序转换" : "系统是小端字节序，不需要进行字节序转换");
    
    // 初始化内存映射（如果启用）
    if (useMemoryMap_) {
        try {
            initializeMemoryMap();
        } catch (const std::exception& e) {
            CLLM_WARN("初始化内存映射失败: %s，将使用传统文件I/O", e.what());
            useMemoryMap_ = false;
            // 不需要调用releaseMemoryMap()，因为initializeMemoryMap()在失败时已经清理了资源
        }
    }
}

GGUFLoader::~GGUFLoader() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    releaseMemoryMap();
}

bool GGUFLoader::load() {
    try {
        // 解析文件头
        GGUFHeader header = parseHeader();
        CLLM_INFO("GGUF文件头解析成功: 版本 %u, 张量数量 %zu, 元数据数量 %zu", 
                  header.version, header.tensorCount, header.metadataCount);
        
        // 解析元数据
        parseMetadata(header.metadataCount);
        CLLM_INFO("元数据解析完成，共 %zu 项", metadata_.size());
        
        // 解析张量信息（在提取配置之前，以便可以从张量信息推断vocab_size等参数）
        parseTensorInfos(header.tensorCount);
        CLLM_INFO("张量信息解析完成，共 %zu 个张量", tensorInfos_.size());
        
        // 提取模型配置
        extractModelConfig();
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF模型失败: %s", e.what());
        return false;
    }
}

bool GGUFLoader::loadWeights(cllm::model::ModelWeights& weights, bool loadAll) {
    try {
        // 如果还没有加载文件，先加载
        if (metadata_.empty() || tensorInfos_.empty()) {
            if (!load()) {
                return false;
            }
        }
        
        // 为权重分配基本属性
        weights.embedding.name = "embedding";
        weights.embedding.dtype = kylin::WeightDType::FP32;
        
        weights.finalNorm.name = "finalNorm";
        weights.finalNorm.dtype = kylin::WeightDType::FP32;
        
        weights.lmHead.name = "lmHead";
        weights.lmHead.dtype = kylin::WeightDType::FP32;
        
        // 根据配置创建层
        for (size_t i = 0; i < config_.numLayers; ++i) {
            model::LayerWeights layer;
            
            // 设置层权重的基本属性
            layer.wq.name = "layers." + std::to_string(i) + ".attention.wq.weight";
            layer.wq.dtype = kylin::WeightDType::FP32;
            
            layer.wk.name = "layers." + std::to_string(i) + ".attention.wk.weight";
            layer.wk.dtype = kylin::WeightDType::FP32;
            
            layer.wv.name = "layers." + std::to_string(i) + ".attention.wv.weight";
            layer.wv.dtype = kylin::WeightDType::FP32;
            
            layer.wo.name = "layers." + std::to_string(i) + ".attention.wo.weight";
            layer.wo.dtype = kylin::WeightDType::FP32;
            
            layer.wGate.name = "layers." + std::to_string(i) + ".feed_forward.wGate.weight";
            layer.wGate.dtype = kylin::WeightDType::FP32;
            
            layer.wUp.name = "layers." + std::to_string(i) + ".feed_forward.wUp.weight";
            layer.wUp.dtype = kylin::WeightDType::FP32;
            
            layer.wDown.name = "layers." + std::to_string(i) + ".feed_forward.wDown.weight";
            layer.wDown.dtype = kylin::WeightDType::FP32;
            
            layer.norm1.name = "layers." + std::to_string(i) + ".attention_norm.weight";
            layer.norm1.dtype = kylin::WeightDType::FP32;
            
            layer.norm2.name = "layers." + std::to_string(i) + ".ffn_norm.weight";
            layer.norm2.dtype = kylin::WeightDType::FP32;
            
            // Q/K 独立归一化权重（Qwen3等模型需要，可选）
            layer.attnQNorm.name = "layers." + std::to_string(i) + ".attention.attn_q_norm.weight";
            layer.attnQNorm.dtype = kylin::WeightDType::FP32;
            
            layer.attnKNorm.name = "layers." + std::to_string(i) + ".attention.attn_k_norm.weight";
            layer.attnKNorm.dtype = kylin::WeightDType::FP32;
            
            weights.layers.push_back(layer);
        }
        
        // 如果需要立即加载所有权重数据
        if (loadAll) {
            // 加载embedding
            loadWeightByName("embedding", weights.embedding);
            
            // 加载finalNorm
            loadWeightByName("finalNorm", weights.finalNorm);
            
            // 加载lmHead
            loadWeightByName("lmHead", weights.lmHead);
            
            // 加载所有层的权重
            for (size_t i = 0; i < config_.numLayers; ++i) {
                model::LayerWeights& layer = weights.layers[i];
                
                loadWeightByName("layers." + std::to_string(i) + ".attention.wq.weight", layer.wq);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wk.weight", layer.wk);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wv.weight", layer.wv);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wo.weight", layer.wo);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wGate.weight", layer.wGate);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wUp.weight", layer.wUp);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wDown.weight", layer.wDown);
                loadWeightByName("layers." + std::to_string(i) + ".attention_norm.weight", layer.norm1);
                loadWeightByName("layers." + std::to_string(i) + ".ffn_norm.weight", layer.norm2);
                
                // 尝试加载 Q/K 独立归一化权重（可选，如果不存在会返回false但不报错）
                // 支持多种命名格式：Qwen3可能使用 blk.X.attn_q_norm.weight 或 layers.X.attention.attn_q_norm.weight
                std::string layerPrefix = "layers." + std::to_string(i);
                std::string blkPrefix = "blk." + std::to_string(i);
                
                // 尝试多种可能的命名格式
                bool qNormLoaded = loadWeightByName(layerPrefix + ".attention.attn_q_norm.weight", layer.attnQNorm);
                if (!qNormLoaded) {
                    qNormLoaded = loadWeightByName(blkPrefix + ".attn_q_norm.weight", layer.attnQNorm);
                }
                if (!qNormLoaded) {
                    qNormLoaded = loadWeightByName(layerPrefix + ".attn_q_norm.weight", layer.attnQNorm);
                }
                
                bool kNormLoaded = loadWeightByName(layerPrefix + ".attention.attn_k_norm.weight", layer.attnKNorm);
                if (!kNormLoaded) {
                    kNormLoaded = loadWeightByName(blkPrefix + ".attn_k_norm.weight", layer.attnKNorm);
                }
                if (!kNormLoaded) {
                    kNormLoaded = loadWeightByName(layerPrefix + ".attn_k_norm.weight", layer.attnKNorm);
                }
                
                if (qNormLoaded && kNormLoaded) {
                    CLLM_DEBUG("Loaded attn_q_norm and attn_k_norm for layer %zu", i);
                } else if (qNormLoaded || kNormLoaded) {
                    CLLM_WARN("Layer %zu: Only one of attn_q_norm/attn_k_norm loaded (q=%d, k=%d)", 
                             i, qNormLoaded, kNormLoaded);
                }
            }
        }
        
        // 更新权重映射
        weights.updateWeightMap();
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF权重失败: %s", e.what());
        return false;
    }
}

bool GGUFLoader::loadWeightByName(const std::string& name, cllm::model::WeightData& weight) {
    try {
        auto it = tensorNameMap_.find(name);
        if (it == tensorNameMap_.end()) {
            CLLM_WARN("权重 '%s' 不存在", name.c_str());
            return false;
        }
        
        const GGULTensorInfo& tensorInfo = tensorInfos_[it->second];
        
        // 保存当前位置
        uint64_t savedPos = getCurrentFilePosition();
        
        // 定位到张量数据位置
        // 注意：GGUF中的tensorInfo.offset是相对数据段起点(data section)的偏移
        // dataSectionOffset_在parseTensorInfos()对齐到数据段后确定
        setFilePosition(dataSectionOffset_ + tensorInfo.offset);
        
        // 设置权重的元数据
        weight.name = name;
        weight.shape.clear();
        weight.shape.reserve(tensorInfo.shape.size());
        for (uint64_t dim : tensorInfo.shape) {
            weight.shape.push_back(static_cast<size_t>(dim));
        }
        
        // 根据张量类型设置数据类型
        weight.dtype = kylin::WeightDType::FP32;
        
        // 计算权重元素数量
        size_t elementCount = weight.elementCount();

        // 输出缓冲区：仅在成功读取/反量化后填充
        // （重要）失败时必须保持为空，避免上层把“失败”当成“成功但全0”。
        weight.data.clear();
        weight.data.reserve(elementCount);
        
        // 根据张量类型读取并反量化权重数据
        uint32_t tensorType = static_cast<uint32_t>(tensorInfo.type);
        CLLM_INFO("张量 %s 类型: %u", name.c_str(), tensorType);
        
        switch (tensorType) {
            case 0: // F32
                // 直接读取
                weight.data.resize(elementCount);
                readValues(weight.data.data(), elementCount);
                break;
            case 1: // F16
                {
                    // 预分配F16数据缓冲区
                    std::vector<uint16_t> f16Data;
                    f16Data.reserve(elementCount);
                    f16Data.resize(elementCount);

                    // 批量读取F16数据
                    readValues(f16Data.data(), elementCount);

                    // 反量化输出
                    weight.data.resize(elementCount);
                    dequantizeF16ToF32(f16Data.data(), weight.data.data(), elementCount);
                }
                break;
            case 2: // Q8_0
                {
                    // 预分配Q8_0数据缓冲区
                    std::vector<int8_t> q8Data;
                    q8Data.reserve(elementCount);
                    q8Data.resize(elementCount);
                    
                    // 批量读取Q8_0数据
                    readValues(q8Data.data(), elementCount);
                    
                    // 使用SIMD优化的反量化
                    dequantizeQ8ToF32(q8Data.data(), weight.data.data(), elementCount);
                }
                break;
            case 12: // Q4_K (Q4_K_M)
                {
                    // Q4_K格式: 每个块144字节，包含256个元素
                    using namespace kylin::quantization;

                    const size_t blockCount = (elementCount + QK_K - 1) / QK_K;
                    const size_t q4DataSize = blockCount * sizeof(block_q4_K);

                    std::vector<uint8_t> q4Data;
                    q4Data.reserve(q4DataSize);
                    q4Data.resize(q4DataSize);

                    // 批量读取Q4_K数据
                    readValues(q4Data.data(), q4DataSize);

                    // 反量化输出
                    weight.data.resize(elementCount);
                    dequantize_q4_K_to_f32(q4Data.data(), weight.data.data(), elementCount);
                }
                break;
            case 13: // Q5_K (Q5_K_M)
                {
                    // Q5_K格式: 每个块包含1个缩放因子 + 2位高位数据 + 4位低位数据 (共7字节/块)
                    size_t blockCount = (elementCount + 7) / 8;
                    size_t q5DataSize = blockCount * 7;
                    
                    // 预分配Q5_K数据缓冲区
                    std::vector<uint8_t> q5Data;
                    q5Data.reserve(q5DataSize);
                    q5Data.resize(q5DataSize);
                    
                    // 批量读取Q5_K数据
                    readValues(q5Data.data(), q5DataSize);
                    
                    // 使用SIMD优化的反量化
                    dequantizeQ5KMF32(q5Data.data(), weight.data.data(), weight.shape);
                }
                break;
            case 14: // Q6_K
                {
                    // Q6_K格式：256个元素/块，使用 kylin::quantization 的反量化实现
                    using namespace kylin::quantization;

                    const size_t blockCount = (elementCount + QK_K - 1) / QK_K;
                    const size_t q6DataSize = blockCount * sizeof(block_q6_K);

                    std::vector<uint8_t> q6Data;
                    q6Data.reserve(q6DataSize);
                    q6Data.resize(q6DataSize);

                    // 批量读取Q6_K数据
                    readValues(q6Data.data(), q6DataSize);

                    // 反量化输出
                    weight.data.resize(elementCount);
                    dequantize_q6_K_to_f32(q6Data.data(), weight.data.data(), elementCount);
                }
                break;
            default:
                CLLM_ERROR("不支持的张量类型: %u", static_cast<uint32_t>(tensorInfo.type));
                weight.data.clear();
                setFilePosition(savedPos);
                return false;
        }
        
        // 恢复当前位置
        setFilePosition(savedPos);
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载张量 %s 失败: %s", name.c_str(), e.what());
        weight.data.clear();
        return false;
    }
}

bool GGUFLoader::hasWeight(const std::string& name) {
    return tensorNameMap_.find(name) != tensorNameMap_.end();
}

bool GGUFLoader::loadInto(
    kylin::Tensor& embedding,
    std::vector<kylin::Tensor>& wq,
    std::vector<kylin::Tensor>& wk,
    std::vector<kylin::Tensor>& wv,
    std::vector<kylin::Tensor>& wo,
    std::vector<kylin::Tensor>& wGate,
    std::vector<kylin::Tensor>& wUp,
    std::vector<kylin::Tensor>& wDown,
    std::vector<kylin::Tensor>& norm1,
    std::vector<kylin::Tensor>& norm2,
    kylin::Tensor& finalNorm,
    kylin::Tensor& lmHead
) {
    try {
        // 调试：打印所有张量名称（用于调试）
        CLLM_INFO("可用的张量名称（前30个）:");
        size_t count = 0;
        for (const auto& [name, index] : tensorNameMap_) {
            if (count++ < 30) {
                CLLM_INFO("  [%zu] %s", index, name.c_str());
            } else {
                break;
            }
        }
        
        // 查找embedding相关的张量名称
        CLLM_INFO("查找embedding相关张量:");
        std::vector<std::string> possibleEmbeddingNames;
        for (const auto& [name, index] : tensorNameMap_) {
            std::string nameLower = name;
            std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);
            if (nameLower.find("embed") != std::string::npos || 
                nameLower.find("tok") != std::string::npos ||
                nameLower.find("token") != std::string::npos) {
                possibleEmbeddingNames.push_back(name);
                CLLM_INFO("  找到可能的embedding张量: %s", name.c_str());
            }
        }
        
        // 如果没有找到，尝试查找不以blk.开头的第一个张量（可能是embedding）
        if (possibleEmbeddingNames.empty()) {
            CLLM_INFO("未找到明确的embedding张量，查找不以blk.开头的张量:");
            for (const auto& [name, index] : tensorNameMap_) {
                if (name.find("blk.") != 0 && name.find("output") == std::string::npos) {
                    possibleEmbeddingNames.push_back(name);
                    CLLM_INFO("  候选张量: %s", name.c_str());
                    if (possibleEmbeddingNames.size() >= 5) break;  // 只显示前5个
                }
            }
        }
        
        // 辅助函数：加载权重并复制到Tensor
        auto loadWeightToTensor = [this](const std::string& tensorName, kylin::Tensor& tensor) -> bool {
            cllm::model::WeightData weight;
            if (!loadWeightByName(tensorName, weight)) {
                // 尝试其他可能的名称
                std::vector<std::string> alternatives;
                if (tensorName == "embedding") {
                    alternatives = {"tok_embeddings.weight", "embed_tokens.weight"};
                } else if (tensorName.find("finalNorm") != std::string::npos || tensorName.find("output_norm") != std::string::npos) {
                    alternatives = {"output_norm.weight", "norm.weight", "model.norm.weight"};
                } else if (tensorName.find("lmHead") != std::string::npos || tensorName.find("output") != std::string::npos) {
                    alternatives = {"output.weight", "lm_head.weight", "output.weight"};
                }
                
                for (const auto& alt : alternatives) {
                    if (loadWeightByName(alt, weight)) {
                        break;
                    }
                }
                
                if (weight.data.empty()) {
                    CLLM_WARN("无法找到张量: %s", tensorName.c_str());
                    return false;
                }
            }
            
            // 设置Tensor形状并复制数据
            std::vector<size_t> shape(weight.shape.begin(), weight.shape.end());
            
            // 调试：打印权重形状信息
            CLLM_INFO("加载权重 %s: 形状 [", tensorName.c_str());
            for (size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) CLLM_INFO(", ");
                CLLM_INFO("%zu", shape[i]);
            }
            CLLM_INFO("], 元素数: %zu", weight.data.size());
            
            // 对于embedding和lmHead，可能需要转置
            // GGUF中通常是 [hidden, vocab]，但代码期望 [vocab, hidden] 和 [hidden, vocab]
            bool needTranspose = false;
            if (tensorName.find("embedding") != std::string::npos || 
                tensorName.find("token_embd") != std::string::npos ||
                tensorName.find("tok_emb") != std::string::npos) {
                // embedding: GGUF通常是 [hidden, vocab]，但代码期望 [vocab, hidden]
                if (shape.size() == 2 && shape[0] < shape[1]) {
                    // 如果第一个维度小于第二个，可能是 [hidden, vocab]，需要转置
                    CLLM_INFO("检测到embedding权重可能需要转置: [%zu, %zu] -> [%zu, %zu]", 
                             shape[0], shape[1], shape[1], shape[0]);
                    std::swap(shape[0], shape[1]);
                    needTranspose = true;
                }
            }
            // 注意：lmHead不需要转置，因为代码期望 [hidden, vocab]，GGUF中也是 [hidden, vocab]
            
            tensor = kylin::Tensor(shape);
            if (tensor.size() != weight.data.size()) {
                CLLM_ERROR("张量 %s 大小不匹配: Tensor=%zu, WeightData=%zu", 
                          tensorName.c_str(), tensor.size(), weight.data.size());
                return false;
            }
            
            // 如果需要转置，先转置再复制
            if (needTranspose && shape.size() == 2) {
                size_t rows = weight.shape[0];
                size_t cols = weight.shape[1];
                if (rows == 0 || cols == 0) {
                    CLLM_ERROR("转置时发现无效维度: rows=%zu, cols=%zu", rows, cols);
                    return false;
                }
                float* dst = tensor.data();
                const float* src = weight.data.data();
                if (dst == nullptr || src == nullptr) {
                    CLLM_ERROR("转置时发现空指针: dst=%p, src=%p", dst, src);
                    return false;
                }
                CLLM_DEBUG("执行转置: [%zu, %zu] -> [%zu, %zu]", rows, cols, cols, rows);
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        size_t srcIdx = i * cols + j;
                        size_t dstIdx = j * rows + i;
                        if (srcIdx >= weight.data.size() || dstIdx >= tensor.size()) {
                            CLLM_ERROR("转置时索引越界: srcIdx=%zu (size=%zu), dstIdx=%zu (size=%zu)", 
                                     srcIdx, weight.data.size(), dstIdx, tensor.size());
                            return false;
                        }
                        dst[dstIdx] = src[srcIdx];
                    }
                }
            } else {
                if (tensor.size() != weight.data.size()) {
                    CLLM_ERROR("复制时大小不匹配: tensor.size()=%zu, weight.data.size()=%zu", 
                             tensor.size(), weight.data.size());
                    return false;
                }
                if (tensor.data() == nullptr || weight.data.data() == nullptr) {
                    CLLM_ERROR("复制时发现空指针: tensor.data()=%p, weight.data.data()=%p", 
                             tensor.data(), weight.data.data());
                    return false;
                }
                std::copy(weight.data.begin(), weight.data.end(), tensor.data());
            }
            
            // 验证并清理数据
            float* data = tensor.data();
            if (data == nullptr) {
                CLLM_ERROR("tensor.data() 返回空指针");
                return false;
            }
            bool hasInvalid = false;
            size_t invalidCount = 0;
            for (size_t i = 0; i < tensor.size(); ++i) {
                if (std::isnan(data[i]) || std::isinf(data[i])) {
                    hasInvalid = true;
                    invalidCount++;
                    // 将NaN/Inf替换为0.0
                    data[i] = 0.0f;
                }
            }
            if (hasInvalid) {
                CLLM_WARN("权重 %s 包含NaN或Inf值 (共检测到 %zu 个，已替换为0.0)", 
                         tensorName.c_str(), invalidCount);
            }
            
            return true;
        };
        
        // 加载embedding - 根据分析结果，Qwen模型使用 token_embd.weight
        std::vector<std::string> embeddingNames = {
            "token_embd.weight",        // Qwen实际格式（从分析结果确认）
            "tok_emb.weight",           // Qwen可能格式
            "tok_embeddings.weight",    // Llama格式
            "embed_tokens.weight",      // HuggingFace格式
            "embedding.weight",         // 通用格式
            "embeddings.weight",        // 通用格式
            "token_embeddings.weight",  // 完整格式
            "tok_emb",                  // 无.weight后缀
            "embedding"                 // 无.weight后缀
        };
        
        // 将找到的可能embedding名称添加到列表前面（优先使用自动发现的）
        embeddingNames.insert(embeddingNames.begin(), possibleEmbeddingNames.begin(), possibleEmbeddingNames.end());
        bool embeddingLoaded = false;
        for (const auto& name : embeddingNames) {
            if (loadWeightToTensor(name, embedding)) {
                embeddingLoaded = true;
                CLLM_INFO("成功加载embedding权重: %s", name.c_str());
                break;
            }
        }
        if (!embeddingLoaded) {
            CLLM_ERROR("无法加载embedding权重，尝试的名称: %s", 
                      [&embeddingNames]() {
                          std::string result;
                          for (size_t i = 0; i < embeddingNames.size(); ++i) {
                              if (i > 0) result += ", ";
                              result += embeddingNames[i];
                          }
                          return result;
                      }().c_str());
            return false;
        }
        
        // 加载各层权重
        const size_t numLayers = config_.numLayers;
        wq.resize(numLayers);
        wk.resize(numLayers);
        wv.resize(numLayers);
        wo.resize(numLayers);
        wGate.resize(numLayers);
        wUp.resize(numLayers);
        wDown.resize(numLayers);
        norm1.resize(numLayers);
        norm2.resize(numLayers);
        
        for (size_t i = 0; i < numLayers; ++i) {
            // Qwen模型使用 blk.{layer}.xxx.weight 格式
            std::string blkPrefix = "blk." + std::to_string(i);
            // 也尝试其他可能的格式
            std::string layerPrefix = "layers." + std::to_string(i);
            
            // 尝试多种可能的命名约定（Qwen格式优先）
            std::vector<std::pair<std::string, kylin::Tensor*>> layerWeights = {
                // Qwen格式 (blk.X.xxx)
                {blkPrefix + ".attn_q.weight", &wq[i]},
                {blkPrefix + ".attn_k.weight", &wk[i]},
                {blkPrefix + ".attn_v.weight", &wv[i]},
                {blkPrefix + ".attn_output.weight", &wo[i]},
                {blkPrefix + ".ffn_gate.weight", &wGate[i]},
                {blkPrefix + ".ffn_up.weight", &wUp[i]},
                {blkPrefix + ".ffn_down.weight", &wDown[i]},
                {blkPrefix + ".attn_norm.weight", &norm1[i]},
                {blkPrefix + ".ffn_norm.weight", &norm2[i]},
                // Llama格式 (layers.X.xxx)
                {layerPrefix + ".attention.wq.weight", &wq[i]},
                {layerPrefix + ".attn.q_proj.weight", &wq[i]},
                {layerPrefix + ".attention.wk.weight", &wk[i]},
                {layerPrefix + ".attn.k_proj.weight", &wk[i]},
                {layerPrefix + ".attention.wv.weight", &wv[i]},
                {layerPrefix + ".attn.v_proj.weight", &wv[i]},
                {layerPrefix + ".attention.wo.weight", &wo[i]},
                {layerPrefix + ".attn.o_proj.weight", &wo[i]},
                {layerPrefix + ".feed_forward.wGate.weight", &wGate[i]},
                {layerPrefix + ".mlp.gate_proj.weight", &wGate[i]},
                {layerPrefix + ".feed_forward.wUp.weight", &wUp[i]},
                {layerPrefix + ".mlp.up_proj.weight", &wUp[i]},
                {layerPrefix + ".feed_forward.wDown.weight", &wDown[i]},
                {layerPrefix + ".mlp.down_proj.weight", &wDown[i]},
                {layerPrefix + ".attention_norm.weight", &norm1[i]},
                {layerPrefix + ".input_layernorm.weight", &norm1[i]},
                {layerPrefix + ".ffn_norm.weight", &norm2[i]},
                {layerPrefix + ".post_attention_layernorm.weight", &norm2[i]}
            };
            
            // 加载每个权重（为每个权重类型分别尝试）
            // 注意：这里需要为每个权重类型分别尝试，不能只尝试第一个就停止
            std::vector<std::pair<std::vector<std::string>, kylin::Tensor*>> weightGroups = {
                {{blkPrefix + ".attn_q.weight", layerPrefix + ".attention.wq.weight", layerPrefix + ".attn.q_proj.weight"}, &wq[i]},
                {{blkPrefix + ".attn_k.weight", layerPrefix + ".attention.wk.weight", layerPrefix + ".attn.k_proj.weight"}, &wk[i]},
                {{blkPrefix + ".attn_v.weight", layerPrefix + ".attention.wv.weight", layerPrefix + ".attn.v_proj.weight"}, &wv[i]},
                {{blkPrefix + ".attn_output.weight", layerPrefix + ".attention.wo.weight", layerPrefix + ".attn.o_proj.weight"}, &wo[i]},
                {{blkPrefix + ".ffn_gate.weight", layerPrefix + ".feed_forward.wGate.weight", layerPrefix + ".mlp.gate_proj.weight"}, &wGate[i]},
                {{blkPrefix + ".ffn_up.weight", layerPrefix + ".feed_forward.wUp.weight", layerPrefix + ".mlp.up_proj.weight"}, &wUp[i]},
                {{blkPrefix + ".ffn_down.weight", layerPrefix + ".feed_forward.wDown.weight", layerPrefix + ".mlp.down_proj.weight"}, &wDown[i]},
                {{blkPrefix + ".attn_norm.weight", layerPrefix + ".attention_norm.weight", layerPrefix + ".input_layernorm.weight"}, &norm1[i]},
                {{blkPrefix + ".ffn_norm.weight", layerPrefix + ".ffn_norm.weight", layerPrefix + ".post_attention_layernorm.weight"}, &norm2[i]}
            };
            
            for (auto& [names, tensor] : weightGroups) {
                bool found = false;
                for (const auto& name : names) {
                    if (loadWeightToTensor(name, *tensor)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    CLLM_WARN("层 %zu 的权重未找到（尝试了 %zu 个名称）", i, names.size());
                }
            }
        }
        
        // 加载finalNorm - 根据分析结果，Qwen模型使用 output_norm.weight
        std::vector<std::string> finalNormNames = {
            "output_norm.weight",       // Qwen实际格式（从分析结果确认）
            "norm.weight",
            "final_norm.weight",
            "model.norm.weight"
        };
        bool finalNormLoaded = false;
        for (const auto& name : finalNormNames) {
            if (loadWeightToTensor(name, finalNorm)) {
                finalNormLoaded = true;
                CLLM_INFO("成功加载finalNorm权重: %s", name.c_str());
                break;
            }
        }
        if (!finalNormLoaded) {
            CLLM_WARN("无法加载finalNorm权重，将使用默认值");
            finalNorm = kylin::Tensor({config_.hiddenSize});
            finalNorm.fill(1.0f);  // 使用默认值
        }
        
        // 加载lmHead - 根据分析结果，Qwen模型使用 output.weight
        CLLM_INFO("开始加载lmHead权重...");
        std::vector<std::string> lmHeadNames = {
            "output.weight",            // Qwen实际格式（从分析结果确认）
            "lm_head.weight",
            "head.weight"
        };
        bool lmHeadLoaded = false;
        for (const auto& name : lmHeadNames) {
            CLLM_INFO("尝试加载lmHead权重: %s", name.c_str());
            try {
                if (loadWeightToTensor(name, lmHead)) {
                    lmHeadLoaded = true;
                    CLLM_INFO("成功加载lmHead权重: %s", name.c_str());
                    break;
                } else {
                    CLLM_DEBUG("权重 %s 加载失败，尝试下一个", name.c_str());
                }
            } catch (const std::exception& e) {
                CLLM_ERROR("加载权重 %s 时发生异常: %s", name.c_str(), e.what());
            } catch (...) {
                CLLM_ERROR("加载权重 %s 时发生未知异常", name.c_str());
            }
        }
        if (!lmHeadLoaded) {
            CLLM_ERROR("无法加载lmHead权重（尝试了 %zu 个名称）", lmHeadNames.size());
            return false;
        }
        
        CLLM_INFO("成功加载GGUF模型到张量结构");
        CLLM_INFO("loadInto 即将返回 true");
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF模型到张量结构失败: %s", e.what());
        return false;
    }
}

const ModelConfig& GGUFLoader::getConfig() const {
    return config_;
}

const std::string& GGUFLoader::getModelPath() const {
    return filepath_;
}

kylin::WeightDType GGUFLoader::getDType() const {
    // 根据模型配置返回数据类型
    return kylin::WeightDType::FP32; // 默认返回FP32，需要根据实际情况修改
}

GGUFHeader GGUFLoader::parseHeader() {
    GGUFHeader header;
    
    // 先读取原始魔数字节
    uint8_t magic_bytes[4];
    if (useMemoryMap_) {
        if (currentPosition_ + 4 > fileSize_) {
            throw std::runtime_error("文件太小，无法读取魔数");
        }
        memcpy(magic_bytes, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, 4);
        currentPosition_ += 4;
    } else {
        if (fread(magic_bytes, 1, 4, file_) != 4) {
            throw std::runtime_error("无法读取魔数");
        }
        currentPosition_ += 4;
    }
    
    // 检查魔数的字节序（参考llama.cpp的实现）
    // GGUF魔数：'GGUF' = 字节序列 0x47 0x47 0x55 0x46
    // 小端格式：按字节顺序存储，读取为uint32_t时是 0x46554747
    // 大端格式：按字节顺序存储，读取为uint32_t时是 0x47554746
    const uint32_t GGUF_MAGIC_LE = 0x46554747; // 'GGUF' in little-endian
    
    // 直接比较字节序列（不依赖字节序解释）
    // 'GGUF' = 0x47 0x47 0x55 0x46
    bool isValidMagic = (magic_bytes[0] == 0x47 && magic_bytes[1] == 0x47 && 
                         magic_bytes[2] == 0x55 && magic_bytes[3] == 0x46);
    
    if (isValidMagic) {
        // 文件是小端字节序（标准格式）
        needByteOrderSwap_ = !isSystemLittleEndian();
        header.magic = GGUF_MAGIC_LE;
        CLLM_INFO("检测到GGUF文件（小端格式）");
    } else {
        // 检查是否是大端格式（GGUF v3支持）
        // 大端格式：字节顺序相反 0x46 0x55 0x47 0x47
        bool isBigEndianMagic = (magic_bytes[0] == 0x46 && magic_bytes[1] == 0x55 && 
                                 magic_bytes[2] == 0x47 && magic_bytes[3] == 0x47);
        if (isBigEndianMagic) {
            // 文件是大端字节序（GGUF v3支持）
            needByteOrderSwap_ = isSystemLittleEndian();
            header.magic = GGUF_MAGIC_LE; // 统一存储为小端格式
            CLLM_INFO("检测到大端字节序GGUF文件（需要字节序转换）");
    } else {
        // 魔数不匹配 - 打印实际读取的魔数用于调试
        char c0 = std::isprint(magic_bytes[0]) ? magic_bytes[0] : '?';
        char c1 = std::isprint(magic_bytes[1]) ? magic_bytes[1] : '?';
        char c2 = std::isprint(magic_bytes[2]) ? magic_bytes[2] : '?';
        char c3 = std::isprint(magic_bytes[3]) ? magic_bytes[3] : '?';
        throw std::runtime_error(std::string("无效的GGUF文件格式: 魔数不匹配, 读取到 '") + 
                                   c0 + c1 + c2 + c3 + "' (0x" + 
                                   std::to_string(static_cast<unsigned>(magic_bytes[0])) + 
                                   std::to_string(static_cast<unsigned>(magic_bytes[1])) + 
                                   std::to_string(static_cast<unsigned>(magic_bytes[2])) + 
                                   std::to_string(static_cast<unsigned>(magic_bytes[3])) + 
                                   "), 期望 'GGUF'");
        }
    }
    
    // 读取版本号
    readValues(&header.version, 1);
    ggufVersion_ = header.version;
    
    // 验证版本号 (参考llama.cpp的字节序检测逻辑)
    if (header.version == 0) {
        throw std::runtime_error("无效的GGUF版本号: 0");
    }
    
    // 检测字节序不匹配: 如果版本号的低16位为0，说明字节序可能不匹配
    // 例如版本3 (0x00000003) 在大端系统上读取为 (0x03000000)
    if ((header.version & 0x0000FFFF) == 0x00000000) {
        throw std::runtime_error("GGUF文件版本号 " + std::to_string(header.version) + 
                               " 异常大，可能存在主机与模型字节序不匹配的问题");
    }
    
    // 检查版本号是否支持
    if (header.version == 1) {
        throw std::runtime_error("GGUFv1不再被支持，请使用更新的版本");
    }
    if (header.version > 3) {
        CLLM_WARN("GGUF版本号 %u 高于当前支持的版本 3，可能存在兼容性问题", header.version);
    }
    
    // 读取张量数量和元数据数量
    readValues(&header.tensorCount, 1);
    readValues(&header.metadataCount, 1);
    
    // 验证张量数量和元数据数量的合理性
    const uint64_t MAX_TENSORS = SIZE_MAX / sizeof(GGULTensorInfo);
    const uint64_t MAX_METADATA = SIZE_MAX / sizeof(GGUFMetadata);
    
    if (header.tensorCount > MAX_TENSORS) {
        throw std::runtime_error("张量数量 " + std::to_string(header.tensorCount) + 
                               " 超过允许的最大值 " + std::to_string(MAX_TENSORS));
    }
    
    if (header.metadataCount > MAX_METADATA) {
        throw std::runtime_error("元数据数量 " + std::to_string(header.metadataCount) + 
                               " 超过允许的最大值 " + std::to_string(MAX_METADATA));
    }
    
    return header;
}

// 验证元数据键是否符合GGUF规范
// 规范要求：lower_snake_case，用.分隔，最多65535字节
static bool validateMetadataKey(const std::string& key) {
    if (key.empty()) {
        return false;
    }
    
    // 检查长度（最多65535字节，即2^16-1）
    if (key.length() > 65535) {
        return false;
    }
    
    // 检查是否为ASCII字符串
    for (char c : key) {
        if (static_cast<unsigned char>(c) > 127) {
            return false; // 非ASCII字符
        }
    }
    
    // 检查格式：lower_snake_case，用.分隔
    // 允许：字母、数字、下划线、点号
    // 不允许：大写字母、连续点号、开头/结尾点号
    if (key[0] == '.' || key[key.length() - 1] == '.') {
        return false;
    }
    
    bool lastWasDot = false;
    for (size_t i = 0; i < key.length(); ++i) {
        char c = key[i];
        if (c == '.') {
            if (lastWasDot) {
                return false; // 连续点号
            }
            lastWasDot = true;
        } else if (c == '_') {
            lastWasDot = false;
        } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
            lastWasDot = false;
        } else if (c >= 'A' && c <= 'Z') {
            return false; // 不允许大写字母
        } else {
            return false; // 不允许其他字符
        }
    }
    
    return true;
}

void GGUFLoader::parseMetadata(uint64_t metadataCount) {
    for (uint64_t i = 0; i < metadataCount; ++i) {
        GGUFMetadata metadata;
        
        // 保存当前文件位置，以便在解析失败时恢复
        uint64_t savedPosition = getCurrentFilePosition();
        
        try {
            // 读取键名
            try {
                metadata.key = readString();
            } catch (const std::length_error& e) {
                CLLM_ERROR("读取元数据键 %zu 时遇到length_error: %s", i, e.what());
                throw;
            } catch (const std::bad_alloc& e) {
                CLLM_ERROR("读取元数据键 %zu 时遇到bad_alloc: %s", i, e.what());
                throw;
            }
            
            // 验证键名是否符合GGUF规范
            if (!validateMetadataKey(metadata.key)) {
                CLLM_WARN("元数据键 '%s' 不符合GGUF规范（应为lower_snake_case，用.分隔），将跳过", metadata.key.c_str());
                setFilePosition(savedPosition);
                continue;
            }
            
            // 检查键名是否重复 (参考llama.cpp的实现)
            for (const auto& [existingKey, existingMetadata] : metadata_) {
                if (metadata.key == existingKey) {
                    CLLM_ERROR("发现重复的元数据键 '%s' (位置 %zu)", metadata.key.c_str(), i);
                    throw std::runtime_error("元数据键重复: " + metadata.key);
                }
            }
            
            // 读取值类型
            uint32_t valueTypeRaw = readValue<uint32_t>();
            
            // 验证值类型是否在有效范围内（参考GGUF规范）
            if (valueTypeRaw >= static_cast<uint32_t>(GGUFValueType::COUNT)) {
                CLLM_WARN("元数据值类型 %u 超出有效范围 [0, %u]，将跳过此元数据项", 
                         valueTypeRaw, static_cast<uint32_t>(GGUFValueType::COUNT) - 1);
                setFilePosition(savedPosition);
                continue;
            }
            
            metadata.type = static_cast<GGUFValueType>(valueTypeRaw);
            
            // 处理数组类型：需要先读取数组元素类型和长度
            bool isArray = false;
            uint64_t arrayElementCount = 1;
            if (metadata.type == GGUFValueType::ARRAY) {
                isArray = true;
                // 读取数组元素类型
                uint32_t arrayElementTypeRaw = readValue<uint32_t>();
                if (arrayElementTypeRaw >= static_cast<uint32_t>(GGUFValueType::COUNT)) {
                    CLLM_WARN("数组元素类型 %u 超出有效范围，将跳过此元数据项", arrayElementTypeRaw);
                    setFilePosition(savedPosition);
                    continue;
                }
                GGUFValueType arrayElementType = static_cast<GGUFValueType>(arrayElementTypeRaw);
                
                // 读取数组长度
                arrayElementCount = readValue<uint64_t>();
                
                // 验证数组长度合理性（防止恶意文件）
                const uint64_t MAX_ARRAY_SIZE = 1024 * 1024 * 1024; // 1GB元素上限
                if (arrayElementCount > MAX_ARRAY_SIZE) {
                    CLLM_WARN("数组长度 %llu 过大，将跳过此元数据项", arrayElementCount);
                    setFilePosition(savedPosition);
                    continue;
                }
                
                // 设置数组元数据
                metadata.array_val.elementType = arrayElementType;
                metadata.array_val.elementCount = arrayElementCount;
                metadata.array_val.elements.reserve(arrayElementCount);
                
                // 读取数组元素
                bool arrayReadSuccess = true;
                for (uint64_t j = 0; j < arrayElementCount; ++j) {
                    GGUFMetadata element;
                    element.type = arrayElementType;
                    try {
                        readMetadataValue(element);
                        metadata.array_val.elements.push_back(element);
                    } catch (const std::exception& e) {
                        CLLM_WARN("读取数组元素 %llu/%llu 失败: %s，将跳过此元数据项", 
                                 j, arrayElementCount, e.what());
                        arrayReadSuccess = false;
                        break; // 跳出内层循环
                    }
                }
                
                // 如果数组读取成功，存储元数据
                if (arrayReadSuccess && metadata.array_val.elements.size() == arrayElementCount) {
                    metadata_[metadata.key] = metadata;
                } else {
                    // 数组读取失败，恢复文件位置并跳过
                    setFilePosition(savedPosition);
                    continue; // 继续外层循环的下一个元数据项
                }
            } else {
                // 读取非数组值
            try {
                readMetadataValue(metadata);
                    // 存储元数据
                    metadata_[metadata.key] = metadata;
            } catch (const std::exception& e) {
                CLLM_WARN("读取元数据值失败: %s，将跳过此元数据项", e.what());
                // 恢复文件位置
                setFilePosition(savedPosition);
                continue;
                }
            }
            
            // 检查是否是对齐值（在存储元数据之后）
            if (metadata.key == "general.alignment") {
                // 根据值类型获取对齐值
                switch (metadata.type) {
                    case GGUFValueType::UINT32:
                        alignment_ = metadata.value.u32_val;
                        break;
                    case GGUFValueType::UINT64:
                        alignment_ = static_cast<uint32_t>(metadata.value.u64_val);
                        break;
                    default:
                        CLLM_WARN("general.alignment 元数据的类型 %u 不是预期的整数类型，将使用默认对齐值 %u", 
                                  valueTypeRaw, alignment_);
                }
                
                // 验证对齐值是否为2的幂
                if (alignment_ != 0 && (alignment_ & (alignment_ - 1)) != 0) {
                    CLLM_WARN("对齐值 %u 不是2的幂，可能导致问题。将使用默认对齐值32", alignment_);
                    alignment_ = 32;
                } else {
                    CLLM_INFO("使用元数据中指定的对齐值: %u", alignment_);
                }
            }
            
        } catch (const std::exception& e) {
            CLLM_ERROR("解析元数据第 %zu 项失败: %s，位置: %zu", i, e.what(), savedPosition);
            
            // 恢复文件位置，避免后续数据错位
            try {
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前元数据项", savedPosition);
            } catch (const std::exception& restoreError) {
                CLLM_ERROR("无法恢复文件位置: %s，解析可能已损坏", restoreError.what());
                // 如果无法恢复位置，抛出异常，因为继续解析会导致数据错位
                throw std::runtime_error("元数据解析失败且无法恢复文件位置: " + std::string(e.what()));
            }
            
            // 跳过当前元数据项，继续解析下一个
            continue;
        }
    }
}

void GGUFLoader::parseTensorInfos(uint64_t tensorCount) {
    tensorInfos_.reserve(tensorCount);
    tensorNameMap_.reserve(tensorCount);
    
    for (uint64_t i = 0; i < tensorCount; ++i) {
        GGULTensorInfo tensorInfo;
        
        // 保存当前文件位置，以便在解析失败时恢复
        uint64_t savedPosition = getCurrentFilePosition();
        
        try {
            // 读取张量名称
            try {
                tensorInfo.name = readString();
            } catch (const std::length_error& e) {
                CLLM_ERROR("读取张量 %zu 名称时遇到length_error: %s", i, e.what());
                throw;
            } catch (const std::bad_alloc& e) {
                CLLM_ERROR("读取张量 %zu 名称时遇到bad_alloc: %s", i, e.what());
                throw;
            }
            
            // 验证张量名称长度（GGUF规范：最多64字节）
            const size_t MAX_TENSOR_NAME_LENGTH = 64;
            if (tensorInfo.name.length() >= MAX_TENSOR_NAME_LENGTH) {
                throw std::runtime_error("张量名称 '" + tensorInfo.name + "' 长度 " + 
                                        std::to_string(tensorInfo.name.length()) + 
                                        " 超过最大允许长度 " + std::to_string(MAX_TENSOR_NAME_LENGTH));
            }
            
            // 检查张量名称是否重复 (参考llama.cpp)
            for (const auto& [existingName, existingIndex] : tensorNameMap_) {
                if (tensorInfo.name == existingName) {
                    CLLM_ERROR("发现重复的张量名称 '%s' (位置 %zu 和 %zu)", 
                             tensorInfo.name.c_str(), existingIndex, i);
                    throw std::runtime_error("张量名称重复: " + tensorInfo.name);
                }
            }
            
            // 读取维度数
            tensorInfo.dimensions = readValue<uint32_t>();
            
            // 验证维度数是否在合理范围内 (参考GGUF规范和llama.cpp)
            // GGUF规范：目前最多4维，但未来可能扩展，这里使用更宽松的限制
            const uint32_t MAX_DIMENSIONS = 4; // GGML_MAX_DIMS
            if (tensorInfo.dimensions > MAX_DIMENSIONS) {
                throw std::runtime_error("张量 " + tensorInfo.name + " 维度数异常: " + 
                                        std::to_string(tensorInfo.dimensions) + 
                                        " (最大允许: " + std::to_string(MAX_DIMENSIONS) + ")");
            }
            
            // 读取形状
            tensorInfo.shape.reserve(tensorInfo.dimensions);
            tensorInfo.shape.resize(tensorInfo.dimensions);
            for (uint32_t j = 0; j < tensorInfo.dimensions; ++j) {
                tensorInfo.shape[j] = readValue<uint64_t>();
                
                // 验证每个维度的大小是否合理（不能为负数或0，除非是标量）
                if (tensorInfo.shape[j] < 0) {
                    throw std::runtime_error("张量 " + tensorInfo.name + " 的第 " + 
                                            std::to_string(j) + " 个维度大小为负数: " + 
                                            std::to_string(tensorInfo.shape[j]));
                }
                if (tensorInfo.shape[j] == 0 && tensorInfo.dimensions > 0) {
                    throw std::runtime_error("张量 " + tensorInfo.name + " 的第 " + 
                                            std::to_string(j) + " 个维度大小为0");
                }
            }
            
            // 验证总元素数不会导致溢出（参考llama.cpp的实现）
            int64_t totalElements = 1;
            for (uint32_t j = 0; j < tensorInfo.dimensions; ++j) {
                if (totalElements > INT64_MAX / static_cast<int64_t>(tensorInfo.shape[j])) {
                    throw std::runtime_error("张量 " + tensorInfo.name + " 的总元素数溢出");
                }
                totalElements *= static_cast<int64_t>(tensorInfo.shape[j]);
            }
            
            // 读取张量类型
            int32_t tensorTypeRaw = readValue<int32_t>();
            
            // 验证张量类型是否在有效范围内（参考GGUF规范和llama.cpp）
            if (tensorTypeRaw < 0 || tensorTypeRaw >= static_cast<int32_t>(GGMLType::COUNT)) {
                throw std::runtime_error("张量 " + tensorInfo.name + " 类型值异常: " + 
                                        std::to_string(tensorTypeRaw) + 
                                        " (有效范围: 0-" + 
                                        std::to_string(static_cast<int32_t>(GGMLType::COUNT) - 1) + ")");
            }
            
            tensorInfo.type = static_cast<GGMLType>(tensorTypeRaw);
            
            // 计算张量大小并验证块大小对齐（参考llama.cpp）
            // 这里需要根据张量类型计算块大小，但为了简化，我们先读取偏移量
            // 块大小验证将在后续添加
            
            // 读取偏移量
            tensorInfo.offset = readValue<uint64_t>();
            
            // 验证偏移量是否对齐（参考GGUF规范：偏移量必须是对齐值的倍数）
            if (alignment_ > 0 && (tensorInfo.offset % alignment_ != 0)) {
                throw std::runtime_error("张量 " + tensorInfo.name + " 的偏移量 " + 
                                        std::to_string(tensorInfo.offset) + 
                                        " 不是对齐值 " + std::to_string(alignment_) + " 的倍数");
            }
            
            // 存储张量信息
            tensorNameMap_[tensorInfo.name] = tensorInfos_.size();
            tensorInfos_.push_back(tensorInfo);
            
        } catch (const std::exception& e) {
            CLLM_ERROR("解析张量信息第 %zu 项失败: %s，位置: %zu", i, e.what(), savedPosition);
            
            // 恢复文件位置，避免后续数据错位
            try {
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前张量信息项", savedPosition);
            } catch (const std::exception& restoreError) {
                CLLM_ERROR("无法恢复文件位置: %s，解析可能已损坏", restoreError.what());
                throw std::runtime_error("张量信息解析失败且无法恢复文件位置: " + std::string(e.what()));
            }
            
            // 跳过当前张量信息，继续解析下一个
            continue;
        }
    }
    
    // 对齐文件位置到数据段开始位置（参考llama.cpp的实现）
    // GGUF规范要求：张量信息后需要填充到对齐边界，然后才是数据段
    uint64_t alignedPosition = alignOffset(currentPosition_);
    if (alignedPosition > currentPosition_) {
        // 需要跳过填充字节
        uint64_t paddingSize = alignedPosition - currentPosition_;
        if (useMemoryMap_) {
            currentPosition_ = alignedPosition;
        } else {
            // 对于文件I/O，需要实际跳过填充字节
            if (fseek(file_, static_cast<long>(alignedPosition), SEEK_SET) != 0) {
                throw std::runtime_error("无法对齐文件位置到数据段开始");
            }
            currentPosition_ = alignedPosition;
        }
        CLLM_INFO("将文件位置从 %zu 对齐到 %zu (对齐值 %u，填充 %zu 字节)", 
                 currentPosition_ - paddingSize, alignedPosition, alignment_, paddingSize);
    }

    // 记录数据段起始位置：GGUF张量offset均相对该位置
    dataSectionOffset_ = currentPosition_;

    // 验证张量偏移的对齐（参考llama.cpp的实现）
    // 注意：GGUF文件中的偏移量是相对于数据段开始位置(dataSectionOffset_)的相对偏移
    // 不同GGUF文件可能使用不同的张量排列顺序，所以不强制要求连续偏移
    // 我们只验证偏移量是否对齐，以及是否在文件范围内
    
    uint64_t dataSectionOffset = dataSectionOffset_;
    uint64_t minOffset = UINT64_MAX;
    uint64_t maxOffset = 0;
    
    for (size_t i = 0; i < tensorInfos_.size(); ++i) {
        const GGULTensorInfo& ti = tensorInfos_[i];
        
        // 验证偏移量是否对齐
        if (alignment_ > 0 && (ti.offset % alignment_ != 0)) {
            CLLM_WARN("张量 '%s' 的偏移量 %zu 不是对齐值 %u 的倍数，但继续处理", 
                     ti.name.c_str(), ti.offset, alignment_);
        }
        
        // 计算张量大小（需要考虑对齐）
        size_t tensorSize = getTensorByteSize(ti);
        size_t paddedSize = alignOffset(tensorSize);
        
        // 验证偏移量是否在文件范围内
        // ti.offset 是相对 dataSectionOffset_ 的偏移，因此这里需要加上 dataSectionOffset_
        uint64_t tensorEndOffset = dataSectionOffset_ + ti.offset + paddedSize;
        if (tensorEndOffset > fileSize_) {
            throw std::runtime_error("张量 '" + ti.name + "' 的结束偏移量 " + 
                                    std::to_string(tensorEndOffset) + 
                                    " 超出文件大小 " + std::to_string(fileSize_));
        }
        
        // 跟踪最小和最大偏移量（这里记录的是“相对数据段”的范围，便于观察）
        if (ti.offset < minOffset) {
            minOffset = ti.offset;
        }
        uint64_t endOffsetRelative = ti.offset + paddedSize;
        if (endOffsetRelative > maxOffset) {
            maxOffset = endOffsetRelative;
        }
    }
    
    CLLM_INFO("张量偏移验证完成（相对数据段），范围: [%zu, %zu] 字节，总大小: %zu 字节", 
             minOffset, maxOffset, maxOffset - minOffset);
    CLLM_INFO("GGUF 数据段起始位置: %zu", dataSectionOffset_);
}

// 辅助函数：从元数据中提取uint32_t值
template<typename T>
uint32_t GGUFLoader::extractUInt32(const GGUFMetadata& metadata) {
    switch (metadata.type) {
        case GGUFValueType::UINT32: return metadata.value.u32_val;
        case GGUFValueType::UINT64: return static_cast<uint32_t>(metadata.value.u64_val);
        case GGUFValueType::INT32: return static_cast<uint32_t>(metadata.value.i32_val);
        case GGUFValueType::INT64: return static_cast<uint32_t>(metadata.value.i64_val);
        case GGUFValueType::UINT16: return static_cast<uint32_t>(metadata.value.u16_val);
        case GGUFValueType::INT16: return static_cast<uint32_t>(metadata.value.i16_val);
        case GGUFValueType::UINT8: return static_cast<uint32_t>(metadata.value.u8_val);
        case GGUFValueType::INT8: return static_cast<uint32_t>(metadata.value.i8_val);
        case GGUFValueType::BOOL: return metadata.value.bool_val ? 1 : 0;
        default:
            CLLM_WARN("无法从元数据类型 %u 提取uint32_t值", static_cast<uint32_t>(metadata.type));
            return 0;
    }
}

// 辅助函数：尝试从多个元数据键中提取uint32_t值
bool GGUFLoader::tryExtractUInt32(uint32_t& result, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            result = extractUInt32<uint32_t>(it->second);
            return true;
        }
    }
    return false;
}

// 辅助函数：尝试从多个元数据键中提取字符串值
bool GGUFLoader::tryExtractString(std::string& result, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = metadata_.find(key);
        if (it != metadata_.end() && it->second.type == GGUFValueType::STRING) {
            result = it->second.string_val;
            return true;
        }
    }
    return false;
}

// 辅助函数：尝试从多个元数据键中提取浮点数值
bool GGUFLoader::tryExtractFloat32(float& result, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            if (it->second.type == GGUFValueType::FLOAT32) {
                result = it->second.value.f32_val;
                return true;
            } else if (it->second.type == GGUFValueType::FLOAT64) {
                result = static_cast<float>(it->second.value.f64_val);
                return true;
            }
        }
    }
    return false;
}

void GGUFLoader::extractModelConfig() {
    // 从元数据中提取模型配置
    // 设置默认值
    config_.modelType = "llama";
    config_.vocabSize = 32000;
    config_.hiddenSize = 768;
    config_.numLayers = 6;
    config_.numAttentionHeads = 8;
    config_.numKeyValueHeads = 8;
    config_.maxSequenceLength = 512;
    config_.intermediateSize = 3072;
    config_.ropeTheta = 10000.0f;  // 默认 RoPE theta
    config_.rmsNormEps = 1e-5f;    // 默认 RMSNorm eps
    config_.useKVCache = true;
    config_.useQuantization = false;
    config_.useMemoryCompression = false;
    config_.quantizationType = "";
    
    // 提取模型类型
    std::vector<std::string> modelTypeKeys = {
        "general.architecture",
        "model_type",
        "tokenizer.ggml.model"
    };
    tryExtractString(config_.modelType, modelTypeKeys);
    CLLM_INFO("检测到模型类型: %s", config_.modelType.c_str());
    
    // 提取上下文长度
    uint32_t maxSequenceLength = 512;
    std::vector<std::string> contextLengthKeys = {"ggml.context_length"};
    if (tryExtractUInt32(maxSequenceLength, contextLengthKeys)) {
        config_.maxSequenceLength = maxSequenceLength;
    } else if (metadata_.count("tokenizer.chat_template") > 0) {
        config_.maxSequenceLength = 4096; // 默认较大值
    }
    
    // 提取层数
    uint32_t numLayers = 6;
    std::vector<std::string> numLayersKeys = {
        "ggml.n_layers",
        "llama.layers",
        "n_layer"
    };
    if (tryExtractUInt32(numLayers, numLayersKeys)) {
        config_.numLayers = numLayers;
    }
    
    // 提取注意力头数
    uint32_t numAttentionHeads = 8;
    std::vector<std::string> attentionHeadsKeys = {
        "ggml.n_heads",
        "llama.attention.head_count",
        "n_head"
    };
    if (tryExtractUInt32(numAttentionHeads, attentionHeadsKeys)) {
        config_.numAttentionHeads = numAttentionHeads;
    }
    
    // 提取KV头数
    uint32_t numKeyValueHeads = 8;
    std::vector<std::string> kvHeadsKeys = {
        "ggml.n_kv_heads",
        "n_kv_head"
    };
    if (tryExtractUInt32(numKeyValueHeads, kvHeadsKeys)) {
        config_.numKeyValueHeads = numKeyValueHeads;
    } else {
        // 默认与注意力头数相等
        config_.numKeyValueHeads = config_.numAttentionHeads;
    }
    
    // 提取隐藏维度
    uint32_t hiddenSize = 768;
    std::vector<std::string> hiddenSizeKeys = {
        "ggml.dim",
        "llama.hidden_size",
        "llama.embedding_length",
        "n_embd",
        "qwen3.hidden_size",
        "qwen.hidden_size"
    };
    if (tryExtractUInt32(hiddenSize, hiddenSizeKeys)) {
        config_.hiddenSize = hiddenSize;
        CLLM_INFO("从元数据提取hidden_size: %u", hiddenSize);
    }
    
    // 提取词汇表大小
    uint32_t vocabSize = 32000;
    std::vector<std::string> vocabSizeKeys = {
        "ggml.vocab_size",
        "tokenizer.ggml.vocab_size",
        "qwen3.vocab_size",
        "qwen.vocab_size"
    };
    if (tryExtractUInt32(vocabSize, vocabSizeKeys)) {
        config_.vocabSize = vocabSize;
        CLLM_INFO("从元数据提取vocab_size: %u", vocabSize);
    }
    
    // 对于Qwen模型，从embedding层权重或输出层权重推断vocab size和hidden size
    if (config_.modelType == "qwen3" || config_.modelType == "qwen") {
        // 尝试从token_embd.weight（embedding）和output.weight（LM head）推断
        for (const auto& tensorInfo : tensorInfos_) {
            if (tensorInfo.name == "token_embd.weight" || tensorInfo.name == "output.weight") {
                if (tensorInfo.shape.size() >= 2) {
                    uint32_t dim0 = static_cast<uint32_t>(tensorInfo.shape[0]);
                    uint32_t dim1 = static_cast<uint32_t>(tensorInfo.shape[1]);
                    
                    // 对于 embedding: 通常是 [vocab, hidden] 或 [hidden, vocab]
                    // 对于 output.weight: 通常是 [hidden, vocab] 或 [vocab, hidden]
                    // vocab_size 通常是较大的维度，hidden_size 是较小的维度
                    uint32_t largerDim = std::max(dim0, dim1);
                    uint32_t smallerDim = std::min(dim0, dim1);
                    
                    if (tensorInfo.name == "token_embd.weight") {
                        // embedding: 通常是 [vocab, hidden]（转置后）
                        // 如果 dim0 < dim1，可能是 [hidden, vocab]，需要转置
                        // 如果 dim0 > dim1，可能是 [vocab, hidden]，已经是正确格式
                        if (dim0 < dim1) {
                            // [hidden, vocab] -> hidden_size = dim0, vocab_size = dim1
                            if (smallerDim > hiddenSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断hidden_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, smallerDim);
                                config_.hiddenSize = smallerDim;
                                hiddenSize = smallerDim;
                            }
                            if (largerDim > vocabSize) {
                    CLLM_INFO("从 %s (shape=[%u,%u]) 推断vocab_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, largerDim);
                                config_.vocabSize = largerDim;
                                vocabSize = largerDim;
                            }
                        } else {
                            // [vocab, hidden] -> vocab_size = dim0, hidden_size = dim1
                            if (largerDim > vocabSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断vocab_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, largerDim);
                                config_.vocabSize = largerDim;
                                vocabSize = largerDim;
                            }
                            if (smallerDim > hiddenSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断hidden_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, smallerDim);
                                config_.hiddenSize = smallerDim;
                                hiddenSize = smallerDim;
                            }
                        }
                    } else if (tensorInfo.name == "output.weight") {
                        // output.weight: 通常是 [hidden, vocab]
                        if (dim0 < dim1) {
                            // [hidden, vocab] -> hidden_size = dim0, vocab_size = dim1
                            if (smallerDim > hiddenSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断hidden_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, smallerDim);
                                config_.hiddenSize = smallerDim;
                                hiddenSize = smallerDim;
                            }
                            if (largerDim > vocabSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断vocab_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, largerDim);
                                config_.vocabSize = largerDim;
                                vocabSize = largerDim;
                            }
                        } else {
                            // [vocab, hidden] -> vocab_size = dim0, hidden_size = dim1
                            if (largerDim > vocabSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断vocab_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, largerDim);
                                config_.vocabSize = largerDim;
                                vocabSize = largerDim;
                            }
                            if (smallerDim > hiddenSize) {
                                CLLM_INFO("从 %s (shape=[%u,%u]) 推断hidden_size = %u",  
                                         tensorInfo.name.c_str(), dim0, dim1, smallerDim);
                                config_.hiddenSize = smallerDim;
                                hiddenSize = smallerDim;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 提取中间层大小
    uint32_t intermediateSize = 3072;
    std::vector<std::string> intermediateSizeKeys = {
        "ggml.intermediate_size",
        "n_ff"
    };
    if (tryExtractUInt32(intermediateSize, intermediateSizeKeys)) {
        config_.intermediateSize = intermediateSize;
    } else {
        // 对于Llama系列，中间层大小通常是隐藏层大小的2.65倍
        config_.intermediateSize = static_cast<size_t>(config_.hiddenSize * 2.65f);
    }
    
    // 为Qwen3模型添加特定的配置支持
    if (config_.modelType == "qwen3" || config_.modelType == "qwen") {
        CLLM_INFO("应用Qwen特定配置");
        
        // 提取Qwen隐藏维度
        uint32_t qwenHiddenSize = 0;
        std::vector<std::string> qwenHiddenSizeKeys = {
            "qwen3.hidden_size",
            "qwen.hidden_size"
        };
        if (tryExtractUInt32(qwenHiddenSize, qwenHiddenSizeKeys)) {
            config_.hiddenSize = qwenHiddenSize;
        }
        
        // 提取Qwen层数
        uint32_t qwenNumLayers = 0;
        std::vector<std::string> qwenNumLayersKeys = {
            "qwen3.num_layers",
            "qwen.num_layers",
            "llama.block_count",  // 通用层数键
            "llama.n_layer"       // Llama格式
        };
        if (tryExtractUInt32(qwenNumLayers, qwenNumLayersKeys)) {
            config_.numLayers = qwenNumLayers;
        } else {
            // 如果元数据中没有，尝试从张量名称推断层数
            // 查找最大的 blk.X 编号
            uint32_t maxLayerNum = 0;
            for (const auto& [name, index] : tensorNameMap_) {
                if (name.find("blk.") == 0) {
                    size_t dotPos = name.find('.', 4);  // 跳过 "blk."
                    if (dotPos != std::string::npos) {
                        try {
                            uint32_t layerNum = std::stoul(name.substr(4, dotPos - 4));
                            if (layerNum > maxLayerNum) {
                                maxLayerNum = layerNum;
                            }
                        } catch (...) {
                            // 忽略解析错误
                        }
                    }
                }
            }
            if (maxLayerNum > 0) {
                config_.numLayers = maxLayerNum + 1;  // 层编号从0开始，所以+1
                CLLM_INFO("从张量名称推断层数: %u (最大blk编号: %u)", config_.numLayers, maxLayerNum);
            }
        }
        
        // 提取Qwen注意力头数
        uint32_t qwenAttentionHeads = 0;
        std::vector<std::string> qwenAttentionHeadsKeys = {
            "qwen3.num_attention_heads",
            "qwen.num_attention_heads"
        };
        if (tryExtractUInt32(qwenAttentionHeads, qwenAttentionHeadsKeys)) {
            config_.numAttentionHeads = qwenAttentionHeads;
        }
        
        // 提取Qwen KV头数
        uint32_t qwenKeyValueHeads = 0;
        std::vector<std::string> qwenKeyValueHeadsKeys = {
            "qwen3.num_key_value_heads",
            "qwen.num_key_value_heads"
        };
        if (tryExtractUInt32(qwenKeyValueHeads, qwenKeyValueHeadsKeys)) {
            config_.numKeyValueHeads = qwenKeyValueHeads;
        }
        
        // 提取Qwen中间层大小
        uint32_t qwenIntermediateSize = 0;
        std::vector<std::string> qwenIntermediateSizeKeys = {
            "qwen3.intermediate_size",
            "qwen.intermediate_size"
        };
        if (tryExtractUInt32(qwenIntermediateSize, qwenIntermediateSizeKeys)) {
            config_.intermediateSize = qwenIntermediateSize;
        }
        
        // 提取Qwen最大序列长度
        uint32_t qwenContextLength = 0;
        std::vector<std::string> qwenContextLengthKeys = {
            "qwen3.context_length",
            "qwen.context_length"
        };
        if (tryExtractUInt32(qwenContextLength, qwenContextLengthKeys)) {
            config_.maxSequenceLength = qwenContextLength;
        }
    }
    
    // 提取 RoPE theta（rope.freq_base）
    float ropeTheta = 10000.0f;  // 默认值
    std::vector<std::string> ropeThetaKeys = {
        "llm.rope.freq_base",
        "rope.freq_base",
        "qwen3.rope.freq_base",
        "qwen.rope.freq_base"
    };
    if (tryExtractFloat32(ropeTheta, ropeThetaKeys)) {
        config_.ropeTheta = ropeTheta;
        CLLM_INFO("从元数据提取rope_theta (freq_base): %f", ropeTheta);
    } else {
        config_.ropeTheta = ropeTheta;  // 使用默认值
        CLLM_INFO("使用默认rope_theta: %f", ropeTheta);
    }
    
    // 提取 RMSNorm epsilon
    float rmsNormEps = 1e-5f;  // 默认值
    std::vector<std::string> rmsNormEpsKeys = {
        "llm.rms_norm_eps",
        "rms_norm_eps",
        "qwen3.rms_norm_eps",
        "qwen.rms_norm_eps"
    };
    if (tryExtractFloat32(rmsNormEps, rmsNormEpsKeys)) {
        config_.rmsNormEps = rmsNormEps;
        CLLM_INFO("从元数据提取rms_norm_eps: %e", rmsNormEps);
    } else {
        config_.rmsNormEps = rmsNormEps;  // 使用默认值
        CLLM_INFO("使用默认rms_norm_eps: %e", rmsNormEps);
    }
    
    // P3修复：提取 RoPE 扩展参数
    // 1. rope_n_ctx_orig (原始上下文长度)
    uint32_t ropeNctxOrig = config_.maxSequenceLength;  // 默认等于maxSequenceLength
    std::vector<std::string> ropeNctxOrigKeys = {
        "llm.rope.n_ctx_orig",
        "rope.n_ctx_orig",
        "qwen3.rope.n_ctx_orig",
        "qwen.rope.n_ctx_orig"
    };
    if (tryExtractUInt32(ropeNctxOrig, ropeNctxOrigKeys)) {
        config_.ropeNctxOrig = ropeNctxOrig;
        CLLM_INFO("从元数据提取rope_n_ctx_orig: %u", ropeNctxOrig);
    } else {
        config_.ropeNctxOrig = config_.maxSequenceLength;  // 默认等于maxSequenceLength
        CLLM_INFO("使用默认rope_n_ctx_orig: %u (等于maxSequenceLength)", config_.ropeNctxOrig);
    }
    
    // 2. rope_freq_scale (频率缩放因子)
    float ropeFreqScale = 1.0f;  // 默认值
    std::vector<std::string> ropeFreqScaleKeys = {
        "llm.rope.freq_scale",
        "rope.freq_scale",
        "qwen3.rope.freq_scale",
        "qwen.rope.freq_scale"
    };
    if (tryExtractFloat32(ropeFreqScale, ropeFreqScaleKeys)) {
        config_.ropeFreqScale = ropeFreqScale;
        CLLM_INFO("从元数据提取rope_freq_scale: %f", ropeFreqScale);
    } else {
        config_.ropeFreqScale = 1.0f;  // 使用默认值
        CLLM_INFO("使用默认rope_freq_scale: %f", config_.ropeFreqScale);
    }
    
    // 3. rope_type (RoPE类型)
    int32_t ropeType = 0;  // 默认标准RoPE
    std::vector<std::string> ropeTypeKeys = {
        "llm.rope.type",
        "rope.type",
        "qwen3.rope.type",
        "qwen.rope.type"
    };
    // 注意：rope_type 可能是 UINT32 或 INT32，先尝试 UINT32
    uint32_t ropeTypeUint = 0;
    if (tryExtractUInt32(ropeTypeUint, ropeTypeKeys)) {
        ropeType = static_cast<int32_t>(ropeTypeUint);
        config_.ropeType = ropeType;
        CLLM_INFO("从元数据提取rope_type: %d", ropeType);
    } else {
        config_.ropeType = 0;  // 使用默认值（标准RoPE）
        CLLM_INFO("使用默认rope_type: %d", config_.ropeType);
    }
    
    // 4. rope_ext_factor (扩展因子)
    float ropeExtFactor = 1.0f;  // 默认值
    std::vector<std::string> ropeExtFactorKeys = {
        "llm.rope.ext_factor",
        "rope.ext_factor",
        "qwen3.rope.ext_factor",
        "qwen.rope.ext_factor"
    };
    if (tryExtractFloat32(ropeExtFactor, ropeExtFactorKeys)) {
        config_.ropeExtFactor = ropeExtFactor;
        CLLM_INFO("从元数据提取rope_ext_factor: %f", ropeExtFactor);
    } else {
        config_.ropeExtFactor = 1.0f;  // 使用默认值
        CLLM_INFO("使用默认rope_ext_factor: %f", config_.ropeExtFactor);
    }
    
    // 检查是否使用量化
    if (!tensorInfos_.empty()) {
        const GGULTensorInfo& firstTensor = tensorInfos_[0];
        uint32_t tensorType = static_cast<uint32_t>(firstTensor.type);
        
        CLLM_INFO("第一个张量名称: %s, 类型: %u", firstTensor.name.c_str(), tensorType);
        
        // 如果是量化类型，设置量化相关配置
        if (tensorType == 2 || tensorType == 13 || tensorType == 14) {
            config_.useQuantization = true;
            switch (tensorType) {
                case 2: config_.quantizationType = "Q8_0"; break;
                case 13: config_.quantizationType = "Q4_K_M"; break;
                case 14: config_.quantizationType = "Q5_K_M"; break;
                default: config_.quantizationType = "未知"; break;
            }
        }
    }
    
    CLLM_INFO("GGUF model configuration extracted:");
    CLLM_INFO("  Model type: %s", config_.modelType.c_str());
    CLLM_INFO("  Vocab size: %u", config_.vocabSize);
    CLLM_INFO("  Hidden size: %u", config_.hiddenSize);
    CLLM_INFO("  Number of layers: %u", config_.numLayers);
    CLLM_INFO("  Attention heads: %u", config_.numAttentionHeads);
    CLLM_INFO("  KV heads: %u", config_.numKeyValueHeads);
    CLLM_INFO("  Max sequence length: %u", config_.maxSequenceLength);
    CLLM_INFO("  Intermediate size: %u", config_.intermediateSize);
}

// 获取GGML类型的块大小（每个块包含的元素数）
static int64_t getGGMLBlockSize(GGMLType type) {
    switch (type) {
        // "传统"量化：32个元素/块
        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
        case GGMLType::Q5_0:
        case GGMLType::Q5_1:
        case GGMLType::Q8_0:
        case GGMLType::Q8_1:
            return 32;

        // K-quant：256个元素/块 (QK_K)
        case GGMLType::Q2_K:
        case GGMLType::Q3_K:
        case GGMLType::Q4_K:
        case GGMLType::Q5_K:
        case GGMLType::Q6_K:
        case GGMLType::Q8_K:
            return 256;

        // IQ/TQ/MXFP4 这里先按32处理（后续如需严格支持再补齐）
        case GGMLType::IQ2_XXS:
        case GGMLType::IQ2_XS:
        case GGMLType::IQ3_XXS:
        case GGMLType::IQ1_S:
        case GGMLType::IQ4_NL:
        case GGMLType::IQ3_S:
        case GGMLType::IQ2_S:
        case GGMLType::IQ4_XS:
        case GGMLType::IQ1_M:
        case GGMLType::TQ1_0:
        case GGMLType::TQ2_0:
        case GGMLType::MXFP4:
            return 32;

        default:
            return 1;
    }
}

// 获取GGML类型的元素大小（每个元素的字节数，对于量化类型是块的平均大小）
static size_t getGGMLTypeSize(GGMLType type) {
    switch (type) {
        case GGMLType::F32: return 4;
        case GGMLType::F16: return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::I8: return 1;
        case GGMLType::I16: return 2;
        case GGMLType::I32: return 4;
        case GGMLType::I64: return 8;
        case GGMLType::F64: return 8;
        case GGMLType::Q4_0: return 18; // 32个元素，每个块18字节（2字节scale + 16字节数据）
        case GGMLType::Q4_1: return 20; // 32个元素，每个块20字节（2字节scale + 2字节bias + 16字节数据）
        case GGMLType::Q5_0: return 22; // 32个元素，每个块22字节（2字节scale + 20字节数据）
        case GGMLType::Q5_1: return 24; // 32个元素，每个块24字节（2字节scale + 2字节bias + 20字节数据）
        case GGMLType::Q8_0: return 34; // 32个元素，每个块34字节（2字节scale + 32字节数据）
        case GGMLType::Q8_1: return 36; // 32个元素，每个块36字节（2字节scale + 2字节bias + 32字节数据）
        // K-quant：这里的 size 指的是“每个256元素块”的字节数（与llama.cpp的type_traits一致）
        // 说明：当前工程已实现 Q4_K 的反量化/读取，因此这里至少要保证 Q4_K 正确。
        case GGMLType::Q2_K: return 0;   // TODO: 补齐具体块大小后启用严格校验
        case GGMLType::Q3_K: return 0;   // TODO: 补齐具体块大小后启用严格校验
        case GGMLType::Q4_K: return 144; // sizeof(block_q4_K)
        case GGMLType::Q5_K: return 176; // sizeof(block_q5_K) (llama.cpp/ggml)
        case GGMLType::Q6_K: return 210; // sizeof(block_q6_K) (llama.cpp/ggml)
        case GGMLType::Q8_K: return 0;   // TODO: 补齐具体块大小后启用严格校验
        case GGMLType::IQ2_XXS: return 8;  // 32个元素，每个块8字节
        case GGMLType::IQ2_XS: return 10;  // 32个元素，每个块10字节
        case GGMLType::IQ3_XXS: return 10; // 32个元素，每个块10字节
        case GGMLType::IQ1_S: return 6;     // 32个元素，每个块6字节
        case GGMLType::IQ4_NL: return 18;   // 32个元素，每个块18字节
        case GGMLType::IQ3_S: return 12;   // 32个元素，每个块12字节
        case GGMLType::IQ2_S: return 12;   // 32个元素，每个块12字节
        case GGMLType::IQ4_XS: return 20;   // 32个元素，每个块20字节
        case GGMLType::IQ1_M: return 8;     // 32个元素，每个块8字节
        case GGMLType::TQ1_0: return 10;     // 32个元素，每个块10字节
        case GGMLType::TQ2_0: return 18;    // 32个元素，每个块18字节
        case GGMLType::MXFP4: return 18;    // 32个元素，每个块18字节
        default:
            return 4; // 默认4字节
    }
}

size_t GGUFLoader::getTensorByteSize(const GGULTensorInfo& tensorInfo) const {
    // 计算元素总数
    uint64_t elementCount = 1;
    for (uint64_t dim : tensorInfo.shape) {
        if (elementCount > UINT64_MAX / dim) {
            throw std::runtime_error("张量元素总数计算溢出");
        }
        elementCount *= dim;
    }
    
    // 对于量化类型，需要按块计算
    int64_t blockSize = getGGMLBlockSize(tensorInfo.type);
    if (blockSize > 1) {
        // 量化类型：需要确保第一维是块大小的倍数
        if (tensorInfo.shape.empty() || tensorInfo.shape[0] % blockSize != 0) {
            throw std::runtime_error("张量 '" + tensorInfo.name + 
                                    "' 的第一维 " + std::to_string(tensorInfo.shape.empty() ? 0 : tensorInfo.shape[0]) +
                                    " 不是块大小 " + std::to_string(blockSize) + " 的倍数");
        }
        
        // 计算块数
        uint64_t blockCount = elementCount / blockSize;
        size_t blockByteSize = getGGMLTypeSize(tensorInfo.type);
        
        // 检查溢出
        if (blockCount > SIZE_MAX / blockByteSize) {
            throw std::runtime_error("张量字节大小计算溢出");
        }
        
        return blockCount * blockByteSize;
    } else {
        // 非量化类型：直接计算
        size_t elementSize = getGGMLTypeSize(tensorInfo.type);
        
        // 检查溢出
        if (elementCount > SIZE_MAX / elementSize) {
            throw std::runtime_error("张量字节大小计算溢出");
    }
    
        return elementCount * elementSize;
    }
}

uint64_t GGUFLoader::getCurrentFilePosition() const {
    return currentPosition_;
}

void GGUFLoader::setFilePosition(uint64_t offset) {
    if (offset > fileSize_) {
        throw std::runtime_error("文件位置超出范围: 偏移量 " + std::to_string(offset) + 
                               " > 文件大小 " + std::to_string(fileSize_));
    }
    
    currentPosition_ = offset;
    
    if (!useMemoryMap_) {
        // 对于大文件，fseek可能使用long类型，需要检查是否超出范围
        if (offset > static_cast<uint64_t>(std::numeric_limits<long>::max())) {
            throw std::runtime_error("文件位置超出fseek支持范围: " + std::to_string(offset));
        }
        
        if (fseek(file_, static_cast<long>(offset), SEEK_SET) != 0) {
            throw std::runtime_error("无法设置文件位置: fseek失败，偏移量 " + std::to_string(offset));
        }
    }
}

uint64_t GGUFLoader::alignOffset(uint64_t offset) const {
    if (alignment_ == 0) {
        return offset;
    }
    return offset + (alignment_ - (offset % alignment_)) % alignment_;
}

bool GGUFLoader::isSystemLittleEndian() const {
    // 检查系统是否是小端字节序
    uint16_t test = 0x1234;
    return *reinterpret_cast<uint8_t*>(&test) == 0x34;
}

void GGUFLoader::initializeMemoryMap() {
#ifdef _WIN32
    // Windows内存映射实现
    fileHandle_ = CreateFile(filepath_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, 
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("无法创建文件句柄");
    }
    
    // 创建内存映射文件
    mapHandle_ = CreateFileMapping(fileHandle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapHandle_) {
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("无法创建内存映射");
    }
    
    // 将文件映射到进程地址空间
    mappedMemory_ = MapViewOfFile(mapHandle_, FILE_MAP_READ, 0, 0, 0);
    if (!mappedMemory_) {
        CloseHandle(mapHandle_);
        CloseHandle(fileHandle_);
        mapHandle_ = nullptr;
        fileHandle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("无法将文件映射到进程地址空间");
    }
    
    CLLM_INFO("成功创建Windows内存映射，文件大小: %zu 字节", fileSize_);
#else
    // Unix/Linux内存映射实现
    fileDescriptor_ = open(filepath_.c_str(), O_RDONLY);
    if (fileDescriptor_ == -1) {
        throw std::runtime_error("无法打开文件描述符");
    }
    
    // 创建内存映射
    mappedMemory_ = mmap(nullptr, fileSize_, PROT_READ, MAP_PRIVATE, fileDescriptor_, 0);
    if (mappedMemory_ == MAP_FAILED) {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
        throw std::runtime_error("无法创建内存映射");
    }
    
    CLLM_INFO("成功创建Unix/Linux内存映射，文件大小: %zu 字节", fileSize_);
#endif
}

void GGUFLoader::releaseMemoryMap() {
#ifdef _WIN32
    if (mappedMemory_) {
        UnmapViewOfFile(mappedMemory_);
        mappedMemory_ = nullptr;
    }
    if (mapHandle_) {
        CloseHandle(mapHandle_);
        mapHandle_ = nullptr;
    }
    if (fileHandle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mappedMemory_ != nullptr && mappedMemory_ != MAP_FAILED) {
        munmap(mappedMemory_, fileSize_);
        mappedMemory_ = nullptr;
    }
    if (fileDescriptor_ != -1) {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
    }
#endif
}

} // namespace cllm
