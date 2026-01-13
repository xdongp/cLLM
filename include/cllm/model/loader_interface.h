/**
 * @file loader_interface.h
 * @brief Model loader interface and factory for supporting multiple model formats
 * @author cLLM Team
 * @date 2026-01-13
 */
#ifndef CLLM_MODEL_LOADER_INTERFACE_H
#define CLLM_MODEL_LOADER_INTERFACE_H

#include "cllm/model/config.h"
#include "cllm/kylin/tensor.h"
#include "cllm/kylin/model_loader.h"
#include "cllm/model/weight_data.h"
#include <string>
#include <vector>
#include <memory>

// 条件包含 LibTorch 头文件
#ifdef ENABLE_LIBTORCH_BACKEND
#include <torch/torch.h>
#endif

namespace cllm {

/**
 * @brief Model format enumeration
 */
enum class ModelFormat {
    BINARY,      // .bin format (existing)
    GGUF,        // .gguf format (new)
    SAFETENSORS, // .safetensors format (existing)
    UNKNOWN      // unknown format
};

/**
 * @brief Abstract interface for model loaders
 * 
 * This interface provides a unified way to load models from different formats
 * while maintaining compatibility with existing Kylin backend tensor loading.
 */
class IModelLoader {
public:
    virtual ~IModelLoader() = default;
    
    /**
     * @brief Load the model file into memory
     * @return true if successful, false otherwise
     */
    virtual bool load() = 0;
    
    /**
     * @brief Load weights into the provided ModelWeights structure
     * 
     * This method loads all model weights into the universal ModelWeights structure,
     * which can be used by any backend.
     * 
     * @param weights Reference to the ModelWeights structure to fill
     * @param loadAll Whether to load all weights immediately (default: true)
     * @return true if successful, false otherwise
     */
    virtual bool loadWeights(model::ModelWeights& weights, bool loadAll = true) = 0;
    
    /**
     * @brief Load a specific weight by name
     * 
     * This method loads a specific weight by its name, implementing on-demand loading.
     * 
     * @param name Name of the weight to load
     * @param weight Reference to the WeightData structure to fill
     * @return true if successful, false otherwise
     */
    virtual bool loadWeightByName(const std::string& name, model::WeightData& weight) = 0;
    
    /**
     * @brief Check if a weight with the given name exists
     * 
     * @param name Name of the weight to check
     * @return true if the weight exists, false otherwise
     */
    virtual bool hasWeight(const std::string& name) = 0;
    
    /**
     * @brief Load weights into provided tensors (Kylin backend compatibility)
     * 
     * This method maintains compatibility with the existing Kylin backend
     * tensor loading interface.
     * 
     * @param embedding Token embedding weights [vocabSize, hiddenSize]
     * @param wq Query projection weights for each layer [hiddenSize, qDim]
     * @param wk Key projection weights for each layer [hiddenSize, kvDim]
     * @param wv Value projection weights for each layer [hiddenSize, kvDim]
     * @param wo Output projection weights for each layer [qDim, hiddenSize]
     * @param wGate FFN gate weights for each layer [hiddenSize, intermediateSize]
     * @param wUp FFN up projection weights for each layer [hiddenSize, intermediateSize]
     * @param wDown FFN down projection weights for each layer [intermediateSize, hiddenSize]
     * @param norm1 Attention layer norm weights for each layer [hiddenSize]
     * @param norm2 FFN layer norm weights for each layer [hiddenSize]
     * @param finalNorm Final layer norm weights [hiddenSize]
     * @param lmHead Language model head weights [hiddenSize, vocabSize]
     * @return true if successful, false otherwise
     */
    virtual bool loadInto(
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
    ) = 0;
    
    /**
     * @brief Get the model configuration
     * @return Reference to the model configuration
     */
    virtual const ModelConfig& getConfig() const = 0;
    
    /**
     * @brief Get the model file path
     * @return Reference to the model file path
     */
    virtual const std::string& getModelPath() const = 0;
    
    /**
     * @brief Get the data type of the weights
     * @return Weight data type
     */
    virtual kylin::WeightDType getDType() const = 0;
    
    /**
     * @brief Load weights directly into a dictionary of torch::Tensor (LibTorch backend)
     * 
     * This method is only available when the LibTorch backend is enabled.
     * It loads all model weights directly into a dictionary of torch::Tensor objects,
     * which can be used by the LibTorch backend.
     * 
     * @param device The device to load the tensors onto (default: CPU)
     * @return A map of weight names to torch::Tensor objects
     */
    #ifdef ENABLE_LIBTORCH_BACKEND
    virtual std::map<std::string, torch::Tensor> loadToTorchTensorDict(
        torch::Device device = torch::kCPU
    ) {
        // Default implementation: load to ModelWeights first, then convert to torch::Tensor
        model::ModelWeights weights;
        if (!loadWeights(weights)) {
            return {};
        }
        
        std::map<std::string, torch::Tensor> tensorDict;
        auto allWeights = weights.getAllWeights();
        
        for (const auto& pair : allWeights) {
            const std::string& name = pair.first;
            model::WeightData* weightData = pair.second;
            
            if (!weightData || !weightData->isValid()) {
                continue;
            }
            
            // Create torch::Tensor from WeightData
            torch::TensorOptions options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(device);
            
            torch::Tensor tensor = torch::from_blob(
                weightData->data.data(),
                weightData->shape,
                options
            ).clone(); // Clone to create a copy that's managed by LibTorch
            
            tensorDict[name] = tensor;
        }
        
        return tensorDict;
    }
    #endif
};

/**
 * @brief Factory class for creating model loaders based on file format
 */
class ModelLoaderFactory {
public:
    /**
     * @brief Create a model loader for the specified model path
     * 
     * Automatically detects the model format based on file extension
     * and creates the appropriate loader instance.
     * 
     * @param modelPath Path to the model file
     * @param config Model configuration
     * @return Unique pointer to the created model loader
     * @throws std::runtime_error if format is unsupported or file doesn't exist
     */
    static std::unique_ptr<IModelLoader> createLoader(
        const std::string& modelPath,
        const ModelConfig& config
    );
    
    /**
     * @brief Detect the format of a model file
     * 
     * @param modelPath Path to the model file
     * @return Detected model format
     */
    static ModelFormat detectFormat(const std::string& modelPath);
    
    /**
     * @brief Check if a model format is supported
     * 
     * @param format Model format to check
     * @return true if supported, false otherwise
     */
    static bool isFormatSupported(ModelFormat format);
    
    /**
     * @brief Get string representation of model format
     * 
     * @param format Model format
     * @return String representation
     */
    static std::string formatToString(ModelFormat format);
};

/**
 * @brief Binary model loader (existing .bin format)
 * 
 * Adapter class that wraps the existing kylin::ModelLoader
 * to implement the IModelLoader interface.
 */
class BinaryModelLoader : public IModelLoader {
public:
    /**
     * @brief Constructor
     * 
     * @param modelPath Path to the .bin model file
     * @param config Model configuration
     */
    BinaryModelLoader(const std::string& modelPath, const ModelConfig& config);
    
    bool load() override;
    
    bool loadWeights(model::ModelWeights& weights, bool loadAll = true) override;
    
    bool loadWeightByName(const std::string& name, model::WeightData& weight) override;
    
    bool hasWeight(const std::string& name) override;
    
    bool loadInto(
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
    ) override;
    
    const ModelConfig& getConfig() const override;
    const std::string& getModelPath() const override;
    kylin::WeightDType getDType() const override;
    
private:
    std::unique_ptr<kylin::ModelLoader> loader_;
};

} // namespace cllm

#endif // CLLM_MODEL_LOADER_INTERFACE_H