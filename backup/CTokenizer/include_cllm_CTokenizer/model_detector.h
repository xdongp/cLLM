#pragma once

#include "tokenizer.h"
#include <string>

namespace cllm {

class ModelDetector {
public:
    static ModelType detectModelType(const std::string& configPath);
};

} // namespace cllm