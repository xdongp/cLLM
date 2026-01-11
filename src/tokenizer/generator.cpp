#include "cllm/tokenizer/generator.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/tokenizer.h"
#include <stdexcept>

namespace cllm {

StreamGenerator::StreamGenerator(
    const std::string& requestId,
    const std::vector<int>& inputIds,
    int maxTokens,
    float temperature,
    ModelExecutor* modelExecutor,
    ITokenizer* tokenizer
) : requestId_(requestId),
    inputIds_(inputIds),
    maxTokens_(maxTokens),
    temperature_(temperature),
    modelExecutor_(modelExecutor),
    tokenizer_(tokenizer),
    finished_(false),
    currentTokenIndex_(0) {
    
    if (modelExecutor_ == nullptr) {
        throw std::runtime_error("ModelExecutor cannot be null");
    }
    
    if (tokenizer_ == nullptr) {
        throw std::runtime_error("Tokenizer cannot be null");
    }
    
    previousText_ = tokenizer_->decode(inputIds_, true);
}

StreamGenerator::~StreamGenerator() {
}

bool StreamGenerator::hasNext() {
    return !finished_ && currentTokenIndex_ < maxTokens_;
}

GenerationResponse StreamGenerator::next() {
    GenerationResponse response;
    response.setRequestId(requestId_);
    
    if (finished_) {
        response.setFinished(true);
        return response;
    }
    
    generateNextToken();
    
    std::string newText = extractNewText();
    response.setText(newText);
    
    if (newText.empty() || currentTokenIndex_ >= maxTokens_) {
        finished_ = true;
        response.setFinished(true);
    } else {
        response.setFinished(false);
    }
    
    response.setTokens(generatedTokens_);
    
    return response;
}

bool StreamGenerator::isFinished() const {
    return finished_;
}

int StreamGenerator::getGeneratedTokenCount() const {
    return static_cast<int>(generatedTokens_.size());
}

void StreamGenerator::generateNextToken() {
    if (finished_) {
        return;
    }
    
    std::vector<int> currentInput = inputIds_;
    currentInput.insert(currentInput.end(), generatedTokens_.begin(), generatedTokens_.end());
    
    int nextToken = modelExecutor_->sampleToken(currentInput, temperature_);
    
    generatedTokens_.push_back(nextToken);
    currentTokenIndex_++;
    
    // isSpecialToken方法在ITokenizer接口中未定义，暂时跳过特殊token检查
    // if (tokenizer_->isSpecialToken(nextToken)) {
    //     finished_ = true;
    // }
}

std::string StreamGenerator::extractNewText() {
    std::string currentText = tokenizer_->decode(generatedTokens_, true);
    
    std::string newText;
    if (currentText.length() > previousText_.length()) {
        newText = currentText.substr(previousText_.length());
    }
    
    previousText_ = currentText;
    
    return newText;
}

}
