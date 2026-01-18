/**
 * @file feed_forward.cpp
 * @brief Feed-Forward Network (SwiGLU) 的简化实现
 */

#include "cllm/kylin/feed_forward.h"

#include <stdexcept>

namespace cllm {
namespace kylin {

FeedForwardNetwork::FeedForwardNetwork(size_t hiddenSize, size_t intermediateSize)
    : hiddenSize_(hiddenSize)
    , intermediateSize_(intermediateSize) {}

void FeedForwardNetwork::setWeights(
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown
) {
    wGate_ = wGate;
    wUp_ = wUp;
    wDown_ = wDown;
}

Tensor FeedForwardNetwork::forward(const Tensor& input) const {
    // 检查权重是否已设置
    if (wGate_.shape().empty() || wUp_.shape().empty() || wDown_.shape().empty()) {
        throw std::runtime_error("FeedForwardNetwork weights not set");
    }

    const auto& inShape = input.shape();
    if (inShape.size() != 3) {
        throw std::invalid_argument("FeedForwardNetwork::forward expects [batch, seq, hidden]");
    }

    size_t batch = inShape[0];
    size_t seqLen = inShape[1];
    size_t hidden = inShape[2];
    if (hidden != hiddenSize_) {
        throw std::invalid_argument("FeedForwardNetwork: input hidden size mismatch");
    }

    size_t rows = batch * seqLen;

    // 视为 [rows, hidden]
    using namespace kernels;

    // gate = X @ W_gate, up = X @ W_up
    Tensor gate2d({rows, intermediateSize_});
    Tensor up2d({rows, intermediateSize_});

    matmul(input.data(), wGate_.data(), gate2d.data(), rows, intermediateSize_, hiddenSize_);
    matmul(input.data(), wUp_.data(), up2d.data(), rows, intermediateSize_, hiddenSize_);

    // activated = gate * SiLU(up)
    Tensor activated({rows, intermediateSize_});
    Tensor upAct({rows, intermediateSize_});
    silu(up2d.data(), upAct.data(), rows * intermediateSize_);

    for (size_t i = 0; i < rows * intermediateSize_; ++i) {
        activated[i] = gate2d[i] * upAct[i];
    }

    // out2d = activated @ W_down -> [rows, hidden]
    Tensor out2d({rows, hiddenSize_});
    matmul(activated.data(), wDown_.data(), out2d.data(), rows, hiddenSize_, intermediateSize_);

    // 还原为 [batch, seq, hidden]
    Tensor output({batch, seqLen, hiddenSize_});
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t row = b * seqLen + s;
            for (size_t d = 0; d < hiddenSize_; ++d) {
                size_t srcIndex = row * hiddenSize_ + d;
                size_t dstIndex = (b * seqLen + s) * hiddenSize_ + d;
                output[dstIndex] = out2d[srcIndex];
            }
        }
    }

    return output;
}

}  // namespace kylin
}  // namespace cllm
