#include "cllm/common/request_state.h"

namespace cllm {

float RequestState::calculatePriority(size_t currentTime) const {
    size_t waitTime = currentTime - arrivalTime;
    size_t promptLength = getPromptLength();
    
    float lengthFactor = 1.0f / (1.0f + promptLength * 0.01f);
    float waitFactor = 1.0f / (1.0f + waitTime * 0.001f);
    
    return lengthFactor * waitFactor;
}

}  // namespace cllm