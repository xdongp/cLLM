#include <iostream>
#include <cllm/memory/monitor.h>
#include <cllm/thread_pool/manager.h>

using namespace cllm;

int main() {
    std::cout << "cLLM Basic Usage Example" << std::endl;
    std::cout << "==========================" << std::endl;

    // Create MemoryMonitor instance using singleton pattern
    MemoryMonitor& monitor = MemoryMonitor::instance();
    monitor.setLimit(1024ULL * 1024ULL * 1024ULL);  // 1GB limit

    std::cout << "Memory Monitor initialized" << std::endl;
    std::cout << "Used Memory: " << monitor.getUsed() / (1024ULL * 1024ULL) << " MB" << std::endl;
    std::cout << "Memory Limit: " << monitor.getLimit() / (1024ULL * 1024ULL) << " MB" << std::endl;

    // Create ThreadPoolManager instance
    size_t numThreads = 4;
    ThreadPoolManager pool(numThreads);

    std::cout << "Thread Pool initialized with " << numThreads << " threads" << std::endl;
    std::cout << "Thread count: " << pool.getThreadCount() << std::endl;

    std::cout << "Example completed successfully!" << std::endl;
    return 0;
}
