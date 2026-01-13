#include <iostream>
#include <thread>
#include <chrono>
#include "third_party/BS_thread_pool.hpp"

int main() {
    std::cout << "Creating thread pool with 4 threads..." << std::endl;
    BS::thread_pool<BS::tp::pause> pool(4);
    
    std::cout << "Thread count: " << pool.get_thread_count() << std::endl;
    std::cout << "Tasks total: " << pool.get_tasks_total() << std::endl;
    std::cout << "Tasks running: " << pool.get_tasks_running() << std::endl;
    std::cout << "Tasks queued: " << pool.get_tasks_queued() << std::endl;
    std::cout << "Is paused: " << pool.is_paused() << std::endl;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "\nAfter 100ms sleep:" << std::endl;
    std::cout << "Tasks total: " << pool.get_tasks_total() << std::endl;
    std::cout << "Tasks running: " << pool.get_tasks_running() << std::endl;
    std::cout << "Tasks queued: " << pool.get_tasks_queued() << std::endl;
    
    return 0;
}
