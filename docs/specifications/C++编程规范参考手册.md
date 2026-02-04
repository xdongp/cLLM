# cLLM C++ 编程规范

## 1. 命名规范

### 1.1 文件命名

#### 1.1.1 目录命名

- **目录名**: 使用小写字母，下划线分隔
- **与模块名称保持一致**
- **避免使用缩写**，除非是广泛接受的缩写
- **示例**:
  - `kv_cache/` (KV缓存模块)
  - `request_queue/` (请求队列模块)
  - `thread_pool/` (线程池模块)
  - `batch/` (批处理模块)
  - `scheduler/` (调度器模块)
  - `tokenizer/` (分词器模块)

#### 1.1.2 头文件命名

- **头文件**: 使用小写字母，下划线分隔，`.h` 扩展名
- **文件名与类名保持一致**（将大驼峰转换为下划线分隔的小写）
- **简洁原则**: 头文件名应尽量简短，不重复目录名
- **前缀规则**: 
  - 如果文件属于特定模块，且类名不包含模块名，则使用模块名作为前缀
  - 如果类名已经包含模块信息（如HttpServer），则不需要额外前缀
- **示例**:
  - `kv_cache.h` (对应 KVCache 类)
  - `kv_cache_manager.h` (对应 KVCacheManager 类)
  - `kv_cache_config.h` (对应 KVCacheConfig 结构体)
  - `kv_cache_stats.h` (对应 KVCacheStats 结构体)
  - `request_queue.h` (对应 RequestQueue 类)
  - `thread_pool_manager.h` (对应 ThreadPoolManager 类)
  - `server.h` (对应 Server 类，在http目录下)
  - `request.h` (对应 Request 类，在http目录下)
  - `response.h` (对应 Response 类，在http目录下)
  - `handler.h` (对应 Handler 类，在http目录下)
  - `memory_monitor.h` (对应 MemoryMonitor 类)

#### 1.1.3 源文件命名

- **源文件**: 使用小写字母，下划线分隔，`.cpp` 扩展名
- **源文件名与头文件名保持一致**
- **示例**:
  - `kv_cache.cpp` (对应 kv_cache.h)
  - `kv_cache_manager.cpp` (对应 kv_cache_manager.h)
  - `request_queue.cpp` (对应 request_queue.h)
  - `thread_pool_manager.cpp` (对应 thread_pool_manager.h)
  - `server.cpp` (对应 server.h)
  - `request.cpp` (对应 request.h)
  - `response.cpp` (对应 response.h)
  - `memory_monitor.cpp` (对应 memory_monitor.h)

#### 1.1.4 测试文件命名

- **测试文件**: 以 `test_` 开头，小写字母，下划线分隔
- **与被测试的模块名保持一致**
- **示例**:
  - `test_kv_cache.cpp` (测试 KVCache 类)
  - `test_request_queue.cpp` (测试 RequestQueue 类)
  - `test_thread_pool_manager.cpp` (测试 ThreadPoolManager 类)
  - `test_http_server.cpp` (测试 HttpServer 类)

#### 1.1.5 特殊情况处理

- **避免命名冲突**: 如果不同模块有相同名称的类，使用模块前缀区分
  - 示例: `scheduler_batch_processor.h` vs `batch_processor.h`
- **缩写使用**: 仅使用广泛接受的缩写，如 `http`, `api`, `url`
  - 示例: `server.h` (在http目录下，对应HttpServer类)
- **一致性**: 同一模块内的所有文件使用相同的前缀策略
  - 示例: `kv_cache.h`, `kv_cache_manager.h`, `kv_cache_config.h`, `kv_cache_stats.h`
- **头文件简洁性**: 
  - 在特定模块目录下，文件名不需要重复模块名
  - 示例: 在`http/`目录下，使用`server.h`而不是`http_server.h`
  - 示例: 在`http/`目录下，使用`request.h`而不是`http_request.h`

### 1.2 类命名

- **类名**: 使用大驼峰命名法（PascalCase）
  - 示例: `ThreadPool`, `RequestQueue`, `BatchManager`, `MemoryMonitor`

- **接口类**: 以 `I` 开头，使用大驼峰命名法
  - 示例: `ITokenizer`, `ISampler`, `IModelExecutor`

### 1.3 函数命名

- **成员函数**: 使用小驼峰命名法（camelCase）
  - 示例: `allocate()`, `getUsedMemory()`, `processBatch()`

- **静态函数**: 使用小驼峰命名法
  - 示例: `getInstance()`, `initialize()`

- **私有函数**: 以 `_` 开头，使用小驼峰命名法
  - 示例: `_workerThread()`, `_validateRequest()`

### 1.4 变量命名

- **成员变量**: 以 `_` 结尾，使用小驼峰命名法
  - 示例: `usedMemory_`, `maxMemoryBytes_`, `requestQueue_`

- **局部变量**: 使用小驼峰命名法
  - 示例: `bufferSize`, `requestId`, `batchSize`

- **常量**: 全大写，下划线分隔
  - 示例: `MAX_BATCH_SIZE`, `DEFAULT_TIMEOUT_MS`, `MEMORY_LIMIT_BYTES`

- **全局变量**: 以 `g_` 开头，使用小驼峰命名法
  - 示例: `g_instance`, `g_config`

### 1.5 类型命名

- **结构体**: 使用大驼峰命名法
  - 示例: `RequestState`, `BatchConfig`, `BufferInfo`

- **枚举类**: 使用大驼峰命名法
  - 示例: `RequestStatus`, `SamplingStrategy`

- **枚举值**: 全大写，下划线分隔
  - 示例: `RequestStatus::PENDING`, `SamplingStrategy::TOP_K`

- **类型别名**: 使用大驼峰命名法，以 `_t` 结尾
  - 示例: `RequestId_t`, `Timestamp_t`, `MemorySize_t`

### 1.6 命名空间

- 使用小写字母，下划线分隔
- 示例: `cllm`, `cllm::scheduler`, `cllm::executor`

## 2. 代码风格

### 2.1 缩进和空格

- **缩进**: 使用4个空格，不使用Tab
- **大括号**: K&R风格，左大括号不换行
  ```cpp
  void function() {
      if (condition) {
          doSomething();
      }
  }
  ```

- **空格**: 运算符前后加空格，逗号后加空格
  ```cpp
  int result = a + b;
  function(arg1, arg2, arg3);
  ```

### 2.2 注释规范

- **文件头注释**: 每个文件开头添加文件说明
  ```cpp
  /**
   * @file http_server.h
   * @brief HTTP服务器模块，处理RESTful API请求
   * @author cLLM Team
   * @date 2024-01-01
   */
  ```

- **类注释**: 类定义前添加类说明
  ```cpp
  /**
   * @brief HTTP服务器类，处理HTTP请求和响应
   */
  class HttpServer {
  };
  ```

- **函数注释**: 函数定义前添加函数说明
  ```cpp
  /**
   * @brief 处理HTTP请求
   * @param request HTTP请求对象
   * @return HTTP响应对象
   */
  HttpResponse handleRequest(const HttpRequest& request);
  ```

- **行内注释**: 使用 `//` 注释，放在代码上方或右侧
  ```cpp
  // 初始化内存监控器
  MemoryMonitor::instance().initialize();

  int result = calculate();  // 计算结果
  ```

### 2.3 代码组织

- **头文件包含顺序**:
  1. 对应的头文件
  2. C标准库
  3. C++标准库
  4. 第三方库
  5. 项目内部头文件

  ```cpp
  #include "http_server.h"  // 对应的头文件

  #include <stdio.h>         // C标准库
  #include <stdlib.h>

  #include <string>          // C++标准库
  #include <vector>

  #include <drogon/drogon.h> // 第三方库

  #include "request.h"        // 项目内部头文件
  #include "response.h"
  ```

- **类成员顺序**:
  1. public成员
  2. protected成员
  3. private成员

  ```cpp
  class Example {
  public:
      Example();
      ~Example();

      void publicMethod();

  protected:
      void protectedMethod();

  private:
      void privateMethod();
      int privateVariable_;
  };
  ```

## 3. C++特性使用规范

### 3.1 智能指针和RAII

- **优先使用RAII包装器**，避免使用智能指针
  ```cpp
  FloatArray array(100);  // 使用RAII包装器
  ```

- **必要时使用unique_ptr**，避免使用shared_ptr
  ```cpp
  std::unique_ptr<Buffer> buffer(new Buffer(size));
  ```

### 3.2 传递参数

- **基本类型**: 按值传递
  ```cpp
  void process(int value, float ratio);
  ```

- **大对象**: 按const引用传递
  ```cpp
  void process(const Request& request);
  ```

- **需要修改**: 按指针或引用传递
  ```cpp
  void modify(Request* request);
  void modify(Request& request);
  ```

### 3.3 返回值

- **基本类型**: 按值返回
  ```cpp
  int calculate();
  ```

- **大对象**: 按移动语义返回
  ```cpp
  std::vector<int> getData();
  ```

- **可能失败**: 使用返回值+参数或异常
  ```cpp
  bool getResult(Result* result);
  ```

### 3.4 const正确性

- **成员函数**: 不修改成员变量的函数标记为const
  ```cpp
  int getValue() const;
  ```

- **参数**: 不修改的参数标记为const
  ```cpp
  void process(const std::string& input);
  ```

- **成员变量**: 不修改的成员变量标记为const
  ```cpp
  const int max_size_;
  ```

### 3.5 异常处理

- **使用异常处理错误**，避免使用错误码
  ```cpp
  try {
      process();
  } catch (const std::exception& e) {
      LOG_ERROR << "Error: " << e.what();
  }
  ```

- **自定义异常类**: 继承std::exception
  ```cpp
  class MemoryException : public std::exception {
  public:
      const char* what() const noexcept override {
          return "Memory limit exceeded";
      }
  };
  ```

## 4. 内存管理规范

### 4.1 内存分配

- **使用mimalloc替代系统malloc/free**
  ```cpp
  #include <mimalloc.h>

  void* ptr = mi_malloc(size);
  mi_free(ptr);
  ```

- **优先使用RAII包装器管理内存**
  ```cpp
  FloatArray array(100);  // 自动管理内存
  ```

### 4.2 内存监控

- **使用MemoryMonitor监控内存使用**
  ```cpp
  MemoryMonitor::instance().set_limit(1024 * 1024 * 1024);  // 1GB
  MemoryMonitor::instance().allocate(1024);
  size_t used = MemoryMonitor::instance().get_used();
  ```

### 4.3 避免内存泄漏

- **确保每个new都有对应的delete**
- **使用RAII自动管理资源**
- **使用智能指针管理动态分配的对象**

## 5. 并发编程规范

### 5.1 线程管理

- **使用ThreadPool管理线程**
  ```cpp
  ThreadPool pool(4);
  auto result = pool.submit([]() { return calculate(); });
  ```

### 5.2 同步原语

- **使用std::mutex保护共享数据**
  ```cpp
  std::mutex mutex_;
  std::lock_guard<std::mutex> lock(mutex_);
  ```

- **使用std::condition_variable进行线程同步**
  ```cpp
  std::condition_variable cv_;
  cv_.wait(lock, []() { return condition; });
  ```

### 5.3 原子操作

- **使用std::atomic进行原子操作**
  ```cpp
  std::atomic<int> counter_;
  counter_.fetch_add(1);
  ```

## 6. 日志规范

### 6.1 日志级别

- **TRACE**: 详细的调试信息
- **DEBUG**: 调试信息
- **INFO**: 一般信息
- **WARN**: 警告信息
- **ERROR**: 错误信息
- **FATAL**: 致命错误

### 6.2 日志使用

- **使用spdlog进行日志记录**
  ```cpp
  #include <spdlog/spdlog.h>

  spdlog::info("Server started on port {}", port);
  spdlog::error("Failed to process request: {}", error);
  ```

## 7. 测试规范

### 7.1 单元测试

- **使用Google Test框架**
- **测试文件命名**: `test_模块名.cpp`
- **测试用例命名**: `TEST(测试套件, 测试用例)`

  ```cpp
  #include <gtest/gtest.h>

  TEST(MemoryMonitorTest, SetLimit) {
      MemoryMonitor::instance().set_limit(1024);
      EXPECT_EQ(MemoryMonitor::instance().get_limit(), 1024);
  }
  ```

### 7.2 测试覆盖

- **核心模块**: 测试覆盖率 > 80%
- **关键路径**: 测试覆盖率 > 90%
- **边界条件**: 必须测试

## 8. 性能优化规范

### 8.1 性能考虑

- **避免不必要的拷贝**
- **使用移动语义**
- **使用const引用传递大对象**
- **预分配内存避免频繁分配**

### 8.2 SIMD优化

- **使用SIMD指令优化数值计算**
- **使用AVX2/AVX-512指令集**
- **确保内存对齐**

## 9. 代码审查规范

### 9.1 审查要点

- **命名规范**: 符合项目命名规范
- **代码风格**: 符合项目代码风格
- **内存管理**: 无内存泄漏
- **并发安全**: 线程安全
- **性能**: 无明显性能问题
- **测试**: 有充分的测试覆盖

### 9.2 审查流程

1. **自审**: 提交前自审代码
2. **同行评审**: 至少一人评审
3. **修改**: 根据评审意见修改
4. **确认**: 评审通过后合并

## 10. 文档规范

### 10.1 代码文档

- **公共API**: 必须有文档注释
- **复杂逻辑**: 必须有注释说明
- **关键算法**: 必须有注释说明

### 10.2 设计文档

- **模块设计**: 每个模块有详细设计文档
- **接口文档**: 公共接口有接口文档
- **架构文档**: 系统架构有架构文档

## 11. 版本控制规范

### 11.1 提交信息

- **格式**: `[类型] 简短描述`
  - `feat`: 新功能
  - `fix`: 修复bug
  - `refactor`: 重构
  - `docs`: 文档更新
  - `test`: 测试
  - `chore`: 构建/工具

  ```bash
  git commit -m "feat: add HTTP server module"
  git commit -m "fix: memory leak in memory monitor"
  ```

### 11.2 分支管理

- **main**: 主分支，稳定版本
- **develop**: 开发分支
- **feature/xxx**: 功能分支
- **bugfix/xxx**: 修复分支

## 12. 安全规范

### 12.1 输入验证

- **验证所有外部输入**
- **防止缓冲区溢出**
- **防止SQL注入**

### 12.2 资源限制

- **限制内存使用**
- **限制文件大小**
- **限制并发连接数**

## 13. 兼容性规范

### 13.1 编译器支持

- **GCC**: 9.0+
- **Clang**: 10.0+
- **MSVC**: 2019+

### 13.2 平台支持

- **Linux**: Ubuntu 20.04+, CentOS 8+
- **macOS**: 10.15+
- **Windows**: Windows 10+

## 14. 工具和依赖

### 14.1 构建工具

- **CMake**: 3.15+
- **Ninja**: 推荐使用

### 14.2 依赖库

- **Drogon**: HTTP服务器框架
- **LibTorch**: PyTorch C++ API
- **优化内核**: 自实现矩阵运算
- **nlohmann/json**: JSON处理
- **Asio**: 异步框架
- **Intel TBB**: 并行计算
- **sentencepiece**: 分词器
- **spdlog**: 日志
- **Google Test**: 测试框架
- **mimalloc**: 内存分配器

## 15. 最佳实践

### 15.1 代码质量

- **保持函数简短**: 单个函数不超过50行
- **保持类简洁**: 单个类不超过500行
- **避免深层嵌套**: 嵌套深度不超过4层
- **提取重复代码**: 避免代码重复

### 15.2 错误处理

- **检查返回值**: 检查所有可能失败的函数返回值
- **处理异常**: 捕获并处理所有可能的异常
- **日志记录**: 记录所有错误和异常

### 15.3 性能优化

- **先正确后优化**: 确保代码正确后再优化
- **测量后优化**: 使用性能分析工具找出瓶颈
- **优化关键路径**: 优化最耗时的代码路径

## 16. 示例代码

### 16.1 完整的类示例

```cpp
/**
 * @file memory_monitor.h
 * @brief 内存监控器，监控和限制内存使用
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <atomic>
#include <functional>

namespace cllm {

/**
 * @brief 内存监控器类
 */
class MemoryMonitor {
public:
    typedef std::function<void(size_t used, size_t limit)> MemoryLimitCallback;
    
    /**
     * @brief 获取单例实例
     * @return MemoryMonitor实例
     */
    static MemoryMonitor& instance();
    
    /**
     * @brief 设置内存限制
     * @param limit_bytes 内存限制（字节）
     */
    void set_limit(size_t limit_bytes);
    
    /**
     * @brief 获取内存限制
     * @return 内存限制（字节）
     */
    size_t get_limit() const;
    
    /**
     * @brief 分配内存
     * @param bytes 分配的字节数
     */
    void allocate(size_t bytes);
    
    /**
     * @brief 释放内存
     * @param bytes 释放的字节数
     */
    void deallocate(size_t bytes);
    
    /**
     * @brief 获取已使用内存
     * @return 已使用内存（字节）
     */
    size_t get_used() const;
    
    /**
     * @brief 获取峰值内存
     * @return 峰值内存（字节）
     */
    size_t get_peak() const;
    
    /**
     * @brief 设置内存超限回调
     * @param callback 回调函数
     */
    void set_limit_callback(MemoryLimitCallback callback);
    
    /**
     * @brief 重置峰值内存
     */
    void reset_peak();
    
private:
    MemoryMonitor();
    
    MemoryMonitor(const MemoryMonitor&) = delete;
    MemoryMonitor& operator=(const MemoryMonitor&) = delete;
    
    std::atomic<size_t> usedMemory_;
    std::atomic<size_t> peakMemory_;
    std::atomic<size_t> memoryLimit_;
    MemoryLimitCallback limitCallback_;
};

}  // namespace cllm
```

### 16.2 完整的实现示例

```cpp
/**
 * @file memory_monitor.cpp
 * @brief 内存监控器实现
 */

#include "memory_monitor.h"
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace cllm {

MemoryMonitor::MemoryMonitor() {
    usedMemory_ = 0;
    peakMemory_ = 0;
    memoryLimit_ = 0;
}

MemoryMonitor& MemoryMonitor::instance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::set_limit(size_t limit_bytes) {
    memoryLimit_.store(limit_bytes);
    spdlog::info("Memory limit set to {} MB", limit_bytes / (1024 * 1024));
}

size_t MemoryMonitor::get_limit() const {
    return memoryLimit_.load();
}

void MemoryMonitor::allocate(size_t bytes) {
    size_t limit = memoryLimit_.load();
    if (limit > 0) {
        size_t used = usedMemory_.load();
        if (used + bytes > limit) {
            if (limitCallback_) {
                limitCallback_(used, limit);
            }
            spdlog::error("Memory limit exceeded: {} + {} > {}", 
                         used, bytes, limit);
            throw std::runtime_error("Memory limit exceeded");
        }
    }
    
    usedMemory_.fetch_add(bytes);
    
    size_t current = usedMemory_.load();
    size_t peak = peakMemory_.load();
    while (current > peak) {
        if (peakMemory_.compare_exchange_weak(peak, current)) {
            break;
        }
    }
}

void MemoryMonitor::deallocate(size_t bytes) {
    usedMemory_.fetch_sub(bytes);
}

size_t MemoryMonitor::get_used() const {
    return usedMemory_.load();
}

size_t MemoryMonitor::get_peak() const {
    return peakMemory_.load();
}

void MemoryMonitor::set_limit_callback(MemoryLimitCallback callback) {
    limitCallback_ = callback;
}

void MemoryMonitor::reset_peak() {
    peakMemory_.store(0);
}

}  // namespace cllm
```

### 16.3 完整的测试示例

```cpp
/**
 * @file test_memory_monitor.cpp
 * @brief 内存监控器测试
 */

#include <gtest/gtest.h>
#include "memory_monitor.h"

using namespace cllm;

TEST(MemoryMonitorTest, SetLimit) {
    MemoryMonitor::instance().set_limit(1024);
    EXPECT_EQ(MemoryMonitor::instance().get_limit(), 1024);
}

TEST(MemoryMonitorTest, AllocateAndDeallocate) {
    MemoryMonitor::instance().set_limit(1024);
    MemoryMonitor::instance().allocate(512);
    EXPECT_EQ(MemoryMonitor::instance().get_used(), 512);
    
    MemoryMonitor::instance().deallocate(256);
    EXPECT_EQ(MemoryMonitor::instance().get_used(), 256);
}

TEST(MemoryMonitorTest, PeakMemory) {
    MemoryMonitor::instance().set_limit(1024);
    MemoryMonitor::instance().reset_peak();
    
    MemoryMonitor::instance().allocate(512);
    EXPECT_EQ(MemoryMonitor::instance().get_peak(), 512);
    
    MemoryMonitor::instance().allocate(256);
    EXPECT_EQ(MemoryMonitor::instance().get_peak(), 768);
    
    MemoryMonitor::instance().deallocate(256);
    EXPECT_EQ(MemoryMonitor::instance().get_peak(), 768);
}

TEST(MemoryMonitorTest, MemoryLimitExceeded) {
    MemoryMonitor::instance().set_limit(1024);
    MemoryMonitor::instance().reset_peak();
    
    EXPECT_THROW({
        MemoryMonitor::instance().allocate(2048);
    }, std::runtime_error);
}
```

## 17. 总结

本编程规范旨在提高代码质量、可维护性和团队协作效率。所有开发人员应严格遵守本规范，并在开发过程中持续改进。

如有疑问或建议，请联系技术负责人。
