# KV缓存模块设计

## 编程规范

本模块的编码实现遵循以下规范和约定：
- [C++编程规范.md](../../C++编程规范.md)：定义编码风格、命名规范等
- [生成代码规范.md](../生成代码规范.md)：定义代码生成流程、设计文档一致性要求、优化同步机制等

## 0. 要生成的文件

### 0.1 头文件（include/cllm/kv_cache/）

根据[C++编程规范.md](../../C++编程规范.md)的命名规范，本模块需要生成以下头文件：

| 文件名 | 对应类/结构体 | 说明 |
|--------|--------------|------|
| `cache.h` | `KVCache` | KV缓存主类，存储和管理键值对缓存 |
| `entry.h` | `KVCacheEntry` | KV缓存条目结构体 |
| `config.h` | `CacheConfig` | 缓存配置结构体 |
| `stats.h` | `CacheStats` | 缓存统计信息结构体 |
| `manager.h` | `KVCacheManager` | KV缓存管理器 |

### 0.2 源文件（src/kv_cache/）

| 文件名 | 对应头文件 | 说明 |
|--------|-----------|------|
| `cache.cpp` | `cache.h` | KVCache类的实现 |
| `manager.cpp` | `manager.h` | KVCacheManager类的实现 |

### 0.3 测试文件（tests/）

| 文件名 | 测试目标 | 说明 |
|--------|---------|------|
| `test_kv_cache.cpp` | KVCache, KVCacheManager | KV缓存模块的单元测试 |

### 0.4 文件命名规范说明

- **头文件名**：使用小写字母+下划线，与类名对应（大驼峰转小写下划线）
- **源文件名**：与对应头文件名保持一致
- **目录结构**：头文件位于 `include/cllm/kv_cache/`，源文件位于 `src/kv_cache/`
- **一致性原则**：所有文件命名遵循[C++编程规范.md](../../C++编程规范.md)第1.1节

## 1. 模块概述

### 1.1 模块职责
KV缓存模块负责存储和管理模型推理过程中的键值对（Key-Value pairs），通过缓存注意力机制的中间结果来大幅提升推理效率。该模块实现了LRU淘汰策略、内存限制和增量更新等优化功能。

### 1.2 核心功能
- KV缓存存储：存储和检索键值对缓存
- LRU淘汰策略：基于最近最少使用原则淘汰缓存
- 内存管理：监控和限制缓存内存使用
- 增量更新：支持缓存条目的增量更新
- 缓存统计：记录缓存命中率和使用情况
- 并发访问：支持多线程安全访问

### 1.3 设计原则
- 高效性：最大化缓存命中率
- 内存效率：合理控制内存使用
- 线程安全：支持并发访问
- 可扩展性：易于添加新的缓存策略

### 1.4 模块依赖

本模块依赖以下模块：

| 依赖模块 | 依赖类/结构体 | 依赖原因 |
|----------|--------------|----------|
| `memory` | `FloatArray` | KV缓存数据存储 |

**重要**：KV缓存模块使用LRU策略管理缓存，必须配置合适的内存限制。

### 1.5 命名空间

所有类和函数都在 `cllm` 命名空间下：

```cpp
namespace cllm {
    class KVCache { ... };
    class KVCacheManager { ... };
    struct KVCacheEntry { ... };
    struct CacheConfig { ... };
    struct CacheStats { ... };
}
```

## 2. 类设计

### 2.1 KVCache
```cpp
class KVCache {
public:
    explicit KVCache(size_t maxSize = 10, size_t maxMemoryMB = 0);
    ~KVCache();
    
    bool get(size_t sequenceId, KVCacheEntry& entry);
    void put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache);
    void remove(size_t sequenceId);
    
    void clear();
    size_t size() const;
    bool contains(size_t sequenceId) const;
    
    CacheStats getStats() const;
    void resetStats();
    
    float getMemoryUsageMB() const;
    size_t getMaxSize() const;
    size_t getMaxMemoryMB() const;
    
private:
    void evictOldest();
    void ensureMemoryLimit();
    float calculateMemoryUsage(const FloatArray& keyCache, const FloatArray& valueCache);
    
    std::map<size_t, KVCacheEntry> cache_;
    std::list<size_t> accessList_;
    
    size_t maxSize_;
    size_t maxMemoryMB_;
    float memoryUsage_;
    
    mutable std::mutex cacheMutex_;
    
    CacheStats stats_;
};
```

### 2.2 KVCacheEntry
```cpp
struct KVCacheEntry {
    FloatArray keyCache;
    FloatArray valueCache;
    size_t sequenceId;
    size_t lastAccessTime;
    size_t hitCount;
    size_t createdTime;
    size_t lastUpdateTime;
    float memoryUsage;
    size_t accessCount;
    
    KVCacheEntry();
    KVCacheEntry(
        const FloatArray& keyCache,
        const FloatArray& valueCache,
        size_t sequenceId
    );
    
    void updateAccess();
    void updateMemoryUsage();
};
```

### 2.3 CacheStats
```cpp
struct CacheStats {
    size_t hits;
    size_t misses;
    size_t evictions;
    size_t memoryReclaims;
    
    float getHitRate() const;
    void updateHit();
    void updateMiss();
    void updateEviction();
    void updateMemoryReclaim();
    void reset();
    
    std::string toString() const;
};
```

### 2.4 KVCacheManager
```cpp
class KVCacheManager {
public:
    explicit KVCacheManager(const CacheConfig& config);
    ~KVCacheManager();
    
    KVCache* createCache(const std::string& cacheName);
    KVCache* getCache(const std::string& cacheName);
    void removeCache(const std::string& cacheName);
    
    std::vector<std::string> getCacheNames() const;
    size_t getTotalMemoryUsage() const;
    
    ManagerStats getStats() const;
    
private:
    std::map<std::string, KVCache*> caches_;
    CacheConfig config_;
    
    mutable std::mutex managerMutex_;
};
```

### 2.5 CacheConfig
```cpp
struct CacheConfig {
    size_t defaultMaxSize = 10;
    size_t defaultMaxMemoryMB = 0;
    bool enableLRU = true;
    bool enableMemoryLimit = false;
    bool enableStats = true;
    
    float evictionThreshold = 0.9f;
    size_t cleanupInterval = 1000;
};
```

### 2.6 CacheOptimizer
```cpp
class CacheOptimizer {
public:
    explicit CacheOptimizer(KVCache* cache);
    ~CacheOptimizer();
    
    void optimizeCacheLayout();
    void compressCache();
    void defragmentCache();
    
    OptimizerStats getStats() const;
    
private:
    void optimizeMemoryLayout();
    void optimizeAccessPattern();
    
    KVCache* cache_;
    OptimizerStats stats_;
};
```

## 3. 接口设计

### 3.1 KVCache接口
```cpp
class KVCache {
public:
    explicit KVCache(size_t maxSize = 10, size_t maxMemoryMB = 0);
    ~KVCache();
    
    bool get(size_t sequenceId, KVCacheEntry& entry);
    void put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache);
    void remove(size_t sequenceId);
    
    void clear();
    size_t size() const;
    bool contains(size_t sequenceId) const;
    
    CacheStats getStats() const;
    void resetStats();
    
    float getMemoryUsageMB() const;
    size_t getMaxSize() const;
    size_t getMaxMemoryMB() const;
    
private:
    void evictOldest();
    void ensureMemoryLimit();
    float calculateMemoryUsage(const FloatArray& keyCache, const FloatArray& valueCache);
    
    std::map<size_t, KVCacheEntry> cache_;
    std::list<size_t> accessList_;
    
    size_t maxSize_;
    size_t maxMemoryMB_;
    float memoryUsage_;
    
    mutable std::mutex cacheMutex_;
    
    CacheStats stats_;
};
```

### 3.2 KVCacheManager接口
```cpp
class KVCacheManager {
public:
    explicit KVCacheManager(const CacheConfig& config);
    ~KVCacheManager();
    
    KVCache* createCache(const std::string& cacheName);
    KVCache* getCache(const std::string& cacheName);
    void removeCache(const std::string& cacheName);
    
    std::vector<std::string> getCacheNames() const;
    size_t getTotalMemoryUsage() const;
    
    ManagerStats getStats() const;
    
private:
    std::map<std::string, KVCache*> caches_;
    CacheConfig config_;
    
    mutable std::mutex managerMutex_;
};
```

## 4. 算法实现

### 4.1 缓存获取
```cpp
bool KVCache::get(size_t sequenceId, KVCacheEntry& entry) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    auto it = cache_.find(sequenceId);
    if (it != cache_.end()) {
        entry = it->second;
        entry.updateAccess();
        
        accessList_.remove(sequenceId);
        accessList_.push_back(sequenceId);
        
        stats_.updateHit();
        return true;
    }
    
    stats_.updateMiss();
    return false;
}
```

### 4.2 缓存存储
```cpp
void KVCache::put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    float newMemoryUsage = calculateMemoryUsage(keyCache, valueCache);
    
    auto it = cache_.find(sequenceId);
    if (it != cache_.end()) {
        memoryUsage_ -= it->second.memoryUsage;
        it->second = KVCacheEntry(keyCache, valueCache, sequenceId);
        it->second.memoryUsage = newMemoryUsage;
        it->second.updateAccess();
        
        accessList_.remove(sequenceId);
        accessList_.push_back(sequenceId);
    } else {
        ensureMemoryLimit();
        
        if (cache_.size() >= maxSize_) {
            evictOldest();
        }
        
        cache_[sequenceId] = KVCacheEntry(keyCache, valueCache, sequenceId);
        cache_[sequenceId].memoryUsage = newMemoryUsage;
        accessList_.push_back(sequenceId);
    }
    
    memoryUsage_ += newMemoryUsage;
}
```

### 4.3 LRU淘汰
```cpp
void KVCache::evictOldest() {
    if (accessList_.empty()) {
        return;
    }
    
    size_t oldestId = accessList_.front();
    accessList_.pop_front();
    
    auto it = cache_.find(oldestId);
    if (it != cache_.end()) {
        memoryUsage_ -= it->second.memoryUsage;
        cache_.erase(it);
        stats_.updateEviction();
    }
}
```

### 4.4 内存限制
```cpp
void KVCache::ensureMemoryLimit() {
    if (maxMemoryMB_ == 0) {
        return;
    }
    
    while (memoryUsage_ > maxMemoryMB_ && !cache_.empty()) {
        evictOldest();
        stats_.updateMemoryReclaim();
    }
}
```

### 4.5 增量更新
```cpp
void KVCache::updateIncremental(
    size_t sequenceId,
    const FloatArray& newKeyPart,
    const FloatArray& newValuePart
) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    auto it = cache_.find(sequenceId);
    if (it == cache_.end()) {
        return;
    }
    
    KVCacheEntry& entry = it->second;
    
    size_t oldKeySize = entry.keyCache.size();
    size_t newValueSize = newKeyPart.size();
    
    FloatArray updatedKey(oldKeySize + newValueSize);
    FloatArray updatedValue(oldKeySize + newValueSize);
    
    std::copy(entry.keyCache.data(), entry.keyCache.data() + oldKeySize, updatedKey.data());
    std::copy(newKeyPart.data(), newKeyPart.data() + newValueSize, updatedKey.data() + oldKeySize);
    
    std::copy(entry.valueCache.data(), entry.valueCache.data() + oldKeySize, updatedValue.data());
    std::copy(newValuePart.data(), newValuePart.data() + newValueSize, updatedValue.data() + oldKeySize);
    
    memoryUsage_ -= entry.memoryUsage;
    
    entry.keyCache = updatedKey;
    entry.valueCache = updatedValue;
    entry.updateMemoryUsage();
    entry.updateAccess();
    
    memoryUsage_ += entry.memoryUsage;
}
```

## 5. 并发设计

### 5.1 线程安全保证
- 使用互斥锁保护缓存访问
- 使用读写锁优化读多写少场景
- 使用原子变量保证统计数据的原子性

### 5.2 并发访问优化
```cpp
class ConcurrentKVCache {
public:
    explicit ConcurrentKVCache(KVCache* cache);
    ~ConcurrentKVCache();
    
    bool get(size_t sequenceId, KVCacheEntry& entry);
    void put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache);
    
private:
    KVCache* cache_;
    mutable std::shared_mutex cacheMutex_;
};
```

## 6. 内存管理

### 6.1 内存分配策略
- 使用mimalloc进行高效内存分配
- 使用RAII包装器管理动态数组
- 预分配缓存空间减少频繁分配

### 6.2 内存优化
```cpp
class KVCache {
private:
    void optimizeMemoryUsage() {
        if (memoryUsage_ > maxMemoryMB_ * 0.9f) {
            evictOldest();
        }
    }
    
    void compressCache() {
        for (auto& pair : cache_) {
            if (pair.second.hitCount < 2) {
                remove(pair.first);
            }
        }
    }
};
```

## 7. 错误处理

### 7.1 错误类型
```cpp
enum class CacheError {
    CACHE_FULL,
    OUT_OF_MEMORY,
    INVALID_SEQUENCE_ID,
    CACHE_NOT_FOUND
};

class CacheException : public std::runtime_error {
public:
    CacheException(CacheError error, const std::string& message)
        : std::runtime_error(message), error_(error) {}
    
    CacheError getError() const { return error_; }
    
private:
    CacheError error_;
};
```

### 7.2 错误处理策略
- 缓存满时自动淘汰旧条目
- 内存不足时释放缓存
- 使用日志记录错误信息
- 提供错误码供上层处理

## 8. 性能优化

### 8.1 缓存优化
- 使用LRU策略提高命中率
- 预分配缓存空间
- 优化缓存布局

### 8.2 内存优化
- 重用缓存缓冲区
- 压缩不常用缓存
- 定期清理过期缓存

### 8.3 并发优化
- 使用读写锁优化读操作
- 减少锁竞争
- 批量操作优化

## 9. 测试策略

### 9.1 单元测试
```cpp
class KVCacheTest {
public:
    void testGet();
    void testPut();
    void testRemove();
    void testLRUEviction();
    void testMemoryLimit();
    void testIncrementalUpdate();
    void testConcurrency();
    void testStats();
};
```

### 9.2 性能测试
- 测试缓存命中率
- 测试内存使用效率
- 测试并发访问性能
- 测试淘汰策略效果

### 9.3 集成测试
- 与Model Executor集成测试
- 与Scheduler集成测试
- 端到端性能测试

## 10. 使用示例

### 10.1 基本使用
```cpp
KVCache cache(100, 1024);

FloatArray keyCache(1024);
FloatArray valueCache(1024);

cache.put(1, keyCache, valueCache);

KVCacheEntry entry;
if (cache.get(1, entry)) {
    std::cout << "Cache hit!" << std::endl;
}

cache.remove(1);
```

### 10.2 增量更新
```cpp
KVCache cache(100, 1024);

FloatArray initialKey(512);
FloatArray initialValue(512);
cache.put(1, initialKey, initialValue);

FloatArray newKeyPart(256);
FloatArray newValuePart(256);
cache.updateIncremental(1, newKeyPart, newValuePart);
```

### 10.3 缓存管理器
```cpp
CacheConfig config;
config.defaultMaxSize = 100;
config.defaultMaxMemoryMB = 1024;

KVCacheManager manager(config);

KVCache* cache1 = manager.createCache("model1");
KVCache* cache2 = manager.createCache("model2");

cache1->put(1, keyCache, valueCache);
cache2->put(1, keyCache, valueCache);

std::cout << "Total memory: " << manager.getTotalMemoryUsage() << " MB" << std::endl;
```

## 11. 配置参数

### 11.1 缓存配置
```cpp
struct CacheConfig {
    size_t defaultMaxSize = 10;
    size_t defaultMaxMemoryMB = 0;
    bool enableLRU = true;
    bool enableMemoryLimit = false;
    bool enableStats = true;
    
    float evictionThreshold = 0.9f;
    size_t cleanupInterval = 1000;
};
```

### 11.2 性能配置
```cpp
struct PerformanceConfig {
    bool enableCompression = false;
    bool enableDefragmentation = true;
    size_t defragmentationInterval = 10000;
    bool enablePrefetch = true;
    size_t prefetchSize = 10;
};
```

## 12. 监控指标

### 12.1 缓存指标
- 缓存命中率
- 缓存大小
- 内存使用量
- 淘汰次数

### 12.2 性能指标
- 平均访问时间
- 命中延迟
- 未命中延迟
- 吞吐量

## 13. 依赖关系

### 13.1 外部依赖
- C++标准库（std::map, std::list, std::mutex）
- mimalloc（内存管理）
- 日志库（日志记录）

### 13.2 内部依赖
- FloatArray（RAII包装器）
- MemoryMonitor（内存监控）

## 14. 后续优化方向

### 14.1 短期优化
- 实现更智能的淘汰策略
- 添加缓存压缩功能
- 优化内存布局

### 14.2 长期优化
- 实现分布式缓存
- 支持缓存预取
- 添加缓存预测功能
- 实现自适应缓存大小
