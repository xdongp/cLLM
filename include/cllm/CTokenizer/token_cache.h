#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <shared_mutex>
#include <optional>

namespace cllm {

// 用于 std::vector<int> 作为 unordered_map key 的哈希函数
struct VectorIntHash {
    std::size_t operator()(const std::vector<int> &v) const noexcept {
        std::size_t seed = v.size();
        for (int x : v) {
            seed ^= static_cast<std::size_t>(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * @brief Token 编解码结果缓存
 *
 * 设计对照文档: docs/modules/CTokenizer分词设计.md 3.3.1 内存缓存设计
 * - encodeCache_: 文本 -> token 序列
 * - decodeCache_: token 序列 -> 文本
 * - 线程安全: 使用 shared_mutex 支持多读单写
 * - 淘汰策略: 简单的 FIFO/LRU 近似策略（按插入顺序淘汰）
 */
class TokenCache {
public:
    explicit TokenCache(std::size_t maxSize = 10000)
        : maxSize_(maxSize) {}

    // 禁止拷贝，允许移动
    TokenCache(const TokenCache &) = delete;
    TokenCache & operator=(const TokenCache &) = delete;
    TokenCache(TokenCache &&) = default;
    TokenCache & operator=(TokenCache &&) = default;

    /**
     * @brief 写入编码缓存: text -> tokens
     */
    void putEncode(const std::string & text, const std::vector<int> & tokens) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (encodeCache_.size() >= maxSize_) {
            // 简单淘汰一条旧记录（这里使用 unordered_map.begin() 近似 LRU）
            encodeCache_.erase(encodeCache_.begin());
        }
        encodeCache_[text] = tokens;
    }

    /**
     * @brief 读取编码缓存
     * @return 存在则返回 tokens，否则 std::nullopt
     */
    std::optional<std::vector<int>> getEncode(const std::string & text) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = encodeCache_.find(text);
        if (it == encodeCache_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    /**
     * @brief 写入解码缓存: tokens -> text
     */
    void putDecode(const std::vector<int> & tokens, const std::string & text) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (decodeCache_.size() >= maxSize_) {
            decodeCache_.erase(decodeCache_.begin());
        }
        decodeCache_[tokens] = text;
    }

    /**
     * @brief 读取解码缓存
     */
    std::optional<std::string> getDecode(const std::vector<int> & tokens) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = decodeCache_.find(tokens);
        if (it == decodeCache_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    /**
     * @brief 清空所有缓存
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        encodeCache_.clear();
        decodeCache_.clear();
    }

    /**
     * @brief 当前缓存条目数（encode + decode）
     */
    std::size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return encodeCache_.size() + decodeCache_.size();
    }
    
    /**
     * @brief 获取最大缓存大小
     */
    size_t maxSize() const {
        return maxSize_;
    }
    
    /**
     * @brief 设置最大缓存大小
     */
    void setMaxSize(size_t newSize) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        maxSize_ = newSize;
        // 如果当前缓存超过新大小，进行淘汰
        while (encodeCache_.size() + decodeCache_.size() > maxSize_ && !encodeCache_.empty()) {
            encodeCache_.erase(encodeCache_.begin());
        }
        while (encodeCache_.size() + decodeCache_.size() > maxSize_ && !decodeCache_.empty()) {
            decodeCache_.erase(decodeCache_.begin());
        }
    }

private:
    std::unordered_map<std::string, std::vector<int>> encodeCache_;
    std::unordered_map<std::vector<int>, std::string, VectorIntHash> decodeCache_;

    mutable std::shared_mutex mutex_;
    std::size_t maxSize_;
};

} // namespace cllm
