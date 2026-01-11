/**
 * @file token.h
 * @brief Token类，表示单个token
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_TOKEN_H
#define CLLM_TOKEN_H

#include <string>

namespace cllm {

/**
 * @brief Token类
 * 
 * 表示单个token，包含ID、文本和分数。
 */
class Token {
public:
    /**
     * @brief 默认构造函数
     */
    Token();
    
    /**
     * @brief 构造函数
     * @param id Token ID
     * @param text Token文本
     * @param score Token分数
     */
    Token(int id, const std::string& text, float score = 1.0f);
    
    /**
     * @brief 获取Token ID
     * @return Token ID
     */
    int getId() const;
    
    /**
     * @brief 获取Token文本
     * @return Token文本
     */
    std::string getText() const;
    
    /**
     * @brief 获取Token分数
     * @return Token分数
     */
    float getScore() const;
    
    /**
     * @brief 设置Token ID
     * @param id Token ID
     */
    void setId(int id);
    
    /**
     * @brief 设置Token文本
     * @param text Token文本
     */
    void setText(const std::string& text);
    
    /**
     * @brief 设置Token分数
     * @param score Token分数
     */
    void setScore(float score);
    
private:
    int id_;            ///< Token ID
    std::string text_;  ///< Token文本
    float score_;       ///< Token分数
};

}

#endif
