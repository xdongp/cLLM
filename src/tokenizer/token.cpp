#include "cllm/tokenizer/token.h"

namespace cllm {

Token::Token() : id_(0), text_(""), score_(1.0f) {
}

Token::Token(int id, const std::string& text, float score)
    : id_(id), text_(text), score_(score) {
}

int Token::getId() const {
    return id_;
}

std::string Token::getText() const {
    return text_;
}

float Token::getScore() const {
    return score_;
}

void Token::setId(int id) {
    id_ = id;
}

void Token::setText(const std::string& text) {
    text_ = text;
}

void Token::setScore(float score) {
    score_ = score;
}

}
