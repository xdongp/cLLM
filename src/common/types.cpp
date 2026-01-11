#include "cllm/common/types.h"
#include <cstring>
#include <stdexcept>

namespace cllm {

IntArray::IntArray(size_t initialCapacity)
    : data(nullptr), size(0), capacity(0) {
    if (initialCapacity > 0) {
        reserve(initialCapacity);
    }
}

IntArray::~IntArray() {
    if (data != nullptr) {
        delete[] data;
    }
}

IntArray::IntArray(const IntArray& other)
    : data(nullptr), size(other.size), capacity(other.size) {
    if (capacity > 0) {
        data = new int[capacity];
        std::memcpy(data, other.data, size * sizeof(int));
    }
}

IntArray& IntArray::operator=(const IntArray& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        capacity = other.size;
        if (capacity > 0) {
            data = new int[capacity];
            std::memcpy(data, other.data, size * sizeof(int));
        } else {
            data = nullptr;
        }
    }
    return *this;
}

IntArray::IntArray(IntArray&& other) noexcept
    : data(other.data), size(other.size), capacity(other.capacity) {
    other.data = nullptr;
    other.size = 0;
    other.capacity = 0;
}

IntArray& IntArray::operator=(IntArray&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data = other.data;
        size = other.size;
        capacity = other.capacity;
        other.data = nullptr;
        other.size = 0;
        other.capacity = 0;
    }
    return *this;
}

void IntArray::resize(size_t newSize) {
    if (newSize > capacity) {
        reserve(newSize);
    }
    size = newSize;
}

void IntArray::reserve(size_t newCapacity) {
    if (newCapacity <= capacity) {
        return;
    }
    
    int* newData = new int[newCapacity];
    if (data != nullptr) {
        std::memcpy(newData, data, size * sizeof(int));
        delete[] data;
    }
    data = newData;
    capacity = newCapacity;
}

void IntArray::push_back(int value) {
    if (size >= capacity) {
        reserve(capacity == 0 ? 1 : capacity * 2);
    }
    data[size++] = value;
}

int& IntArray::operator[](size_t index) {
    if (index >= size) {
        throw std::out_of_range("IntArray index out of range");
    }
    return data[index];
}

const int& IntArray::operator[](size_t index) const {
    if (index >= size) {
        throw std::out_of_range("IntArray index out of range");
    }
    return data[index];
}

void IntArray::clear() {
    size = 0;
}

}  // namespace cllm
