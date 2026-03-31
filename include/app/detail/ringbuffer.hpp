#pragma once

#include <cstddef>
#include <atomic>
#include <optional>
#include <algorithm>

namespace app {
namespace detail {

// SPSC (Single Producer Single Consumer) lock-free ring buffer.
// T must be trivially copyable.
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buffer_(new T[capacity]), capacity_(capacity) {}

    ~RingBuffer() { delete[] buffer_; }

    // Producer side
    bool push(const T& item) {
        size_t w = write_idx_.load(std::memory_order_relaxed);
        size_t r = read_idx_.load(std::memory_order_acquire);
        if ((w - r) >= capacity_) return false;
        buffer_[w % capacity_] = item;
        write_idx_.store(w + 1, std::memory_order_release);
        return true;
    }

    bool push(T&& item) {
        size_t w = write_idx_.load(std::memory_order_relaxed);
        size_t r = read_idx_.load(std::memory_order_acquire);
        if ((w - r) >= capacity_) return false;
        buffer_[w % capacity_] = std::move(item);
        write_idx_.store(w + 1, std::memory_order_release);
        return true;
    }

    // Consumer side
    std::optional<T> pop() {
        size_t r = read_idx_.load(std::memory_order_relaxed);
        size_t w = write_idx_.load(std::memory_order_acquire);
        if (w == r) return std::nullopt;
        T item = buffer_[r % capacity_];
        read_idx_.store(r + 1, std::memory_order_release);
        return item;
    }

    size_t size() const {
        size_t w = write_idx_.load(std::memory_order_relaxed);
        size_t r = read_idx_.load(std::memory_order_relaxed);
        return w - r;
    }

    bool empty() const { return size() == 0; }

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

private:
    T* const buffer_;
    const size_t capacity_;
    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};
};

} // namespace detail
} // namespace app
