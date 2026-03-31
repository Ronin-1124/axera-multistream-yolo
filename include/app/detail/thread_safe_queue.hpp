#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <optional>

namespace app {
namespace detail {

// Thread-safe MPSC (Multi-Producer Single Consumer) bounded queue.
// Uses std::mutex + std::condition_variable.
// ThreadSafe: safe for producer and consumer to be different threads.
template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

    // Producer side (thread-safe)
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_push_.wait(lock, [&] { return queue_.size() < max_size_ || closed_; });
        if (closed_) return false;
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    bool try_push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || queue_.size() >= max_size_) {
            ++dropped_count_;
            return false;
        }
        queue_.push(std::move(item));
        cv_pop_.notify_one();
        return true;
    }

    // Consumer side (single thread)
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_pop_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return item;
    }

    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        cv_push_.notify_one();
        return item;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t dropped() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_count_;
    }

    bool empty() const { return size() == 0; }

    // Close the queue: wakes all waiting threads, subsequent ops return false
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!closed_) {
            closed_ = true;
            cv_push_.notify_all();
            cv_pop_.notify_all();
        }
    }

    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

private:
    std::queue<T> queue_;
    const size_t max_size_;
    mutable std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    bool closed_ = false;
    size_t dropped_count_ = 0;
};

} // namespace detail
} // namespace app
