// test_thread_safe_queue.cpp — ThreadSafeQueue unit tests (no hardware required)
#include <app/detail/thread_safe_queue.hpp>
#include <thread>
#include <cassert>
#include <chrono>
#include <mutex>

int main() {
    app::detail::ThreadSafeQueue<int> q(4);

    assert(q.empty());
    q.push(1);
    q.push(2);
    assert(q.size() == 2);
    assert(q.pop() == 1);
    assert(q.pop() == 2);
    assert(q.empty());

    // MPSC test: 4 producers, 1 consumer
    app::detail::ThreadSafeQueue<int> q2(64);
    const int N = 5000;
    int total = 0;
    std::mutex mu;
    std::vector<std::thread> producers;
    for (int t = 0; t < 4; ++t) {
        producers.emplace_back([&, t]() {
            for (int i = 0; i < N; ++i) {
                q2.push(t * N + i);
            }
        });
    }
    std::thread consumer([&]() {
        int received = 0;
        while (received < 4 * N) {
            auto v = q2.pop();
            if (v.has_value()) {
                std::lock_guard<std::mutex> lock(mu);
                total += *v;
                ++received;
            }
        }
    });
    for (auto& p : producers) p.join();
    consumer.join();
    assert(total == (long long)4 * N * (4 * N - 1) / 2);

    return 0;
}
