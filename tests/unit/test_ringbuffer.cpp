// test_ringbuffer.cpp — SPSC RingBuffer unit tests (no hardware required)
#include <app/detail/ringbuffer.hpp>
#include <vector>
#include <thread>
#include <cassert>

int main() {
    app::detail::RingBuffer<int> rb(16);

    // basic push/pop
    assert(rb.empty());
    assert(rb.push(42));
    assert(!rb.empty());
    assert(rb.pop() == 42);
    assert(rb.empty());

    // overwrite when full
    for (int i = 0; i < 16; ++i) assert(rb.push(i));
    assert(!rb.push(999)); // full, should fail

    // SPSC stress test
    app::detail::RingBuffer<long long> rb2(1024);
    const int N = 100000;
    std::thread producer([&]() {
        for (long long i = 0; i < N; ++i) {
            while (!rb2.push(i)) {}
        }
    });
    long long sum = 0;
    long long cnt = 0;
    std::thread consumer([&]() {
        while (cnt < N) {
            auto v = rb2.pop();
            if (v.has_value()) { sum += *v; ++cnt; }
        }
    });
    producer.join();
    consumer.join();
    assert(cnt == N);
    assert(sum == (long long)N * (N - 1) / 2);

    return 0;
}
