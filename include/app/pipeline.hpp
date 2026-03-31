#pragma once

#include <app/config.hpp>
#include <app/stream_source.hpp>
#include <app/inference_engine.hpp>
#include <app/result_collector.hpp>
#include <app/detail/thread_safe_queue.hpp>
#include <atomic>
#include <list>
#include <vector>
#include <thread>

namespace app {

// Main pipeline orchestrator.
// Manages all StreamWorkers and ResultCollector.
class Pipeline {
public:
    explicit Pipeline(const AppConfig& config);
    ~Pipeline();

    void start();   // start all workers
    void stop();    // stop all workers and join threads

    // Get display mosaic (call from main thread)
    cv::Mat get_mosaic();

    bool ok() const;  // true if all workers healthy
    size_t total_dropped_frames() const;

private:
    struct StreamWorker {
        explicit StreamWorker(size_t queue_depth) : frame_queue(queue_depth) {}
        int stream_id;
        std::unique_ptr<StreamSource> source;
        std::unique_ptr<InferenceEngine> engine;
        std::thread decode_thread;
        std::thread inference_thread;
        detail::ThreadSafeQueue<cv::Mat> frame_queue;  // bounded, capacity set by ctor
    };

    AppConfig config_;
    std::list<std::unique_ptr<StreamWorker>> workers_;
    std::unique_ptr<ResultCollector> collector_;
    std::atomic<bool> running_{false};
};

} // namespace app
