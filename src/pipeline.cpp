#include "app/pipeline.hpp"
#include "app/inference_engine.hpp"
#include "app/stream_source.hpp"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

namespace app {

Pipeline::Pipeline(const AppConfig& config) : config_(config) {
}

Pipeline::~Pipeline() {
    stop();
}

void Pipeline::start() {
    if (running_.load(std::memory_order_acquire)) return;
    running_.store(true, std::memory_order_release);

    const int num_streams = config_.streams.size();

    // Create collector
    int display_w = config_.output.display_w;
    int display_h = config_.output.display_h;
    int cols = config_.output.mosaic_cols > 0 ? config_.output.mosaic_cols : 2;
    collector_ = std::make_unique<ResultCollector>(num_streams, cols, display_w, display_h);
    collector_->set_pipeline(this);

    // Shared config for inference
    auto cfg = std::make_shared<InferenceEngine::Config>();
    cfg->model_path = config_.model.path;
    cfg->input_h = config_.model.input_h;
    cfg->input_w = config_.model.input_w;
    cfg->conf_thresh = config_.inference.conf_thresh;
    cfg->iou_thresh = config_.inference.iou_thresh;

    // Create one StreamWorker per stream
    for (int i = 0; i < num_streams; ++i) {
        const auto& sc = config_.streams[i];
        if (!sc.enabled) continue;

        auto worker = std::make_unique<StreamWorker>(config_.output.queue_depth);
        worker->stream_id = sc.id;

        // Decode thread: runs StreamSource, pushes frames to queue
        worker->decode_thread = std::thread([queue = &worker->frame_queue, stream_id = sc.id, url = sc.url]() {
            StreamSource::FrameCallback frame_cb = [queue](int sid, cv::Mat frame, int64_t ts) {
                queue->try_push(std::move(frame));
            };
            StreamSource src(stream_id, url, frame_cb);
            src.start();
            src.stop();
        });

        // Inference thread: pops frames, runs detection, pushes results
        worker->inference_thread = std::thread([this, queue = &worker->frame_queue,
                                                  stream_id = sc.id, cfg]() {
            InferenceEngine engine(1);
            int err = engine.init(*cfg);
            if (err != 0) {
                fprintf(stderr, "[Pipeline] InferenceEngine init failed for stream %d\n", stream_id);
                queue->close(); // unblock any waiting pops
                return;
            }
            fprintf(stderr, "[Pipeline] Inference thread started for stream %d\n", stream_id);

            while (running_.load(std::memory_order_acquire)) {
                auto opt_frame = queue->pop();
                if (!opt_frame) break; // queue closed

                cv::Mat frame = std::move(*opt_frame);
                std::vector<detail::DetectedObject> objects;
                int ret = engine.detect(frame, objects);

                if (ret == 0) {
                    StreamResult result;
                    result.stream_id = stream_id;
                    result.timestamp_us = 0;
                    result.objects = std::move(objects);
                    result.frame = std::move(frame);  // frame now owned by result
                    collector_->push(std::move(result));
                }
            }
            fprintf(stderr, "[Pipeline] Inference thread stopped for stream %d\n", stream_id);
        });

        workers_.push_back(std::move(worker));
    }

    fprintf(stderr, "[Pipeline] started %zu stream workers\n", workers_.size());
}

void Pipeline::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);

    // 1. Stop StreamSource decode threads first (FFmpeg resources)
    for (auto& w : workers_) {
        if (w->decode_thread.joinable()) {
            // Add timeout to avoid hanging on blocked FFmpeg I/O
            std::thread& t = w->decode_thread;
            std::chrono::milliseconds timeout(500);
            auto start = std::chrono::steady_clock::now();
            while (t.joinable() && std::chrono::steady_clock::now() - start < timeout) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (t.joinable()) {
                fprintf(stderr, "[Pipeline] stop: decode thread timed out, detaching\n");
                t.detach();
            } else {
                t.join();
            }
        }
    }

    // 2. Close frame queues (wakes inference threads' blocking pop())
    for (auto& w : workers_) {
        w->frame_queue.close();
    }

    // 3. Join inference threads
    for (auto& w : workers_) {
        if (w->inference_thread.joinable()) {
            w->inference_thread.join();
        }
    }

    // 4. Clear workers
    workers_.clear();

    // 5. Reset collector
    collector_.reset();
}

cv::Mat Pipeline::get_mosaic() {
    if (collector_) {
        return collector_->get_mosaic();
    }
    return {};
}

bool Pipeline::ok() const {
    return running_.load(std::memory_order_acquire);
}

size_t Pipeline::total_dropped_frames() const {
    size_t total = 0;
    for (const auto& w : workers_) {
        total += w->frame_queue.dropped();
    }
    return total;
}

} // namespace app
