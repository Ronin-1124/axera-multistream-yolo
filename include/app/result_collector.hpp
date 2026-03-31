#pragma once

#include <app/mosaic_renderer.hpp>
#include <app/detail/postprocess.hpp>
#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <mutex>
#include <memory>
#include <chrono>

namespace app {

struct StreamResult {
    int stream_id;
    int64_t timestamp_us;
    std::vector<detail::DetectedObject> objects;
    cv::Mat frame;  // original frame (for display)
};

// Collects inference results from all streams and feeds MosaicRenderer.
class ResultCollector {
public:
    ResultCollector(int num_streams, int mosaic_cols, int display_w, int display_h);

    // Called by inference threads (MPSC safe)
    void push(StreamResult result);

    // Called by main thread to get latest combined mosaic
    cv::Mat get_mosaic();

    // Set weak pointer to Pipeline for dropped frame reporting
    void set_pipeline(class Pipeline* p);

    void stop();

private:
    void update_fps();

    struct StreamState {
        StreamResult latest;
        uint32_t frame_version = 0;  // increments each time a new frame arrives
        std::mutex mtx;
        int frame_count = 0;  // for FPS calculation
        float fps = 0.f;     // latest computed FPS for this stream
    };

    int num_streams_;
    std::deque<StreamState> streams_;
    int mosaic_cols_;
    cv::Size mosaic_size_;
    std::unique_ptr<MosaicRenderer> renderer_;
    std::chrono::steady_clock::time_point last_fps_time_;
    std::vector<uint32_t> last_rendered_versions_;  // per-stream version at last render
    cv::Mat cached_mosaic_;  // last rendered mosaic, valid only when versions unchanged
    void* pipeline_ = nullptr;  // weak pointer to Pipeline for dropped frame reporting
};

} // namespace app
