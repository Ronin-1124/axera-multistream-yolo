#include "app/result_collector.hpp"
#include "app/mosaic_renderer.hpp"
#include "app/inference_engine.hpp"
#include "app/pipeline.hpp"
#include <opencv2/opencv.hpp>

namespace app {

ResultCollector::ResultCollector(int num_streams, int mosaic_cols, int display_w, int display_h)
    : num_streams_(num_streams),
      mosaic_cols_(mosaic_cols),
      mosaic_size_(display_w, display_h),
      last_fps_time_(std::chrono::steady_clock::now()) {
    streams_.resize(num_streams);
    last_rendered_versions_.resize(num_streams, 0);
    int rows = (num_streams + mosaic_cols - 1) / mosaic_cols;
    int cell_w = display_w / mosaic_cols;
    int cell_h = display_h / rows;
    renderer_ = std::make_unique<MosaicRenderer>(num_streams, mosaic_cols, cell_w, cell_h);
}

void ResultCollector::push(StreamResult result) {
    int id = result.stream_id;
    if (id < 0 || id >= num_streams_) return;
    {
        std::lock_guard<std::mutex> lock(streams_[id].mtx);
        streams_[id].latest = std::move(result);
        ++streams_[id].frame_version;  // increment version to signal new frame
        ++streams_[id].frame_count;
    }
    update_fps();
}

void ResultCollector::update_fps() {
    static std::mutex fps_mtx;
    std::lock_guard<std::mutex> lock(fps_mtx);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_time_).count();
    if (elapsed < 1) return;

    for (int i = 0; i < num_streams_; ++i) {
        int cnt = 0;
        {
            std::lock_guard<std::mutex> lock(streams_[i].mtx);
            cnt = streams_[i].frame_count;
            streams_[i].frame_count = 0;
        }
        streams_[i].fps = cnt / static_cast<double>(elapsed);
    }

    fprintf(stderr, "[Collector] ");
    for (int i = 0; i < num_streams_; ++i) {
        fprintf(stderr, "S%d:%.1f ", i, streams_[i].fps);
    }
    if (pipeline_) {
        size_t dropped = static_cast<class Pipeline*>(pipeline_)->total_dropped_frames();
        if (dropped > 0) fprintf(stderr, "dropped:%zu ", dropped);
    }
    fprintf(stderr, "fps  (%ds)\n", static_cast<int>(elapsed));
    last_fps_time_ = now;
}

cv::Mat ResultCollector::get_mosaic() {
    std::vector<const cv::Mat*> frames(num_streams_, nullptr);
    std::vector<std::vector<detail::DetectedObject>> detections(num_streams_);
    std::vector<float> stream_fps(num_streams_, 0.f);
    std::vector<uint32_t> current_versions(num_streams_, 0);
    bool any_changed = false;

    // Acquire all stream locks
    for (int i = 0; i < num_streams_; ++i) {
        streams_[i].mtx.lock();
    }

    for (int i = 0; i < num_streams_; ++i) {
        frames[i] = &streams_[i].latest.frame;
        detections[i] = streams_[i].latest.objects;
        stream_fps[i] = streams_[i].fps;
        current_versions[i] = streams_[i].frame_version;
        if (current_versions[i] != last_rendered_versions_[i]) {
            any_changed = true;
        }
    }

    // Only re-render if any stream has a new frame since last render
    if (any_changed) {
        renderer_->render(frames, detections, stream_fps, COCO_CLASS_NAMES);
        cached_mosaic_ = renderer_->mosaic().clone();
        last_rendered_versions_ = current_versions;
    }

    // Release all stream locks
    for (int i = num_streams_ - 1; i >= 0; --i) {
        streams_[i].mtx.unlock();
    }

    return cached_mosaic_.clone();
}

void ResultCollector::set_pipeline(class Pipeline* p) {
    pipeline_ = p;
}

void ResultCollector::stop() {
    // Nothing to do — no background thread
}

} // namespace app
