#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace app {
namespace detail {

// Forward declaration (defined in postprocess.hpp)
struct DetectedObject;

} // namespace detail

// Manages mosaic layout and drawing of detections.
class MosaicRenderer {
public:
    MosaicRenderer(int num_streams, int cols, int cell_w, int cell_h);

    // Build mosaic from per-stream frames and detections.
    // stream_frames: vector indexed by stream_id, nullptr = no frame available.
    // stream_fps: per-stream FPS values to overlay on each cell.
    void render(const std::vector<const cv::Mat*>& stream_frames,
                const std::vector<std::vector<detail::DetectedObject>>& detections,
                const std::vector<float>& stream_fps,
                const std::vector<std::string>& class_names);

    const cv::Mat& mosaic() const { return mosaic_; }

private:
    int num_streams_;
    int cols_;
    int cell_w_;
    int cell_h_;
    int rows_;
    cv::Mat mosaic_;
};

} // namespace app
