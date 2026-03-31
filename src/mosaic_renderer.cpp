#include "app/mosaic_renderer.hpp"
#include "app/detail/postprocess.hpp"
#include <opencv2/opencv.hpp>

namespace app {

MosaicRenderer::MosaicRenderer(int num_streams, int cols, int cell_w, int cell_h)
    : num_streams_(num_streams),
      cols_(cols),
      cell_w_(cell_w),
      cell_h_(cell_h),
      rows_((num_streams + cols - 1) / cols) {

    int mosaic_w = cell_w_ * cols_;
    int mosaic_h = cell_h_ * rows_;
    mosaic_.create(mosaic_h, mosaic_w, CV_8UC3);
    mosaic_.setTo(cv::Scalar(20, 20, 20)); // dark background
}

void MosaicRenderer::render(
    const std::vector<const cv::Mat*>& stream_frames,
    const std::vector<std::vector<detail::DetectedObject>>& detections,
    const std::vector<float>& stream_fps,
    const std::vector<std::string>& class_names) {

    // Clear background
    mosaic_.setTo(cv::Scalar(20, 20, 20));

    for (int i = 0; i < num_streams_; ++i) {
        int col = i % cols_;
        int row = i / cols_;
        int cell_x = col * cell_w_;
        int cell_y = row * cell_h_;

        const cv::Mat* frame = stream_frames[i];
        if (frame && !frame->empty()) {
            // Compute letterbox placement: scale to fit cell while preserving aspect ratio
            float scale = std::min(
                static_cast<float>(cell_w_) / frame->cols,
                static_cast<float>(cell_h_) / frame->rows);
            int draw_w = static_cast<int>(frame->cols * scale);
            int draw_h = static_cast<int>(frame->rows * scale);
            int draw_x = cell_x + (cell_w_ - draw_w) / 2;  // center horizontally
            int draw_y = cell_y + (cell_h_ - draw_h) / 2;  // center vertically

            // First fill the cell area with black (letterbox bars)
            cv::rectangle(mosaic_, cv::Rect(cell_x, cell_y, cell_w_, cell_h_),
                         cv::Scalar(0, 0, 0), cv::FILLED);

            // Resize frame to fit the scaled size and copy into mosaic
            cv::Mat resized;
            cv::resize(*frame, resized, cv::Size(draw_w, draw_h));
            resized.copyTo(mosaic_(cv::Rect(draw_x, draw_y, draw_w, draw_h)));

            // Scale detection boxes from frame space to drawn image space
            const auto& dets = (i < static_cast<int>(detections.size()))
                                   ? detections[i]
                                   : std::vector<detail::DetectedObject>{};
            for (const auto& obj : dets) {
                cv::Rect_<float> r = obj.rect;
                int x1 = static_cast<int>(r.x * scale) + draw_x;
                int y1 = static_cast<int>(r.y * scale) + draw_y;
                int x2 = static_cast<int>((r.x + r.width) * scale) + draw_x;
                int y2 = static_cast<int>((r.y + r.height) * scale) + draw_y;

                // Clip to drawn area bounds
                x1 = std::max(draw_x, std::min(draw_x + draw_w - 1, x1));
                y1 = std::max(draw_y, std::min(draw_y + draw_h - 1, y1));
                x2 = std::max(draw_x, std::min(draw_x + draw_w - 1, x2));
                y2 = std::max(draw_y, std::min(draw_y + draw_h - 1, y2));

                static const cv::Scalar kColors[] = {
                    cv::Scalar(0, 255, 0),   cv::Scalar(0, 0, 255),
                    cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 255),
                    cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
                };
                cv::Scalar color = kColors[obj.label % 6];

                cv::rectangle(mosaic_, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                std::string label;
                if (obj.label < static_cast<int>(class_names.size())) {
                    label = class_names[obj.label] + " " + std::to_string(int(obj.prob * 100)) + "%";
                } else {
                    label = "obj " + std::to_string(obj.label) + " " + std::to_string(int(obj.prob * 100)) + "%";
                }
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::rectangle(mosaic_,
                             cv::Point(x1, y1 - text_size.height - 4),
                             cv::Point(x1 + text_size.width + 4, y1),
                             color, cv::FILLED);
                cv::putText(mosaic_, label, cv::Point(x1 + 2, y1 - 2),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }
        } else {
            // Draw placeholder
            cv::rectangle(mosaic_, cv::Rect(cell_x, cell_y, cell_w_, cell_h_),
                          cv::Scalar(60, 60, 60), cv::FILLED);
            cv::putText(mosaic_, "No Signal", cv::Point(cell_x + 10, cell_y + cell_h_ / 2),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(150, 150, 150), 2);
        }

        // Cell border
        cv::rectangle(mosaic_, cv::Rect(cell_x, cell_y, cell_w_, cell_h_),
                     cv::Scalar(80, 80, 80), 1);

        // Stream label + FPS
        float fps = (i < static_cast<int>(stream_fps.size())) ? stream_fps[i] : 0.f;
        std::string label = "Stream " + std::to_string(i);
        std::string fps_label = fps > 0 ? std::to_string(static_cast<int>(fps)) + " FPS" : "";

        // Background for stream label
        int label_w = static_cast<int>(label.length() * 10) + 10;
        if (!fps_label.empty()) {
            label_w += static_cast<int>(fps_label.length() * 10) + 20;  // +20 for extra spacing
        }
        cv::rectangle(mosaic_, cv::Rect(cell_x, cell_y, label_w, 22),
                     cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(mosaic_, label, cv::Point(cell_x + 5, cell_y + 16),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        // FPS label next to stream label
        if (!fps_label.empty()) {
            cv::putText(mosaic_, fps_label,
                       cv::Point(cell_x + 5 + static_cast<int>(label.length() * 10) + 20, cell_y + 16),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(150, 255, 150), 1);
        }
    }
}

} // namespace app
