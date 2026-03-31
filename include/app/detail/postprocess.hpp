#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "app/detail/axcl_model.hpp"

namespace app {
namespace detail {

struct DetectedObject {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// Letterbox resize: fit image to target size with black borders.
void letterbox_resize(const cv::Mat& src, cv::Mat& dst, int target_h, int target_w);

// YOLO26 post-processing: proposals from 6 output tensors → NMS → objects.
// Coordinates in output objects are in original image space (src_rows × src_cols).
void postprocess_yolo26(const std::vector<TensorInfo>& outputs,
                        int src_rows, int src_cols,
                        int letterbox_rows, int letterbox_cols,
                        float conf_thresh, float iou_thresh,
                        std::vector<DetectedObject>& objects,
                        int num_classes = 80);

// Draw bounding boxes on image.
void draw_detections(cv::Mat& mat,
                    const std::vector<DetectedObject>& objects,
                    const std::vector<std::string>& class_names);

} // namespace detail
} // namespace app

#pragma GCC diagnostic pop
