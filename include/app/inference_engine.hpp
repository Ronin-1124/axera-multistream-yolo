#pragma once

#include "app/detail/axcl_context.hpp"
#include "app/detail/axcl_model.hpp"
#include "app/detail/postprocess.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace app {

// YOLO26 class names (COCO 80 classes)
extern const std::vector<std::string> COCO_CLASS_NAMES;

// High-level inference engine for a single stream.
// Wraps AxclModel + AxclContext + letterbox preprocessing + postprocess.
class InferenceEngine {
public:
    struct Config {
        std::string model_path;
        int input_h = 640;
        int input_w = 640;
        float conf_thresh = 0.45f;
        float iou_thresh  = 0.45f;
        int num_classes = 80;
    };

    explicit InferenceEngine(int device_id = 0);
    ~InferenceEngine();

    // Initialize: load model, prepare IO, create context.
    // Must be called from the inference thread.
    int init(const Config& cfg);

    // Run detection on a frame.
    // Preprocesses → inference → postprocess.
    // Returns detected objects (coordinates in original image space).
    int detect(const cv::Mat& frame, std::vector<detail::DetectedObject>& objects);

    uint64_t engine_handle() const;
    bool isValid() const { return model_ != nullptr && ctx_ != nullptr && ctx_->isValid(); }

    // Expose internal stages for benchmarking
    int preprocess(const cv::Mat& frame, cv::Mat& letterbox);
    int run_inference(const cv::Mat& letterbox, std::vector<detail::TensorInfo>& outputs);
    int postprocess(const std::vector<detail::TensorInfo>& outputs, const cv::Mat& frame,
                    std::vector<detail::DetectedObject>& objects);

    // non-copyable
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

private:
    std::unique_ptr<detail::AxclModel> model_;
    std::unique_ptr<detail::AxclContext> ctx_;
    Config cfg_;
    int device_id_;
    bool initialized_ = false;
};

} // namespace app
