#include "app/inference_engine.hpp"
#include <axcl.h>
#include <cstring>
#include <stdexcept>

namespace app {

const std::vector<std::string> COCO_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

InferenceEngine::InferenceEngine(int device_id)
    : device_id_(device_id), initialized_(false) {}

InferenceEngine::~InferenceEngine() = default;

int InferenceEngine::init(const Config& cfg) {
    cfg_ = cfg;

    // Bind device context for this thread (required before axclrtEngineCreateContext)
    axclError ret = axclrtSetDevice(device_id_);
    if (ret != 0) {
        fprintf(stderr, "[InferenceEngine] axclrtSetDevice(%d) failed: 0x%x\n", device_id_, ret);
        return -1;
    }

    // Create and bind device context to this thread
    axclrtContext dev_ctx = nullptr;
    ret = axclrtCreateContext(&dev_ctx, device_id_);
    if (ret != 0) {
        fprintf(stderr, "[InferenceEngine] axclrtCreateContext failed: 0x%x\n", ret);
        return -1;
    }
    ret = axclrtSetCurrentContext(dev_ctx);
    if (ret != 0) {
        fprintf(stderr, "[InferenceEngine] axclrtSetCurrentContext failed: 0x%x\n", ret);
        return -1;
    }

    // Initialize engine subsystem (one-time per thread)
    ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (ret != 0) {
        fprintf(stderr, "[InferenceEngine] axclrtEngineInit failed: 0x%x\n", ret);
        return -1;
    }

    model_ = std::make_unique<detail::AxclModel>();

    int err = model_->load(cfg.model_path);
    if (err != 0) {
        fprintf(stderr, "[InferenceEngine] model.load failed: %d\n", err);
        return err;
    }

    err = model_->prepare_io();
    if (err != 0) {
        fprintf(stderr, "[InferenceEngine] model.prepare_io failed: %d\n", err);
        return err;
    }

    ctx_ = std::make_unique<detail::AxclContext>(model_->engine_handle());
    if (!ctx_->isValid()) {
        fprintf(stderr, "[InferenceEngine] context creation failed\n");
        return -1;
    }

    initialized_ = true;
    return 0;
}

int InferenceEngine::detect(const cv::Mat& frame, std::vector<detail::DetectedObject>& objects) {
    if (!initialized_ || !isValid()) return -1;
    if (frame.empty()) return -1;

    cv::Mat letterbox;
    if (preprocess(frame, letterbox) != 0) return -1;

    std::vector<detail::TensorInfo> outputs;
    if (run_inference(letterbox, outputs) != 0) return -1;

    return postprocess(outputs, frame, objects);
}

int InferenceEngine::preprocess(const cv::Mat& frame, cv::Mat& letterbox) {
    if (!initialized_ || !isValid()) return -1;
    if (frame.empty()) return -1;
    detail::letterbox_resize(frame, letterbox, cfg_.input_h, cfg_.input_w);
    return 0;
}

int InferenceEngine::run_inference(const cv::Mat& letterbox, std::vector<detail::TensorInfo>& outputs) {
    if (!initialized_ || !isValid()) return -1;
    if (letterbox.empty()) return -1;
    auto input_info = model_->get_input(0);
    size_t copy_size = std::min(static_cast<size_t>(input_info.size),
                                static_cast<size_t>(letterbox.total() * letterbox.elemSize()));
    axclError ret = model_->inference(ctx_->context_id(), letterbox.data, copy_size, outputs);
    if (ret != 0) {
        return -1;
    }
    return 0;
}

int InferenceEngine::postprocess(const std::vector<detail::TensorInfo>& outputs, const cv::Mat& frame,
                                std::vector<detail::DetectedObject>& objects) {
    if (!initialized_ || !isValid()) return -1;
    if (outputs.empty()) return -1;
    detail::postprocess_yolo26(outputs,
                                frame.rows, frame.cols,
                                cfg_.input_h, cfg_.input_w,
                                cfg_.conf_thresh, cfg_.iou_thresh,
                                objects,
                                cfg_.num_classes);
    return 0;
}

uint64_t InferenceEngine::engine_handle() const {
    return model_ ? model_->engine_handle() : 0;
}

} // namespace app
