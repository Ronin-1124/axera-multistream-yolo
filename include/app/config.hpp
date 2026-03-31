#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace app {

struct StreamConfig {
    int id = 0;
    std::string url;
    bool enabled = true;
};

struct ModelConfig {
    std::string path;
    int input_h = 640;
    int input_w = 640;
    std::string type = "yolo26";  // yolo26 | yolov8 | yolov5
};

struct InferenceConfig {
    int num_threads = 0;      // 0 = auto (1 per stream)
    float conf_thresh = 0.45f;
    float iou_thresh = 0.45f;
};

struct OutputConfig {
    bool display = true;
    int mosaic_cols = 2;
    int queue_depth = 2;  // per-stream frame queue capacity
    int display_w = 1920;  // mosaic canvas width
    int display_h = 720;   // mosaic canvas height
};

struct AppConfig {
    std::vector<StreamConfig> streams;
    ModelConfig model;
    InferenceConfig inference;
    OutputConfig output;
};

// Parse config from JSON file.
AppConfig load_config(const std::string& json_path);

} // namespace app
