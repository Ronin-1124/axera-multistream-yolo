// End-to-end feasibility test for InferenceEngine
// Run with hardware: ./test_inference_engine <model_file> <image_file>
// Example: ./test_inference_engine ../../models/yolo26n.axmodel ../../data/test_picture/bus.jpg

#include "app/inference_engine.hpp"
#include <axcl.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_file> <image_file>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    // 1. Load and validate image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        fprintf(stderr, "FAIL: could not read image: %s\n", image_path);
        return 1;
    }
    printf("PASS: loaded image %dx%d\n", frame.cols, frame.rows);

    // 2. Initialize AXCL
    axclError ret = axclInit(nullptr);
    if (ret != 0) { fprintf(stderr, "FAIL: axclInit: 0x%x\n", ret); return 1; }
    printf("PASS: axclInit\n");

    axclrtDeviceList lst;
    ret = axclrtGetDeviceList(&lst);
    if (ret != 0 || lst.num == 0) { fprintf(stderr, "FAIL: GetDeviceList\n"); return 1; }
    ret = axclrtSetDevice(lst.devices[0]);
    if (ret != 0) { fprintf(stderr, "FAIL: SetDevice\n"); return 1; }
    printf("PASS: set device %d\n", lst.devices[0]);

    ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (ret != 0) { fprintf(stderr, "FAIL: EngineInit: 0x%x\n", ret); return 1; }
    printf("PASS: axclrtEngineInit\n");

    // 3. Create and init InferenceEngine
    app::InferenceEngine engine(lst.devices[0]);
    app::InferenceEngine::Config cfg;
    cfg.model_path = model_path;
    cfg.input_h = 640;
    cfg.input_w = 640;
    cfg.conf_thresh = 0.45f;
    cfg.iou_thresh  = 0.45f;
    cfg.num_classes = 80;

    ret = engine.init(cfg);
    if (ret != 0) {
        fprintf(stderr, "FAIL: engine.init failed: %d\n", ret);
        axclFinalize();
        return 1;
    }
    printf("PASS: InferenceEngine initialized\n");

    // 4. Run detection
    std::vector<app::detail::DetectedObject> objects;
    ret = engine.detect(frame, objects);

    if (ret != 0) {
        fprintf(stderr, "FAIL: engine.detect failed: %d\n", ret);
        axclFinalize();
        return 1;
    }
    printf("PASS: detection done\n");
    printf("PASS: detected %zu objects\n", objects.size());

    // 5. Print detections
    for (const auto& obj : objects) {
        const auto& name = app::COCO_CLASS_NAMES[obj.label];
        printf("  [%2d] %s: %.2f%%  (x=%.1f, y=%.1f, w=%.1f, h=%.1f)\n",
               obj.label, name.c_str(), obj.prob * 100,
               obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }

    // 6. Draw and save
    cv::Mat result = frame.clone();
    app::detail::draw_detections(result, objects, app::COCO_CLASS_NAMES);
    cv::imwrite("/tmp/inference_engine_out.jpg", result);
    printf("PASS: result saved to /tmp/inference_engine_out.jpg\n");

    printf("\nAll tests passed!\n");
    axclFinalize();
    return 0;
}
