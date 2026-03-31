#include "app/pipeline.hpp"
#include "app/config.hpp"
#include <axcl.h>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

volatile std::sig_atomic_t g_running = 1;

void signal_handler(int) {
    g_running = 0;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    // Parse args
    std::string config_path = "configs/app_config.json";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" || arg == "-c") {
            if (i + 1 < argc) config_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --config <path>   Config JSON file (default: configs/app_config.json)\n"
                      << "  --help, -h        Show this help\n";
            return 0;
        }
    }

    // Load config
    app::AppConfig cfg = app::load_config(config_path);
    if (cfg.streams.empty()) {
        std::cerr << "No streams configured. Check config file: " << config_path << "\n";
        return 1;
    }
    std::cerr << "[Main] loaded config: " << cfg.streams.size() << " streams, model=" << cfg.model.path << "\n";

    // Initialize AXCL runtime (once, main thread)
    axclError axcl_ret = axclInit(nullptr);
    if (axcl_ret != 0) {
        std::cerr << "[Main] axclInit failed: 0x" << std::hex << axcl_ret << "\n";
        return 1;
    }
    std::cerr << "[Main] axclInit ok\n";

    // Setup signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Create and start pipeline
    app::Pipeline pipeline(cfg);
    pipeline.start();

    // Display loop (must run in main thread for OpenCV imshow/waitKey)
    bool display = cfg.output.display;
    if (display) {
        int win_w = cfg.output.display_w;
        int win_h = cfg.output.display_h;
        cv::namedWindow("Multistream YOLO26", cv::WINDOW_NORMAL);
        cv::resizeWindow("Multistream YOLO26", win_w, win_h);
    }

    int frame_count = 0;
    auto last_time = std::chrono::steady_clock::now();

    while (g_running && pipeline.ok()) {
        cv::Mat mosaic = pipeline.get_mosaic();

        if (display && !mosaic.empty()) {
            cv::imshow("Multistream YOLO26", mosaic);
        }

        // Handle keyboard (ESC = 27)
        int key = display ? cv::waitKey(1) : 30;
        if (key == 27) {
            std::cerr << "[Main] ESC pressed, stopping...\n";
            break;
        }

        // FPS reporting every 5 seconds
        ++frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        if (elapsed >= 5.0) {
            double fps = frame_count / elapsed;
            std::cerr << "[Main] display FPS: " << fps << "\n";
            frame_count = 0;
            last_time = now;
        }
    }

    // Stop pipeline
    pipeline.stop();

    if (display) {
        cv::destroyAllWindows();
    }

    std::cerr << "[Main] exited cleanly\n";
    return 0;
}
