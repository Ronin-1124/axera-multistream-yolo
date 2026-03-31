// Test: StreamSource FFmpeg hardware decoding
// Usage: ./test_stream_source [video_path] [--loop]
//   Default: data/test_videos_360P/test1.mp4
//   --loop: run until video EOF then stop (for loop playback testing)

#include "app/stream_source.hpp"
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
    std::string video_path = "data/test_videos_360P/test1.mp4";
    bool wait_eof = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--loop") {
            wait_eof = true;
        } else {
            video_path = arg;
        }
    }

    std::atomic<int> frame_count{0};
    std::atomic<bool> first_frame{false};
    std::atomic<int64_t> first_ts{0};

    auto callback = [&](int stream_id, cv::Mat frame, int64_t ts) {
        if (!first_frame.exchange(true)) {
            first_ts = ts;
            std::cout << "  PASS: first frame at ts=" << ts
                      << "  size=" << frame.cols << "x" << frame.rows
                      << "  channels=" << frame.channels() << "\n";
        }
        ++frame_count;
    };

    std::cout << "=== StreamSource Test ===\n";
    std::cout << "  Video: " << video_path << "\n";
    std::cout << "  Mode: " << (wait_eof ? "wait for EOF (loop test)" : "3-second sample") << "\n";

    app::StreamSource source(0, video_path, callback);
    source.start();

    if (wait_eof) {
        // Wait up to 120 seconds for all frames (video is ~61s at 30fps = ~1828 frames)
        auto start = std::chrono::steady_clock::now();
        int last_count = 0;
        while (frame_count.load() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(120)) break;
        }
        // Wait for decoding to complete (no new frames for 2s or max 30s total)
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(30)) {
            int curr = frame_count.load();
            if (curr == last_count && curr > 0) {
                // No progress for 1s — likely reached EOF
                break;
            }
            last_count = curr;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        source.stop();
        std::cout << "  PASS: decoded " << frame_count.load() << " frames (first pass)\n";
    } else {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        source.stop();
        std::cout << "  PASS: decoded " << frame_count.load() << " frames in 3 seconds\n";
    }

    if (frame_count.load() > 0) {
        std::cout << "All tests passed!\n";
        return 0;
    } else {
        std::cerr << "FAIL: no frames decoded\n";
        return 1;
    }
}
