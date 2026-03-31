#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace app {

// Decodes a video/RTSP stream using FFmpeg.
// Runs in its own thread, outputs decoded frames via callback.
class StreamSource {
public:
    using FrameCallback = std::function<void(int stream_id, cv::Mat frame, int64_t timestamp_us)>;

    StreamSource(int stream_id, const std::string& url, FrameCallback callback);
    ~StreamSource();

    void start();   // spawn decode thread
    void stop();    // signal stop and join thread
    bool ok() const;  // false if decode error occurred

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace app
