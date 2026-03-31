#include "app/stream_source.hpp"
#include <thread>
#include <chrono>
#include <cstring>
#include <cerrno>
#include <atomic>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace app {

namespace {

// Determine codec type from URL/file extension
bool is_hevc_url(const std::string& url) {
    if (url.find(".265") != std::string::npos ||
        url.find(".hevc") != std::string::npos ||
        url.find("hevc") != std::string::npos) {
        return true;
    }
    return false;
}

// NV12 → BGR conversion using swscale
cv::Mat nv12_to_bgr(const AVFrame* frame) {
    static thread_local struct {
        int w = 0, h = 0;
        SwsContext* ctx = nullptr;
    } cached;

    int w = frame->width;
    int h = frame->height;

    // Only recreate sws context if dimensions changed
    if (cached.w != w || cached.h != h) {
        if (cached.ctx) {
            sws_freeContext(cached.ctx);
        }
        cached.ctx = sws_getContext(w, h, AV_PIX_FMT_NV12,
                                    w, h, AV_PIX_FMT_BGR24,
                                    SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!cached.ctx) return {};
        cached.w = w;
        cached.h = h;
    }

    // BGR output buffer (24-bit, no padding)
    cv::Mat bgr(h, w, CV_8UC3);
    const uint8_t* const srcSlice[] = { frame->data[0], frame->data[1], nullptr, nullptr };
    int srcStride[] = { frame->linesize[0], frame->linesize[1], 0, 0 };
    uint8_t* dst[] = { bgr.data, nullptr, nullptr, nullptr };
    int dstStride[] = { static_cast<int>(bgr.step[0]), 0, 0, 0 };

    sws_scale(cached.ctx, srcSlice, srcStride, 0, h, dst, dstStride);
    return bgr;
}

} // anonymous namespace

struct StreamSource::Impl {
    int stream_id_;
    std::string url_;
    FrameCallback callback_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    // FFmpeg resources
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* dec_ctx_  = nullptr;
    AVPacket* pkt_            = nullptr;
    AVFrame* frame_           = nullptr;
    int video_stream_idx_     = -1;

    Impl(int stream_id, const std::string& url, FrameCallback cb)
        : stream_id_(stream_id), url_(url), callback_(std::move(cb)) {}

    ~Impl() { stop(); }

    bool open() {
        const char* decoder_name = is_hevc_url(url_) ? "hevc_axdec" : "h264_axdec";

        // 1. Open format context
        fmt_ctx_ = avformat_alloc_context();
        if (!fmt_ctx_) {
            fprintf(stderr, "[StreamSource:%d] avformat_alloc_context failed\n", stream_id_);
            return false;
        }

        // Set RTSP options for TCP transport
        AVDictionary* opts = nullptr;
        if (url_.rfind("rtsp://", 0) == 0) {
            av_dict_set(&opts, "rtsp_transport", "tcp", 0);
            av_dict_set(&opts, "buffer_size", "1024000", 0);   // 1MB buffer
            av_dict_set(&opts, "stimeout", "5000000", 0);       // 5s socket timeout (us)
            av_dict_set(&opts, "max_delay", "500000", 0);       // 0.5s max demux delay
        }

        int ret = avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (ret < 0) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            fprintf(stderr, "[StreamSource:%d] avformat_open_input failed: %s (%d)\n",
                    stream_id_, errbuf, ret);
            return false;
        }

        // 2. Find stream info
        ret = avformat_find_stream_info(fmt_ctx_, nullptr);
        if (ret < 0) {
            fprintf(stderr, "[StreamSource:%d] avformat_find_stream_info failed: %d\n",
                    stream_id_, ret);
            return false;
        }

        // 3. Find video stream
        int video_idx = -1;
        for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
            if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                video_idx = i;
                break;
            }
        }
        if (video_idx < 0) {
            fprintf(stderr, "[StreamSource:%d] no video stream found\n", stream_id_);
            return false;
        }

        AVStream* video_stream = fmt_ctx_->streams[video_idx];
        video_stream_idx_ = video_idx;

        // 4. Find hardware decoder
        const AVCodec* dec = avcodec_find_decoder_by_name(decoder_name);
        if (!dec) {
            fprintf(stderr, "[StreamSource:%d] decoder '%s' not found\n", stream_id_, decoder_name);
            return false;
        }

        dec_ctx_ = avcodec_alloc_context3(dec);
        if (!dec_ctx_) {
            fprintf(stderr, "[StreamSource:%d] avcodec_alloc_context3 failed\n", stream_id_);
            return false;
        }

        ret = avcodec_parameters_to_context(dec_ctx_, video_stream->codecpar);
        if (ret < 0) {
            fprintf(stderr, "[StreamSource:%d] avcodec_parameters_to_context failed: %d\n",
                    stream_id_, ret);
            return false;
        }

        // Hardware device type — AXCL decoder outputs to CUDA device frames by default,
        // but h264_axdec outputs to system memory (NV12) when used without hw_frames_ctx.
        // Force output to system memory.
        dec_ctx_->opaque = nullptr;

        ret = avcodec_open2(dec_ctx_, dec, nullptr);
        if (ret < 0) {
            char errbuf[128];
            av_strerror(ret, errbuf, sizeof(errbuf));
            fprintf(stderr, "[StreamSource:%d] avcodec_open2('%s') failed: %s (%d)\n",
                    stream_id_, decoder_name, errbuf, ret);
            return false;
        }

        // 5. Allocate packet & frame
        pkt_   = av_packet_alloc();
        frame_ = av_frame_alloc();
        if (!pkt_ || !frame_) {
            fprintf(stderr, "[StreamSource:%d] av_packet/frame_alloc failed\n", stream_id_);
            return false;
        }

        fprintf(stderr, "[StreamSource:%d] opened %s with decoder %s (%dx%d, fps=%.2f)\n",
                stream_id_, url_.c_str(), decoder_name,
                dec_ctx_->width, dec_ctx_->height,
                av_q2d(video_stream->avg_frame_rate));
        return true;
    }

    void close() {
        if (frame_)  { av_frame_free(&frame_); }
        if (pkt_)    { av_packet_free(&pkt_); }
        if (dec_ctx_) { avcodec_free_context(&dec_ctx_); }
        if (fmt_ctx_) { avformat_close_input(&fmt_ctx_); }
    }

    void run_loop() {
        running_ = true;
        int ret;

        while (running_.load(std::memory_order_relaxed)) {
            // 1. Read packet
            ret = av_read_frame(fmt_ctx_, pkt_);
            if (ret < 0) {
                if (ret == AVERROR_EOF || avio_feof(fmt_ctx_->pb)) {
                    if (url_.rfind("rtsp://", 0) == 0) {
                        break; // RTSP: end of stream
                    }
                    // Local file: close and fully reopen to reset HW decoder state.
                    if (frame_)  { av_frame_free(&frame_); frame_ = nullptr; }
                    if (pkt_)    { av_packet_free(&pkt_); pkt_ = nullptr; }
                    if (dec_ctx_) { avcodec_free_context(&dec_ctx_); dec_ctx_ = nullptr; }
                    if (fmt_ctx_) { avformat_close_input(&fmt_ctx_); fmt_ctx_ = nullptr; }

                    AVDictionary* opts = nullptr;
                    ret = avformat_open_input(&fmt_ctx_, url_.c_str(), nullptr, &opts);
                    av_dict_free(&opts);
                    if (ret < 0) {
                        char errbuf[128];
                        av_strerror(ret, errbuf, sizeof(errbuf));
                        fprintf(stderr, "[StreamSource:%d] reopen avformat_open_input failed: %s\n", stream_id_, errbuf);
                        break;
                    }
                    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
                    if (ret < 0) {
                        fprintf(stderr, "[StreamSource:%d] reopen avformat_find_stream_info failed: %d\n", stream_id_, ret);
                        break;
                    }

                    video_stream_idx_ = -1;
                    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
                        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                            video_stream_idx_ = i;
                            break;
                        }
                    }
                    if (video_stream_idx_ < 0) {
                        fprintf(stderr, "[StreamSource:%d] reopen no video stream found\n", stream_id_);
                        break;
                    }

                    AVStream* video_stream = fmt_ctx_->streams[video_stream_idx_];
                    const char* decoder_name = is_hevc_url(url_) ? "hevc_axdec" : "h264_axdec";
                    const AVCodec* dec = avcodec_find_decoder_by_name(decoder_name);
                    if (!dec) {
                        fprintf(stderr, "[StreamSource:%d] reopen decoder '%s' not found\n", stream_id_, decoder_name);
                        break;
                    }
                    dec_ctx_ = avcodec_alloc_context3(dec);
                    if (!dec_ctx_) {
                        fprintf(stderr, "[StreamSource:%d] reopen avcodec_alloc_context3 failed\n", stream_id_);
                        break;
                    }
                    ret = avcodec_parameters_to_context(dec_ctx_, video_stream->codecpar);
                    if (ret < 0) {
                        fprintf(stderr, "[StreamSource:%d] reopen avcodec_parameters_to_context failed: %d\n", stream_id_, ret);
                        break;
                    }
                    ret = avcodec_open2(dec_ctx_, dec, nullptr);
                    if (ret < 0) {
                        fprintf(stderr, "[StreamSource:%d] reopen avcodec_open2 failed: %d\n", stream_id_, ret);
                        break;
                    }
                    pkt_   = av_packet_alloc();
                    frame_ = av_frame_alloc();
                    continue;
                }
                break; // Unexpected error
            }

            // 2. Send to decoder
            if (pkt_->stream_index != -1) {
                ret = avcodec_send_packet(dec_ctx_, pkt_);
                if (ret < 0 && ret != AVERROR(EAGAIN) && ret != -EAGAIN) {
                    // Non-EAGAIN send error — log and continue
                }
            }
            av_packet_unref(pkt_);

            // 3. Drain all decoded frames from the decoder.
            // h264_axdec may produce frames from previously buffered packets,
            // so we drain until EAGAIN. EAGAIN can be POSIX=11 (=-11) or
            // FFmpeg AVERROR(EAGAIN)=FFERRTAG('E','O','F',' ').
            while (running_.load(std::memory_order_relaxed)) {
                ret = avcodec_receive_frame(dec_ctx_, frame_);
                if (ret == AVERROR(EAGAIN) || ret == -EAGAIN) {
                    break;
                }
                if (ret == AVERROR_EOF) {
                    running_.store(false, std::memory_order_relaxed);
                    break;
                }
                if (ret < 0) {
                    running_.store(false, std::memory_order_relaxed);
                    break;
                }

                cv::Mat bgr = nv12_to_bgr(frame_);
                if (!bgr.empty()) {
                    int64_t ts = (frame_->pts != AV_NOPTS_VALUE)
                                     ? frame_->pts
                                     : frame_->best_effort_timestamp;
                    callback_(stream_id_, bgr, ts);
                }
                av_frame_unref(frame_);
            }
        }

        running_.store(false, std::memory_order_relaxed);
    }

    void start() {
        if (running_) return;
        thread_ = std::thread([this]() { run_loop(); });
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        close();
    }

    bool ok() const { return running_; }
};

StreamSource::StreamSource(int stream_id, const std::string& url, FrameCallback callback)
    : impl_(std::make_unique<Impl>(stream_id, url, std::move(callback))) {}

StreamSource::~StreamSource() = default;

void StreamSource::start() {
    if (!impl_->open()) return;
    impl_->start();
}

void StreamSource::stop() {
    impl_->stop();
}

bool StreamSource::ok() const { return impl_ && impl_->ok(); }

} // namespace app
