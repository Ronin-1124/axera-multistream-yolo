// NPU Performance Benchmark Tool
// ================================
// Comprehensive AX650N inference performance analysis.
//
// Modes:
//   ./npu_bench <model> <image> --stages [iters]   stage breakdown (pre/inf/post)
//   ./npu_bench <model> <image> --sweep <N> <iters>  1..N thread sweep
//   ./npu_bench <model> <image> --mt <N> <iters>     N-thread throughput test
//   ./npu_bench <model> <image>                      feasibility test (default)
//
// Examples:
//   ./npu_bench ../../models/yolo26n.axmodel ./data/test_picture/bus.jpg --stages 100
//   ./npu_bench ../../models/yolo26n.axmodel ./data/test_picture/bus.jpg --sweep 40 100
//   ./npu_bench ../../models/yolo26n.axmodel ./data/test_picture/bus.jpg --mt 4 200

#include "app/inference_engine.hpp"
#include "ax_model_runner_axcl.hpp"
#include "base/detection.hpp"
#include <axcl.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <numeric>

// ============================================================================
// Stage breakdown mode (--stages) — SDK-level H2D / D2H / NPU split
// ============================================================================
// Uses ax_runner_axcl directly to get independent timing for:
//   H2D   = host-to-device copy (cost_host_to_device)
//   NPU   = pure NPU kernel execution (cost_inference)
//   D2H   = device-to-host copy (cost_device_to_host)
//   Post  = yolo26 postprocessing (NMS + coordinate conversion)
// ============================================================================

static int run_stage_test_axcl(const char* model_path, const char* image_path, int device_id, int iterations) {
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        fprintf(stderr, "[FAIL] could not read image: %s\n", image_path);
        return 1;
    }

    ax_runner_axcl runner;

    // ax_runner_axcl.init() does not call axclrtEngineInit — do it explicitly
    int init_ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
    if (init_ret != 0) {
        fprintf(stderr, "[FAIL] axclrtEngineInit: 0x%x\n", init_ret);
        return 1;
    }

    int ret = runner.init(model_path);
    if (ret != 0) {
        fprintf(stderr, "[FAIL] ax_runner_axcl.init: %d\n", ret);
        return 1;
    }

    int input_h = runner.get_algo_height();
    int input_w = runner.get_algo_width();
    if (input_h <= 0 || input_w <= 0) {
        fprintf(stderr, "[FAIL] invalid model input size: %dx%d\n", input_h, input_w);
        return 1;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        std::vector<uint8_t> dummy(input_h * input_w * 3, 0);
        memcpy(runner.get_input(0).pVirAddr, dummy.data(), dummy.size());
        runner.inference();
    }

    std::vector<double> pre_times, h2d_times, npu_times, d2h_times, post_times;

    for (int i = 0; i < iterations; ++i) {
        cv::Mat letterbox;
        std::vector<uint8_t> input_data(input_h * input_w * 3, 0);

        auto t0 = std::chrono::steady_clock::now();
        app::detail::letterbox_resize(frame, letterbox, input_h, input_w);
        memcpy(input_data.data(), letterbox.data, input_data.size());
        auto t1 = std::chrono::steady_clock::now();

        memcpy(runner.get_input(0).pVirAddr, input_data.data(), input_data.size());
        runner.inference();
        auto t2 = std::chrono::steady_clock::now();

        // Postprocess yolo26: read 6 output tensors (stride 8/16/32: box+cls pairs)
        const ax_runner_tensor_t* outputs = runner.get_outputs_ptr(0);
        int num_outputs = runner.get_num_outputs();

        static const int strides[3] = {8, 16, 32};
        std::vector<app::detail::DetectedObject> proposals;
        proposals.reserve(10000);

        for (int s = 0; s < 3; ++s) {
            int box_idx = s * 2;
            int cls_idx = s * 2 + 1;
            if (box_idx >= num_outputs || cls_idx >= num_outputs) continue;

            const float* box_ptr = reinterpret_cast<const float*>(outputs[box_idx].pVirAddr);
            const float* cls_ptr = reinterpret_cast<const float*>(outputs[cls_idx].pVirAddr);
            int stride = strides[s];

            int feat_w = input_w / stride;
            int feat_h = input_h / stride;
            float prob_thresh = 0.45f;
            float p = std::max(std::min(prob_thresh, 1.f - 1e-6f), 1e-6f);
            float conf_raw = std::log(p / (1.f - p));

            for (int h = 0; h < feat_h; ++h) {
                for (int w = 0; w < feat_w; ++w) {
                    int best_cls = 0;
                    float best_logit = -1e30f;
                    const float* cls_row = cls_ptr + (h * feat_w + w) * 80;
                    for (int c = 0; c < 80; ++c) {
                        if (cls_row[c] > best_logit) {
                            best_logit = cls_row[c];
                            best_cls = c;
                        }
                    }
                    if (best_logit < conf_raw) continue;

                    float score = detection::sigmoid(best_logit);
                    const float* box_row = box_ptr + (h * feat_w + w) * 4;
                    float l = box_row[0], t = box_row[1], r = box_row[2], b = box_row[3];

                    float cx = (w + 0.5f) * stride;
                    float cy = (h + 0.5f) * stride;
                    float x0 = std::max(0.f, std::min(cx - l * stride, static_cast<float>(input_w - 1)));
                    float y0 = std::max(0.f, std::min(cy - t * stride, static_cast<float>(input_h - 1)));
                    float x1 = std::max(0.f, std::min(cx + r * stride, static_cast<float>(input_w - 1)));
                    float y1 = std::max(0.f, std::min(cy + b * stride, static_cast<float>(input_h - 1)));

                    app::detail::DetectedObject obj;
                    obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
                    obj.label = best_cls;
                    obj.prob = score;
                    proposals.push_back(obj);
                }
            }
        }

        // NMS
        if (!proposals.empty()) {
            detection::qsort_descent_inplace(proposals);
        }
        std::vector<int> picked;
        float nms_thresh = 0.45f;
        {
            const int n = static_cast<int>(proposals.size());
            std::vector<float> areas(n);
            for (int i = 0; i < n; ++i) areas[i] = proposals[i].rect.area();
            for (int i = 0; i < n; ++i) {
                bool keep = true;
                for (int j = 0; j < static_cast<int>(picked.size()); ++j) {
                    const auto& a = proposals[i];
                    const auto& b = proposals[picked[j]];
                    cv::Rect_<float> inter = a.rect & b.rect;
                    float inter_area = inter.area();
                    float union_area = areas[i] + areas[picked[j]] - inter_area;
                    if (inter_area / union_area > nms_thresh) {
                        keep = false;
                        break;
                    }
                }
                if (keep) picked.push_back(i);
            }
        }

        // Coordinate conversion
        float scale = std::min(static_cast<float>(input_h) / frame.rows,
                              static_cast<float>(input_w) / frame.cols);
        int resize_rows = static_cast<int>(scale * frame.rows);
        int resize_cols = static_cast<int>(scale * frame.cols);
        int pad_top  = (input_h - resize_rows) / 2;
        int pad_left = (input_w - resize_cols) / 2;
        float ratio_x = static_cast<float>(frame.rows) / resize_rows;
        float ratio_y = static_cast<float>(frame.cols) / resize_cols;

        std::vector<app::detail::DetectedObject> objects;
        objects.resize(picked.size());
        for (size_t pi = 0; pi < picked.size(); ++pi) {
            const auto& p = proposals[picked[pi]];
            float x0 = (p.rect.x - pad_left) * ratio_x;
            float y0 = (p.rect.y - pad_top)  * ratio_y;
            float x1 = (p.rect.x + p.rect.width  - pad_left) * ratio_x;
            float y1 = (p.rect.y + p.rect.height - pad_top)  * ratio_y;
            x0 = std::max(0.f, std::min(x0, static_cast<float>(frame.cols - 1)));
            y0 = std::max(0.f, std::min(y0, static_cast<float>(frame.rows - 1)));
            x1 = std::max(0.f, std::min(x1, static_cast<float>(frame.cols - 1)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(frame.rows - 1)));
            objects[pi].rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            objects[pi].label = p.label;
            objects[pi].prob = p.prob;
        }
        (void)objects; // suppress unused warning

        auto t3 = std::chrono::steady_clock::now();

        pre_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        h2d_times.push_back(runner.cost_host_to_device);
        npu_times.push_back(runner.cost_inference);
        d2h_times.push_back(runner.cost_device_to_host);
        post_times.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    runner.release();

    auto stats = [](const std::vector<double>& v, double& avg, double& mn, double& mx) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        avg = sum / v.size();
        mn = *std::min_element(v.begin(), v.end());
        mx = *std::max_element(v.begin(), v.end());
    };

    double pre_avg, pre_min, pre_max;
    double h2d_avg, h2d_min, h2d_max;
    double npu_avg, npu_min, npu_max;
    double d2h_avg, d2h_min, d2h_max;
    double post_avg, post_min, post_max;
    stats(pre_times, pre_avg, pre_min, pre_max);
    stats(h2d_times, h2d_avg, h2d_min, h2d_max);
    stats(npu_times, npu_avg, npu_min, npu_max);
    stats(d2h_times, d2h_avg, d2h_min, d2h_max);
    stats(post_times, post_avg, post_min, post_max);

    double total_avg = pre_avg + h2d_avg + npu_avg + d2h_avg + post_avg;

    fprintf(stderr, "\n  Stage breakdown (SDK)  (%d iterations, %dx%d input)\n",
           iterations, input_h, input_w);
    fprintf(stderr, "  %-10s %9s  %9s  %9s\n", "", "avg", "min", "max");
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "Pre", pre_avg, pre_min, pre_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "H2D", h2d_avg, h2d_min, h2d_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "NPU", npu_avg, npu_min, npu_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "D2H", d2h_avg, d2h_min, d2h_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "Post", post_avg, post_min, post_max);
    fprintf(stderr, "  %-10s %9.2f  ms\n", "Total", total_avg);
    fprintf(stderr, "\n");

    return 0;
}

// ============================================================================
// Stage breakdown mode (--stages) — InferenceEngine wrapper
// ============================================================================
// Uses InferenceEngine's public API. run_inference() wall-clock = H2D + NPU + D2H.
// For SDK-level H2D/D2H/NPU split, use run_stage_test_axcl above.
// ============================================================================

static int run_stage_test(const char* model_path, const char* image_path, int device_id, int iterations) {
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        fprintf(stderr, "[FAIL] could not read image: %s\n", image_path);
        return 1;
    }

    app::InferenceEngine engine(device_id);
    app::InferenceEngine::Config cfg;
    cfg.model_path = model_path;
    cfg.input_h = 640;
    cfg.input_w = 640;
    cfg.conf_thresh = 0.45f;
    cfg.iou_thresh = 0.45f;
    cfg.num_classes = 80;

    int ret = engine.init(cfg);
    if (ret != 0) {
        fprintf(stderr, "[FAIL] engine.init: %d\n", ret);
        return 1;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        std::vector<app::detail::DetectedObject> dummy;
        engine.detect(frame, dummy);
    }

    // Measure: run full detect() loop to get total throughput
    auto t_start = std::chrono::steady_clock::now();
    std::vector<app::detail::DetectedObject> objects;
    for (int i = 0; i < iterations; ++i) {
        engine.detect(frame, objects);
        objects.clear();
    }
    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double fps = iterations * 1000.0 / total_ms;

    // Per-stage timing (separate calls)
    std::vector<double> pre_times, inf_times, post_times;

    for (int i = 0; i < iterations; ++i) {
        cv::Mat letterbox;
        std::vector<app::detail::TensorInfo> outputs;
        std::vector<app::detail::DetectedObject> objs;

        auto t0 = std::chrono::steady_clock::now();
        engine.preprocess(frame, letterbox);
        auto t1 = std::chrono::steady_clock::now();
        engine.run_inference(letterbox, outputs);
        auto t2 = std::chrono::steady_clock::now();
        engine.postprocess(outputs, frame, objs);
        auto t3 = std::chrono::steady_clock::now();

        pre_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        inf_times.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        post_times.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }

    auto stats = [](const std::vector<double>& v, double& avg, double& mn, double& mx) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        avg = sum / v.size();
        mn = *std::min_element(v.begin(), v.end());
        mx = *std::max_element(v.begin(), v.end());
    };

    double pre_avg, pre_min, pre_max;
    double inf_avg, inf_min, inf_max;
    double post_avg, post_min, post_max;
    stats(pre_times, pre_avg, pre_min, pre_max);
    stats(inf_times, inf_avg, inf_min, inf_max);
    stats(post_times, post_avg, post_min, post_max);

    double total_avg = pre_avg + inf_avg + post_avg;

    fprintf(stderr, "\n  Stage breakdown  (%d iterations, %.0fx%.0f input)\n",
           iterations, cfg.input_h * 1.0, cfg.input_w * 1.0);
    fprintf(stderr, "  %-10s %9s  %9s  %9s\n", "", "avg", "min", "max");
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "Pre", pre_avg, pre_min, pre_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "Inf", inf_avg, inf_min, inf_max);
    fprintf(stderr, "  %-10s %9.2f  %9.2f  %9.2f  ms\n", "Post", post_avg, post_min, post_max);
    fprintf(stderr, "  %-10s %9.2f  ms\n", "Total", total_avg);
    fprintf(stderr, "  %-10s %9.1f  fps\n", "Throughput", fps);
    fprintf(stderr, "\n  Note: Inf = run_inference() wall-clock (H2D + NPU kernel + D2H).\n");
    fprintf(stderr, "        SDK-level H2D/D2H/NPU split available via run_stage_test_axcl().\n\n");

    return 0;
}

// ============================================================================
// Multi-thread sweep mode (--sweep)
// ============================================================================

static int run_sweep_test(const char* model_path, int device_id, int max_threads, int iterations) {
    fprintf(stderr, "# Sweep: max_threads=%d iters=%d\n", max_threads, iterations);
    fprintf(stderr, "#  thr   fps  eng_fps  eff%%   1x   spdup  pre_ms  inf_ms  post_ms\n");

    double base_fps = 0, base_pre = 0, base_inf = 0, base_post = 0;

    for (int n = 1; n <= max_threads; ++n) {
        std::atomic<int> done{0}, fail{0};
        std::atomic<uint64_t> us_pre{0}, us_inf{0}, us_post{0};
        std::mutex mtx;

        std::vector<std::thread> threads;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; ++i) {
            threads.emplace_back([=, &done, &fail, &us_pre, &us_inf, &us_post, &mtx]() {
                axclrtSetDevice(device_id);
                axclrtContext ctx = nullptr;
                axclrtCreateContext(&ctx, device_id);
                axclrtSetCurrentContext(ctx);
                axclrtEngineInit(AXCL_VNPU_DISABLE);

                app::InferenceEngine engine(device_id);
                app::InferenceEngine::Config cfg;
                cfg.model_path = model_path;
                cfg.input_h = 640; cfg.input_w = 640;
                cfg.conf_thresh = 0.45f; cfg.iou_thresh = 0.45f;
                cfg.num_classes = 80;
                int r = engine.init(cfg);
                if (r != 0) {
                    std::lock_guard<std::mutex> l(mtx);
                    fprintf(stderr, "FAIL thread %d init: %d\n", i, r);
                    return;
                }

                cv::Mat fake(360, 640, CV_8UC3);
                cv::randu(fake, cv::Scalar::all(0), cv::Scalar::all(255));
                std::vector<app::detail::DetectedObject> objs;
                cv::Mat letterbox;

                for (int j = 0; j < iterations; ++j) {
                    auto t_a = std::chrono::steady_clock::now();
                    r = engine.preprocess(fake, letterbox);
                    auto t_b = std::chrono::steady_clock::now();
                    if (r == 0) {
                        std::vector<app::detail::TensorInfo> outs;
                        r = engine.run_inference(letterbox, outs);
                        auto t_c = std::chrono::steady_clock::now();
                        if (r == 0) {
                            r = engine.postprocess(outs, fake, objs);
                            us_post += std::chrono::duration_cast<std::chrono::microseconds>(t_c - t_b).count();
                        }
                        us_inf += std::chrono::duration_cast<std::chrono::microseconds>(t_c - t_b).count();
                    }
                    us_pre += std::chrono::duration_cast<std::chrono::microseconds>(t_b - t_a).count();
                    if (r != 0) ++fail;
                    objs.clear();
                    ++done;
                }
            });
        }

        int last_pct = 0;
        while (done < n * iterations) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            int pct = done * 100 / (n * iterations);
            if (pct != last_pct && pct % 20 == 0)
                fprintf(stderr, "\r  sweep %2d  %d%%", n, pct), last_pct = pct;
        }
        for (auto& t : threads) t.join();
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        int total = done.load();
        double fps = total / elapsed_s;
        double eng_fps = fps / n;

        double pre_ms = us_pre.load() / 1000.0 / total;
        double inf_ms = us_inf.load() / 1000.0 / total;
        double post_ms = us_post.load() / 1000.0 / total;

        if (n == 1) {
            base_fps = fps;
            base_pre = pre_ms; base_inf = inf_ms; base_post = post_ms;
        }

        double theoretical = base_fps * n;
        double eff = theoretical > 0 ? fps / theoretical * 100 : 0;
        double spdup = base_fps > 0 ? fps / base_fps : 0;

        fprintf(stderr, "  %2d  %5.1f  %6.1f  %4.0f%%  %4.1f  %5.2f  %6.2f  %6.2f  %7.2f\n",
               n, fps, eng_fps, eff, theoretical, spdup, pre_ms, inf_ms, post_ms);
        if (fail.load()) fprintf(stderr, "  sweep %2d  failures=%d\n", n, fail.load());
        else fprintf(stderr, "\r  sweep %2d  done\n", n);
    }

    fprintf(stderr, "\n  Baseline: %.1f fps  (pre=%.2f inf=%.2f post=%.2f ms)\n\n",
           base_fps, base_pre, base_inf, base_post);
    return 0;
}

// ============================================================================
// Multi-thread throughput mode (--mt)
// ============================================================================

static int run_mt_test(const char* model_path, int device_id, int num_threads, int num_iterations) {
    fprintf(stderr, "  launching %d threads x %d iterations...\n", num_threads, num_iterations);

    std::atomic<int> done{0}, fail{0};
    std::mutex mtx;
    std::vector<std::thread> threads;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([=, &done, &fail, &mtx]() {
            axclrtSetDevice(device_id);
            axclrtContext ctx = nullptr;
            axclrtCreateContext(&ctx, device_id);
            axclrtSetCurrentContext(ctx);
            axclrtEngineInit(AXCL_VNPU_DISABLE);

            app::InferenceEngine engine(device_id);
            app::InferenceEngine::Config cfg;
            cfg.model_path = model_path;
            cfg.input_h = 640; cfg.input_w = 640;
            cfg.conf_thresh = 0.45f; cfg.iou_thresh = 0.45f;
            cfg.num_classes = 80;
            int r = engine.init(cfg);
            if (r != 0) {
                std::lock_guard<std::mutex> l(mtx);
                fprintf(stderr, "  FAIL thread %d init: %d\n", i, r);
                return;
            }

            cv::Mat fake(360, 640, CV_8UC3);
            cv::randu(fake, cv::Scalar::all(0), cv::Scalar::all(255));
            std::vector<app::detail::DetectedObject> objs;
            for (int j = 0; j < num_iterations; ++j) {
                r = engine.detect(fake, objs);
                if (r != 0) {
                    std::lock_guard<std::mutex> l(mtx);
                    fprintf(stderr, "  FAIL thread %d iter %d: %d\n", i, j, r);
                }
                objs.clear();
                ++done;
            }
        });
    }

    int last_pct = 0;
    while (done < num_threads * num_iterations) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        int pct = done * 100 / (num_threads * num_iterations);
        if (pct != last_pct) {
            fprintf(stderr, "\r  progress  %d%%", pct);
            last_pct = pct;
        }
    }
    for (auto& t : threads) t.join();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    int total = done.load();
    double fps = total / elapsed_s;

    fprintf(stderr, "\r  completed  %d/%d frames in %.2fs\n", total, num_threads * num_iterations, elapsed_s);
    fprintf(stderr, "\n  threads=%d  frames=%d  elapsed=%.2fs\n", num_threads, total, elapsed_s);
    fprintf(stderr, "  fps=%.1f  ms/frame=%.2f  eng_fps=%.1f\n\n",
           fps, 1000.0 / fps, fps / num_threads);
    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> <image> [options]\n\n", argv[0]);
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "  (none)            feasibility test\n");
        fprintf(stderr, "  --stages [N]      stage timing breakdown (default N=100)\n");
        fprintf(stderr, "  --sweep <N> <M>   thread sweep: 1..N threads, M iters each\n");
        fprintf(stderr, "  --mt <N> <M>      N-thread throughput test\n\n");
        fprintf(stderr, "Examples:\n");
        fprintf(stderr, "  %s model.axmodel image.jpg\n", argv[0]);
        fprintf(stderr, "  %s model.axmodel image.jpg --stages 100\n", argv[0]);
        fprintf(stderr, "  %s model.axmodel image.jpg --sweep 40 100\n", argv[0]);
        fprintf(stderr, "  %s model.axmodel image.jpg --mt 4 200\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* image_path = argv[2];

    axclError ret = axclInit(nullptr);
    if (ret != 0) { fprintf(stderr, "[FAIL] axclInit: 0x%x\n", ret); return 1; }

    axclrtDeviceList lst;
    ret = axclrtGetDeviceList(&lst);
    if (ret != 0 || lst.num == 0) { fprintf(stderr, "[FAIL] GetDeviceList\n"); axclFinalize(); return 1; }
    ret = axclrtSetDevice(lst.devices[0]);
    if (ret != 0) { fprintf(stderr, "[FAIL] SetDevice\n"); axclFinalize(); return 1; }

    int exit_code = 0;

    if (argc >= 4 && strcmp(argv[3], "--stages") == 0) {
        int iters = (argc >= 5) ? atoi(argv[4]) : 100;
            fprintf(stderr, "\n[ Stages ]  model=%s  image=%s\n", model_path, image_path);
        exit_code = run_stage_test_axcl(model_path, image_path, lst.devices[0], iters);
    } else if (argc >= 5 && strcmp(argv[3], "--sweep") == 0) {
        fprintf(stderr, "\n[ Sweep ]  model=%s  image=%s\n", model_path, image_path);
        exit_code = run_sweep_test(model_path, lst.devices[0], atoi(argv[4]), atoi(argv[5]));
    } else if (argc >= 5 && strcmp(argv[3], "--mt") == 0) {
        fprintf(stderr, "\n[ MT ]  model=%s  image=%s\n", model_path, image_path);
        exit_code = run_mt_test(model_path, lst.devices[0], atoi(argv[4]), atoi(argv[5]));
    } else {
        // Feasibility test
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) { fprintf(stderr, "[FAIL] could not read: %s\n", image_path); axclFinalize(); return 1; }

        app::InferenceEngine engine(lst.devices[0]);
        app::InferenceEngine::Config cfg;
        cfg.model_path = model_path;
        cfg.input_h = 640; cfg.input_w = 640;
        cfg.conf_thresh = 0.45f; cfg.iou_thresh = 0.45f;
        cfg.num_classes = 80;

        ret = engine.init(cfg);
        if (ret != 0) { fprintf(stderr, "[FAIL] init: %d\n", ret); axclFinalize(); return 1; }

        std::vector<app::detail::DetectedObject> objs;
        ret = engine.detect(frame, objs);
        if (ret != 0) { fprintf(stderr, "[FAIL] detect: %d\n", ret); axclFinalize(); return 1; }

        fprintf(stderr, "[ OK ]  detected %zu objects\n", objs.size());
        for (const auto& o : objs)
            fprintf(stderr, "       [%2d] %s  %.0f%%\n", o.label,
                   app::COCO_CLASS_NAMES[o.label].c_str(), o.prob * 100);
        fprintf(stderr, "\n");
    }

    axclFinalize();
    return exit_code;
}
