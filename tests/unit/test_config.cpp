// test_config.cpp — Config loading unit tests
#include <app/config.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

int main() {
    int passed = 0;
    int failed = 0;

    // Helper: write JSON to temp file and parse
    auto parse_json = [](const char* json) -> app::AppConfig {
        static int counter = 0;
        std::ostringstream path;
        path << "/tmp/test_config_" << (++counter) << ".json";
        std::ofstream f(path.str());
        if (!f.is_open()) return {};
        f << json;
        f.close();
        app::AppConfig cfg = app::load_config(path.str());
        std::remove(path.str().c_str());
        return cfg;
    };

    // Test 1: Valid full config
    {
        const char* json = R"({
  "streams": [
    { "id": 0, "url": "rtsp://localhost/stream0", "enabled": true },
    { "id": 1, "url": "rtsp://localhost/stream1", "enabled": false }
  ],
  "model": {
    "path": "/models/yolo26n.axmodel",
    "input_h": 640,
    "input_w": 640,
    "type": "yolo26"
  },
  "inference": {
    "num_threads": 4,
    "conf_thresh": 0.5,
    "iou_thresh": 0.3
  },
  "output": {
    "display": true,
    "mosaic_cols": 3
  }
})";
        app::AppConfig cfg = parse_json(json);

        if (cfg.streams.size() == 2) {
            fprintf(stderr, "PASS: test_valid_full 2 streams\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full expected 2 streams, got %zu\n", cfg.streams.size());
            ++failed;
        }

        if (cfg.streams[0].id == 0 && cfg.streams[0].url == "rtsp://localhost/stream0" && cfg.streams[0].enabled) {
            fprintf(stderr, "PASS: test_valid_full stream[0]\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full stream[0] id=%d url=%s enabled=%d\n",
                    cfg.streams[0].id, cfg.streams[0].url.c_str(), cfg.streams[0].enabled);
            ++failed;
        }

        if (!cfg.streams[1].enabled) {
            fprintf(stderr, "PASS: test_valid_full stream[1] disabled\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full stream[1] should be disabled\n");
            ++failed;
        }

        if (cfg.model.path == "/models/yolo26n.axmodel" && cfg.model.input_h == 640 && cfg.model.input_w == 640) {
            fprintf(stderr, "PASS: test_valid_full model config\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full model path=%s h=%d w=%d\n",
                    cfg.model.path.c_str(), cfg.model.input_h, cfg.model.input_w);
            ++failed;
        }

        if (cfg.inference.num_threads == 4 && cfg.inference.conf_thresh > 0.49 && cfg.inference.conf_thresh < 0.51) {
            fprintf(stderr, "PASS: test_valid_full inference config\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full inference threads=%d conf=%.2f\n",
                    cfg.inference.num_threads, cfg.inference.conf_thresh);
            ++failed;
        }

        if (cfg.output.mosaic_cols == 3 && cfg.output.display) {
            fprintf(stderr, "PASS: test_valid_full output config\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_valid_full output cols=%d display=%d\n",
                    cfg.output.mosaic_cols, cfg.output.display);
            ++failed;
        }
    }

    // Test 2: Minimal config (missing optional fields use defaults)
    {
        const char* json = R"({
  "streams": [
    { "id": 5, "url": "/video/test.mp4" }
  ],
  "model": { "path": "/model.axmodel" }
})";
        app::AppConfig cfg = parse_json(json);

        if (cfg.streams.size() == 1 && cfg.streams[0].enabled) {
            fprintf(stderr, "PASS: test_minimal default enabled\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_minimal enabled=%d\n", cfg.streams[0].enabled);
            ++failed;
        }

        if (cfg.model.input_h == 640 && cfg.model.input_w == 640) {
            fprintf(stderr, "PASS: test_minimal default input size\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_minimal input_h=%d input_w=%d\n",
                    cfg.model.input_h, cfg.model.input_w);
            ++failed;
        }

        if (cfg.inference.conf_thresh > 0.44 && cfg.inference.conf_thresh < 0.46) {
            fprintf(stderr, "PASS: test_minimal default conf_thresh\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_minimal conf_thresh=%.2f\n", cfg.inference.conf_thresh);
            ++failed;
        }

        if (cfg.output.display && cfg.output.mosaic_cols == 2) {
            fprintf(stderr, "PASS: test_minimal default output\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_minimal output display=%d cols=%d\n",
                    cfg.output.display, cfg.output.mosaic_cols);
            ++failed;
        }
    }

    // Test 3: Config with extra/unknown fields (should be ignored)
    {
        const char* json = R"({
  "streams": [{ "id": 0, "url": "rtsp://x", "enabled": true, "unknown_field": 123 }],
  "model": { "path": "/m", "extra": "ignored" },
  "unknown_section": { "foo": "bar" }
})";
        app::AppConfig cfg = parse_json(json);
        if (cfg.streams.size() == 1) {
            fprintf(stderr, "PASS: test_extra_fields ignored\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_extra_fields streams=%zu\n", cfg.streams.size());
            ++failed;
        }
    }

    // Test 4: Empty streams array
    {
        const char* json = R"({
  "streams": [],
  "model": { "path": "/m" }
})";
        app::AppConfig cfg = parse_json(json);
        if (cfg.streams.empty()) {
            fprintf(stderr, "PASS: test_empty_streams\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_empty_streams size=%zu\n", cfg.streams.size());
            ++failed;
        }
    }

    // Test 5: Float numbers (conf_thresh, iou_thresh)
    {
        const char* json = R"({
  "streams": [{ "id": 0, "url": "rtsp://x" }],
  "model": { "path": "/m" },
  "inference": { "conf_thresh": 0.75, "iou_thresh": 0.55 }
})";
        app::AppConfig cfg = parse_json(json);
        if (cfg.inference.conf_thresh > 0.74 && cfg.inference.conf_thresh < 0.76) {
            fprintf(stderr, "PASS: test_float_numbers conf_thresh\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_float_numbers conf=%.2f\n", cfg.inference.conf_thresh);
            ++failed;
        }
        if (cfg.inference.iou_thresh > 0.54 && cfg.inference.iou_thresh < 0.56) {
            fprintf(stderr, "PASS: test_float_numbers iou_thresh\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_float_numbers iou=%.2f\n", cfg.inference.iou_thresh);
            ++failed;
        }
    }

    // Test 6: Negative numbers (should not appear in config, but parser should handle)
    {
        const char* json = R"({
  "streams": [{ "id": -1, "url": "rtsp://x" }],
  "model": { "path": "/m" }
})";
        app::AppConfig cfg = parse_json(json);
        // Parser uses stoi which handles negative numbers
        if (cfg.streams[0].id == -1) {
            fprintf(stderr, "PASS: test_negative_id\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_negative_id got %d\n", cfg.streams[0].id);
            ++failed;
        }
    }

    fprintf(stderr, "\n=== Config test results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
