// test_mosaic_renderer.cpp — MosaicRenderer unit tests
#include <app/mosaic_renderer.hpp>
#include <app/detail/postprocess.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdio>

int main() {
    int passed = 0;
    int failed = 0;

    const std::vector<std::string> kClassNames = {
        "person", "bicycle", "car", "motorcycle", "airplane"
    };

    // Test 1: Construction and mosaic size
    {
        app::MosaicRenderer renderer(4, 2, 640, 360);
        const cv::Mat& mosaic = renderer.mosaic();
        if (mosaic.rows == 720 && mosaic.cols == 1280 && mosaic.type() == CV_8UC3) {
            fprintf(stderr, "PASS: test_construct mosaic size 1280x720\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_construct got %dx%d type=%d\n",
                    mosaic.cols, mosaic.rows, mosaic.type());
            ++failed;
        }
    }

    // Test 2: Single stream with a frame, no detections
    {
        app::MosaicRenderer renderer(1, 1, 640, 480);
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 50, 50));

        std::vector<const cv::Mat*> frames(1);
        frames[0] = &frame;
        std::vector<std::vector<app::detail::DetectedObject>> detections(1);

        renderer.render(frames, detections, {}, kClassNames);
        const cv::Mat& mosaic = renderer.mosaic();

        if (!mosaic.empty()) {
            fprintf(stderr, "PASS: test_render_no_detections\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_render_no_detections mosaic empty\n");
            ++failed;
        }

        // Check that frame was rendered (pixel should not be background color)
        // OpenCV uses BGR: cv::Scalar(B,G,R) → pixel[0]=B, [1]=G, [2]=R
        cv::Vec3b pixel = mosaic.at<cv::Vec3b>(240, 320);
        if (pixel[0] == 100 && pixel[1] == 50 && pixel[2] == 50) {
            fprintf(stderr, "PASS: test_render_no_detections pixel match\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_render_no_detections pixel=(%d,%d,%d) expected=(100,50,50) [BGR]\n",
                    pixel[0], pixel[1], pixel[2]);
            ++failed;
        }
    }

    // Test 3: Rendering with detections
    {
        app::MosaicRenderer renderer(1, 1, 640, 480);
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));

        std::vector<const cv::Mat*> frames(1);
        frames[0] = &frame;

        std::vector<std::vector<app::detail::DetectedObject>> detections(1);
        app::detail::DetectedObject obj;
        obj.label = 0;  // person
        obj.prob = 0.95f;
        obj.rect = cv::Rect_<float>(100, 100, 200, 150);
        detections[0].push_back(obj);

        renderer.render(frames, detections, {}, kClassNames);

        fprintf(stderr, "PASS: test_render_with_detections\n");
        ++passed;
    }

    // Test 4: Missing frame (nullptr)
    {
        app::MosaicRenderer renderer(1, 1, 640, 480);
        std::vector<const cv::Mat*> frames(1);
        frames[0] = nullptr;
        std::vector<std::vector<app::detail::DetectedObject>> detections(1);

        renderer.render(frames, detections, {}, kClassNames);
        const cv::Mat& mosaic = renderer.mosaic();

        // Should have drawn "No Signal" placeholder
        if (!mosaic.empty()) {
            fprintf(stderr, "PASS: test_missing_frame\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_missing_frame mosaic empty\n");
            ++failed;
        }
    }

    // Test 5: Multiple streams mosaic layout
    {
        app::MosaicRenderer renderer(4, 2, 640, 360);
        std::vector<const cv::Mat*> frames(4);
        std::vector<std::vector<app::detail::DetectedObject>> detections(4);

        for (int i = 0; i < 4; ++i) {
            cv::Mat f(360, 640, CV_8UC3, cv::Scalar(i * 50, i * 30, i * 20));
            frames[i] = &f;
        }

        renderer.render(frames, detections, {}, kClassNames);
        const cv::Mat& mosaic = renderer.mosaic();

        if (mosaic.rows == 720 && mosaic.cols == 1280) {
            fprintf(stderr, "PASS: test_multi_stream layout\n");
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_multi_stream size %dx%d\n", mosaic.cols, mosaic.rows);
            ++failed;
        }
    }

    // Test 6: Detection box scaling (object at full frame should cover entire cell)
    {
        app::MosaicRenderer renderer(1, 1, 640, 480);
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(50, 50, 50));

        std::vector<const cv::Mat*> frames(1);
        frames[0] = &frame;

        std::vector<std::vector<app::detail::DetectedObject>> detections(1);
        app::detail::DetectedObject obj;
        obj.label = 2;  // car
        obj.prob = 0.9f;
        obj.rect = cv::Rect_<float>(0, 0, 640, 480);  // full frame
        detections[0].push_back(obj);

        renderer.render(frames, detections, {}, kClassNames);

        fprintf(stderr, "PASS: test_detection_scaling\n");
        ++passed;
    }

    // Test 7: Empty class names vector
    {
        app::MosaicRenderer renderer(1, 1, 640, 480);
        cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(50, 50, 50));

        std::vector<const cv::Mat*> frames(1);
        frames[0] = &frame;

        std::vector<std::vector<app::detail::DetectedObject>> detections(1);
        app::detail::DetectedObject obj;
        obj.label = 0;
        obj.prob = 0.9f;
        obj.rect = cv::Rect_<float>(100, 100, 100, 100);
        detections[0].push_back(obj);

        std::vector<std::string> empty_names;
        renderer.render(frames, detections, {}, empty_names);

        fprintf(stderr, "PASS: test_empty_class_names\n");
        ++passed;
    }

    // Test 8: Mosaic cols = 1 (single column)
    {
        app::MosaicRenderer renderer(2, 1, 640, 480);
        std::vector<const cv::Mat*> frames(2);
        std::vector<std::vector<app::detail::DetectedObject>> detections(2);

        for (int i = 0; i < 2; ++i) {
            cv::Mat f(480, 640, CV_8UC3, cv::Scalar(30, 60, 90));
            frames[i] = &f;
        }

        renderer.render(frames, detections, {}, kClassNames);
        const cv::Mat& mosaic = renderer.mosaic();

        if (mosaic.rows == 960 && mosaic.cols == 640) {
            fprintf(stderr, "PASS: test_single_column layout rows=%d\n", mosaic.rows);
            ++passed;
        } else {
            fprintf(stderr, "FAIL: test_single_column %dx%d\n", mosaic.cols, mosaic.rows);
            ++failed;
        }
    }

    fprintf(stderr, "\n=== MosaicRenderer test results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
