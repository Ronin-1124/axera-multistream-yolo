#include "app/detail/postprocess.hpp"
#include "app/detail/axcl_model.hpp"
#include <algorithm>
#include <cmath>

namespace app {
namespace detail {

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& objs, int left, int right) {
    int i = left, j = right;
    float p = objs[(left + right) / 2].prob;
    while (i <= j) {
        while (objs[i].prob > p) ++i;
        while (objs[j].prob < p) --j;
        if (i <= j) {
            std::swap(objs[i], objs[j]);
            ++i; --j;
        }
    }
    if (left < j)  qsort_descent_inplace(objs, left, j);
    if (i < right) qsort_descent_inplace(objs, i, right);
}

template<typename T>
static void qsort_descent_inplace(std::vector<T>& objs) {
    if (objs.empty()) return;
    qsort_descent_inplace(objs, 0, static_cast<int>(objs.size()) - 1);
}

template<typename T>
static void nms_sorted_bboxes(const std::vector<T>& objs, std::vector<int>& picked, float nms_thresh) {
    picked.clear();
    const int n = static_cast<int>(objs.size());
    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i) areas[i] = objs[i].rect.area();

    for (int i = 0; i < n; ++i) {
        const T& a = objs[i];
        bool keep = true;
        for (int j = 0; j < static_cast<int>(picked.size()); ++j) {
            const T& b = objs[picked[j]];
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

static void generate_proposals_yolo26(int stride, const float* box_feat, const float* cls_feat,
                                       float prob_thresh, std::vector<DetectedObject>& proposals,
                                       int letterbox_cols, int letterbox_rows, int num_classes) {
    const int feat_w = letterbox_cols / stride;
    const int feat_h = letterbox_rows / stride;

    const float p = std::max(std::min(prob_thresh, 1.f - 1e-6f), 1e-6f);
    const float conf_raw = std::log(p / (1.f - p));

    for (int h = 0; h < feat_h; ++h) {
        for (int w = 0; w < feat_w; ++w) {
            // Find best class
            int best_cls = 0;
            float best_logit = -1e30f;
            const float* cls_ptr = cls_feat + (h * feat_w + w) * num_classes;
            for (int c = 0; c < num_classes; ++c) {
                if (cls_ptr[c] > best_logit) {
                    best_logit = cls_ptr[c];
                    best_cls = c;
                }
            }
            if (best_logit < conf_raw) continue;

            float score = sigmoid(best_logit);

            // Decode box: [l, t, r, b] (center-relative offsets)
            const float* box_ptr = box_feat + (h * feat_w + w) * 4;
            float l = box_ptr[0], t = box_ptr[1], r = box_ptr[2], b = box_ptr[3];

            float cx = (w + 0.5f) * stride;
            float cy = (h + 0.5f) * stride;
            float x0 = std::max(0.f, std::min(cx - l * stride, static_cast<float>(letterbox_cols - 1)));
            float y0 = std::max(0.f, std::min(cy - t * stride, static_cast<float>(letterbox_rows - 1)));
            float x1 = std::max(0.f, std::min(cx + r * stride, static_cast<float>(letterbox_cols - 1)));
            float y1 = std::max(0.f, std::min(cy + b * stride, static_cast<float>(letterbox_rows - 1)));

            DetectedObject obj;
            obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            obj.label = best_cls;
            obj.prob = score;
            proposals.push_back(obj);
        }
    }
}

void postprocess_yolo26(const std::vector<TensorInfo>& outputs,
                        int src_rows, int src_cols,
                        int letterbox_rows, int letterbox_cols,
                        float conf_thresh, float iou_thresh,
                        std::vector<DetectedObject>& objects,
                        int num_classes) {
    // YOLO26: outputs[0,1] = stride8 [box, cls]
    //         outputs[2,3] = stride16 [box, cls]
    //         outputs[4,5] = stride32 [box, cls]
    static const int strides[3] = {8, 16, 32};

    std::vector<DetectedObject> proposals;
    proposals.reserve(10000);

    for (int i = 0; i < 3; ++i) {
        int box_idx  = i * 2;     // 0, 2, 4
        int cls_idx  = i * 2 + 1; // 1, 3, 5
        const float* box_ptr = reinterpret_cast<const float*>(outputs[box_idx].vir_addr);
        const float* cls_ptr = reinterpret_cast<const float*>(outputs[cls_idx].vir_addr);
        generate_proposals_yolo26(strides[i], box_ptr, cls_ptr, conf_thresh, proposals,
                                  letterbox_cols, letterbox_rows, num_classes);
    }

    // NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_thresh);

    // Convert coordinates back to original image size
    float scale = std::min(static_cast<float>(letterbox_rows) / src_rows,
                           static_cast<float>(letterbox_cols) / src_cols);
    int resize_rows = static_cast<int>(scale * src_rows);
    int resize_cols = static_cast<int>(scale * src_cols);
    int pad_top  = (letterbox_rows - resize_rows) / 2;
    int pad_left = (letterbox_cols - resize_cols) / 2;
    float ratio_x = static_cast<float>(src_rows) / resize_rows;
    float ratio_y = static_cast<float>(src_cols) / resize_cols;

    objects.resize(picked.size());
    for (size_t i = 0; i < picked.size(); ++i) {
        const DetectedObject& p = proposals[picked[i]];
        float x0 = (p.rect.x - pad_left) * ratio_x;
        float y0 = (p.rect.y - pad_top)  * ratio_y;
        float x1 = (p.rect.x + p.rect.width  - pad_left) * ratio_x;
        float y1 = (p.rect.y + p.rect.height - pad_top)  * ratio_y;

        x0 = std::max(0.f, std::min(x0, static_cast<float>(src_cols - 1)));
        y0 = std::max(0.f, std::min(y0, static_cast<float>(src_rows - 1)));
        x1 = std::max(0.f, std::min(x1, static_cast<float>(src_cols - 1)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(src_rows - 1)));

        objects[i].rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
        objects[i].label = p.label;
        objects[i].prob  = p.prob;
    }
}

// Letterbox preprocessing: resize image to fit target size, pad with black borders.
void letterbox_resize(const cv::Mat& src, cv::Mat& dst, int target_h, int target_w) {
    float scale = std::min(static_cast<float>(target_h) / src.rows,
                           static_cast<float>(target_w) / src.cols);
    int resize_h = static_cast<int>(scale * src.rows);
    int resize_w = static_cast<int>(scale * src.cols);

    // Create black background and resize directly into the target sub-rect
    // to avoid extra alloc+copy of a temporary Mat
    dst.create(target_h, target_w, CV_8UC3);
    dst.setTo(cv::Scalar(0, 0, 0));

    int top  = (target_h - resize_h) / 2;
    int left = (target_w - resize_w) / 2;
    cv::resize(src, dst(cv::Rect(left, top, resize_w, resize_h)), cv::Size(resize_w, resize_h));
}

void draw_detections(cv::Mat& mat, const std::vector<DetectedObject>& objects,
                     const std::vector<std::string>& class_names) {
    // COCO 80 colors — BGR format (matches OpenCV cv::Scalar ordering)
    static const int COCO_COLORS[80][3] = {
        {128,56,0}, {255,226,128}, {94,0,128}, {37,0,128}, {255,0,94}, {0,226,255},
        {18,0,255}, {0,151,255}, {0,170,255}, {56,0,255}, {0,255,75}, {75,0,255},
        {0,18,255}, {0,113,255}, {255,207,0}, {0,255,94}, {18,255,0}, {94,0,255},
        {0,207,255}, {255,0,150}, {0,56,255}, {255,0,207}, {0,245,255}, {151,0,255},
        {132,255,0}, {0,75,255}, {0,0,255}, {255,245,0}, {37,255,0}, {255,0,188},
        {188,255,0}, {0,255,37}, {0,255,113}, {0,245,151}, {255,0,113}, {113,0,255},
        {245,0,255}, {0,151,255}, {0,255,188}, {0,132,255}, {0,245,0}, {255,132,0},
        {226,0,128}, {255,37,0}, {0,207,255}, {94,255,0}, {0,226,255}, {56,255,0},
        {255,94,0}, {255,113,0}, {0,132,255}, {255,0,132}, {255,170,0}, {255,0,188},
        {113,255,0}, {0,245,255}, {113,0,255}, {255,188,0}, {0,113,255}, {255,0,0},
        {0,56,255}, {255,0,113}, {0,255,188}, {255,0,94}, {255,0,18}, {18,255,0},
        {0,255,132}, {0,188,255}, {0,245,255}, {0,169,255}, {37,0,255}, {0,255,151},
        {255,0,151}, {188,0,255}, {0,255,37}, {0,255,0}, {255,0,170}, {255,0,37},
        {255,75,0}, {0,0,255}
    };

    for (const auto& obj : objects) {
        const std::string& label = (obj.label < static_cast<int>(class_names.size()))
                                       ? class_names[obj.label] : std::to_string(obj.label);

        const int* c = COCO_COLORS[obj.label % 80];
        cv::rectangle(mat, obj.rect, cv::Scalar(c[0], c[1], c[2]), 2);

        char text[128];
        std::snprintf(text, sizeof(text), "%s %.1f%%", label.c_str(), obj.prob * 100);

        int baseline = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int x = static_cast<int>(obj.rect.x);
        int y = static_cast<int>(obj.rect.y) - label_size.height - baseline;
        if (y < 0) y = 0;
        if (x + label_size.width > mat.cols) x = mat.cols - label_size.width;

        cv::rectangle(mat, cv::Rect(x, y, label_size.width, label_size.height + baseline),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(mat, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

} // namespace detail
} // namespace app
