#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#if _WIN32
# define M_PI		3.14159265358979323846	/* pi */
# define M_PI_2		1.57079632679489661923	/* pi/2 */
#endif
namespace detection
{
    typedef struct
    {
        int grid0;
        int grid1;
        int stride;
    } GridAndStride;

    typedef struct Object
    {
        cv::Rect_<float> rect;
        int label;
        float prob;
        cv::Point2f landmark[5];
        /* for yolov5-seg */
        cv::Mat mask;
        std::vector<float> mask_feat;
        std::vector<float> kps_feat;
        /* for yolov8-obb */
        float angle;
    } Object;

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    template<typename T>
    static inline float intersection_area(const T& a, const T& b)
    {
        cv::Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    template<typename T>
    static void qsort_descent_inplace(std::vector<T>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }
// #pragma omp parallel sections
        {
// #pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
// #pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    template<typename T>
    static void qsort_descent_inplace(std::vector<T>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    template<typename T>
    static void nms_sorted_bboxes(const std::vector<T>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const T& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const T& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    void get_out_bbox(std::vector<Object>& objects, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
    {
        /* yolov5 draw the result */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / src_rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / src_cols;
        }
        resize_cols = int(scale_letterbox * src_cols);
        resize_rows = int(scale_letterbox * src_rows);

        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;

        float ratio_x = (float)src_rows / resize_rows;
        float ratio_y = (float)src_cols / resize_cols;

        int count = objects.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = (x0 - tmp_w) * ratio_x;
            y0 = (y0 - tmp_h) * ratio_y;
            x1 = (x1 - tmp_w) * ratio_x;
            y1 = (y1 - tmp_h) * ratio_y;

            for (int l = 0; l < 5; l++)
            {
                auto lx = objects[i].landmark[l].x;
                auto ly = objects[i].landmark[l].y;
                objects[i].landmark[l] = cv::Point2f((lx - tmp_w) * ratio_x, (ly - tmp_h) * ratio_y);
            }

            x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
    }

    void get_out_bbox(std::vector<Object>& proposals, std::vector<Object>& objects, const float nms_threshold, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
    {
        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        /* yolov5 draw the result */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
        {
            scale_letterbox = letterbox_rows * 1.0 / src_rows;
        }
        else
        {
            scale_letterbox = letterbox_cols * 1.0 / src_cols;
        }
        resize_cols = int(scale_letterbox * src_cols);
        resize_rows = int(scale_letterbox * src_rows);

        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;

        float ratio_x = (float)src_rows / resize_rows;
        float ratio_y = (float)src_cols / resize_cols;

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = (x0 - tmp_w) * ratio_x;
            y0 = (y0 - tmp_h) * ratio_y;
            x1 = (x1 - tmp_w) * ratio_x;
            y1 = (y1 - tmp_h) * ratio_y;

            for (int l = 0; l < 5; l++)
            {
                auto lx = objects[i].landmark[l].x;
                auto ly = objects[i].landmark[l].y;
                objects[i].landmark[l] = cv::Point2f((lx - tmp_w) * ratio_x, (ly - tmp_h) * ratio_y);
            }

            x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }
    }

    static void generate_proposals_yolo26(int stride, const float* feat, const float* feat_cls, float prob_threshold, std::vector<Object>& objects,
                                          int letterbox_cols, int letterbox_rows, int cls_num = 80)
    {
        const int feat_w = letterbox_cols / stride;
        const int feat_h = letterbox_rows / stride;

        const float p = std::min(std::max(prob_threshold, 1e-6f), 1.f - 1e-6f);
        const float conf_raw = std::log(p / (1.f - p));

        for (int h = 0; h < feat_h; ++h)
        {
            for (int w = 0; w < feat_w; ++w)
            {
                int best_c = 0;
                float best_logit = -1e30f;
                const float* cls_ptr = feat_cls + (h * feat_w + w) * cls_num;
                for (int c = 0; c < cls_num; ++c)
                {
                    float v = cls_ptr[c];
                    if (v > best_logit)
                    {
                        best_logit = v;
                        best_c = c;
                    }
                }

                if (best_logit < conf_raw)
                {
                    continue;
                }

                const float score = sigmoid(best_logit);

                const float* box_ptr = feat + (h * feat_w + w) * 4;
                const float l = box_ptr[0];
                const float t = box_ptr[1];
                const float r = box_ptr[2];
                const float b = box_ptr[3];

                const float cx = (w + 0.5f) * stride;
                const float cy = (h + 0.5f) * stride;

                float x0 = (cx - l * stride);
                float y0 = (cy - t * stride);
                float x1 = (cx + r * stride);
                float y1 = (cy + b * stride);

                x0 = std::max(0.f, std::min(x0, (float)(letterbox_cols - 1)));
                y0 = std::max(0.f, std::min(y0, (float)(letterbox_rows - 1)));
                x1 = std::max(0.f, std::min(x1, (float)(letterbox_cols - 1)));
                y1 = std::max(0.f, std::min(y1, (float)(letterbox_rows - 1)));

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = best_c;
                obj.prob = score;
                objects.push_back(obj);
            }
        }
    }

    static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, const char** class_names, const char* output_name, double fontScale = 0.5, int thickness = 1)
    {
        static const std::vector<cv::Scalar> COCO_COLORS = {
            {128, 56, 0, 255}, {128, 226, 255, 0}, {128, 0, 94, 255}, {128, 0, 37, 255}, {128, 0, 255, 94}, {128, 255, 226, 0}, {128, 0, 18, 255}, {128, 255, 151, 0}, {128, 170, 0, 255}, {128, 0, 255, 56}, {128, 255, 0, 75}, {128, 0, 75, 255}, {128, 0, 255, 169}, {128, 255, 0, 207}, {128, 75, 255, 0}, {128, 207, 0, 255}, {128, 37, 0, 255}, {128, 0, 207, 255}, {128, 94, 0, 255}, {128, 0, 255, 113}, {128, 255, 18, 0}, {128, 255, 0, 56}, {128, 18, 0, 255}, {128, 0, 255, 226}, {128, 170, 255, 0}, {128, 255, 0, 245}, {128, 151, 255, 0}, {128, 132, 255, 0}, {128, 75, 0, 255}, {128, 151, 0, 255}, {128, 0, 151, 255}, {128, 132, 0, 255}, {128, 0, 255, 245}, {128, 255, 132, 0}, {128, 226, 0, 255}, {128, 255, 37, 0}, {128, 207, 255, 0}, {128, 0, 255, 207}, {128, 94, 255, 0}, {128, 0, 226, 255}, {128, 56, 255, 0}, {128, 255, 94, 0}, {128, 255, 113, 0}, {128, 0, 132, 255}, {128, 255, 0, 132}, {128, 255, 170, 0}, {128, 255, 0, 188}, {128, 113, 255, 0}, {128, 245, 0, 255}, {128, 113, 0, 255}, {128, 255, 188, 0}, {128, 0, 113, 255}, {128, 255, 0, 0}, {128, 0, 56, 255}, {128, 255, 0, 113}, {128, 0, 255, 188}, {128, 255, 0, 94}, {128, 255, 0, 18}, {128, 18, 255, 0}, {128, 0, 255, 132}, {128, 0, 188, 255}, {128, 0, 245, 255}, {128, 0, 169, 255}, {128, 37, 255, 0}, {128, 255, 0, 151}, {128, 188, 0, 255}, {128, 0, 255, 37}, {128, 0, 255, 0}, {128, 255, 0, 170}, {128, 255, 0, 37}, {128, 255, 75, 0}, {128, 0, 0, 255}, {128, 255, 207, 0}, {128, 255, 0, 226}, {128, 255, 245, 0}, {128, 188, 255, 0}, {128, 0, 255, 18}, {128, 0, 255, 75}, {128, 0, 255, 151}, {128, 255, 56, 0}, {128, 245, 255, 0}};
        cv::Mat image = bgr.clone();

        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object& obj = objects[i];

            fprintf(stdout, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                    obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

            cv::rectangle(image, obj.rect, COCO_COLORS[obj.label], thickness);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);

            int x = obj.rect.x;
            int y = obj.rect.y - label_size.height - baseLine;
            if (y < 0)
                y = 0;
            if (x + label_size.width > image.cols)
                x = image.cols - label_size.width;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          cv::Scalar(0, 0, 0), -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, fontScale,
                        cv::Scalar(255, 255, 255), thickness);
        }

        cv::imwrite(std::string(output_name) + ".jpg", image);
    }
} // namespace detection

#pragma GCC diagnostic pop