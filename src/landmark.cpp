#include "landmark.h"
#include "common.h"
Landmark::Landmark(const char* param, const char* bin){
    net_.load_param(param);
    net_.load_model(bin);
}

Landmark::~Landmark(){
    net_.clear();
}

int Landmark::detect(const cv::Mat& rgb, const cv::Rect& box, std::vector<cv::Point2f>& pts, int target_size)
{
    if(rgb.empty())
        return -1;
    cv::Rect roi;
    if(target_size == 192)
        roi = adjust_boundingBox(box, rgb.cols, rgb.rows);
    else
        roi = box;
    cv::Mat src = rgb(roi).clone();

    ncnn::Mat input;
    if(target_size == src.cols && target_size == src.rows){
        input = ncnn::Mat::from_pixels(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows);
    }
    else{
        input = ncnn::Mat::from_pixels_resize(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows, target_size, target_size);
    }

    ncnn::Extractor ex = net_.create_extractor();
    ex.input("input", input);
    
    ncnn::Mat out;
    ex.extract("out", out);

    float* ptr = out.channel(0);
    int num_pts = out.w / 2;
    float x = 0.f , y = 0.f;
    for (int i = 0; i < num_pts; ++i) {
        if(num_pts == 106){
            x = (ptr[i * 2] + 1) / 2 * src.cols;
            y = (ptr[i * 2 + 1] + 1) / 2 * src.rows;
        }
        else{
            x = ptr[i * 2] * src.cols;
            y = ptr[i * 2 + 1] * src.rows;
        }
       
        pts.emplace_back(x + roi.x, y + roi.y);
    }

    return 0;
}
