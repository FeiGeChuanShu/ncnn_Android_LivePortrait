#ifndef LANDMARK_H_
#define LANDMARK_H_

#include <opencv2/core/core.hpp>
#include <net.h>


class Landmark
{
public:
    Landmark(const char* param, const char* bin);
    ~Landmark();

    int detect(const cv::Mat& rgb, const cv::Rect& box, std::vector<cv::Point2f>& pts, int target_size);

    int draw(const cv::Mat& rgb, const std::vector<cv::Point2f>& pts);

private:
    ncnn::Net net_;
};

#endif // LANDMARK_H_
