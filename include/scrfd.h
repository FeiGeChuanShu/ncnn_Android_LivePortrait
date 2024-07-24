#ifndef SCRFD_H_
#define SCRFD_H_

#include <opencv2/core/core.hpp>
#include <net.h>

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class SCRFD
{
public:
    SCRFD(const char* param, const char* bin);

    ~SCRFD();


    int detect(const cv::Mat& rgb, std::vector<FaceObject>& faceobjects, float prob_threshold = 0.5f, float nms_threshold = 0.45f);

    int draw(cv::Mat& rgb, const std::vector<FaceObject>& faceobjects);

private:
    ncnn::Net scrfd;
    bool has_kps;
};

#endif // SCRFD_H_
