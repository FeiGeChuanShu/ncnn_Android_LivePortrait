#include "net.h"
#include <opencv2/opencv.hpp>

typedef struct _retargeting_info{
    float lip_close_ratio;
    float eye_close_ratio;
    float head_pitch_variation;
    float head_yaw_variation;
    float head_roll_variation;
    float mov_x;
    float mov_y;
    float mov_z;
    float lip_variation_zero;
    float lip_variation_one;
    float lip_variation_two;
    float lip_variation_three;
    float smile;
    float wink;
    float eyebrow;
    float eyeball_direction_x;
    float eyeball_direction_y;
    _retargeting_info(){
        this->eye_close_ratio = 0.35f;
        this->lip_close_ratio = 0.f;
        this->head_pitch_variation = -7.f; //[-15, 15]
        this->head_yaw_variation = -0.4f; //[-25, 25]
        this->head_roll_variation = 2.f; //[-15, 15]

        this->mov_x = 0.f;
        this->mov_y = 0.f;
        this->mov_z = 1.f;
        this->lip_variation_zero = 0.f;
        this->lip_variation_one = 0.f;
        this->lip_variation_two = 0.f;
        this->lip_variation_three = 0.f;
        this->smile = 0.f;
        this->wink = 0.f;
        this->eyebrow = 0.f;
        this->eyeball_direction_x = 0.f;
        this->eyeball_direction_y = 0.f;
    }
}retargeting_info_t;

typedef struct _rect_info {
    cv::Point2f center;
    cv::Point2f size;
    float angle;
}rect_info_t;
typedef struct _ret_dct {
    cv::Mat M_o2c;
    cv::Mat M_c2o;
    cv::Mat img_crop;
    cv::Mat img_crop_256;
    std::vector<cv::Point2f> pt_crop;
    std::vector<cv::Point2f> lmk_crop;
    std::vector<cv::Point2f> lmk_crop_256;
}ret_dct_t;
typedef struct _crop_cfg {
    int dsize;
    float scale;
    float vx_ratio;
    float vy_ratio;
}crop_cfg_t;

typedef struct _kp_info {
    float pitch;
    float yaw;
    float roll;
    float scale;
    std::vector<float> t;
    std::vector<cv::Point3f> kp;
    std::vector<cv::Point3f> exp;

}kp_info_t;

typedef struct _trajectory {
    int start;
    int end;
    std::vector<std::vector<cv::Point2f>> lmk_lst;
}trajectory_t;
typedef struct _template_dct {
    float scale;
    cv::Mat R_d;
    std::vector<cv::Point3f> exp;
    std::vector<float> t;
}template_dct_t;

#define PI 3.14159265358979323846
void concat(const std::vector<ncnn::Mat>& bottoms, ncnn::Mat& c, int axis);
void reshape(const ncnn::Mat& in, ncnn::Mat& out, int w, int h, int d, int c);

cv::Rect adjust_boundingBox(const cv::Rect& boundingBox, int img_w, int img_h);
int make_coordinate_grid(int d, int h, int w, ncnn::Mat& mesh);

void parse_pt2_from_pt106(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2);
void parse_pt2_from_pt203(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2);

void parse_pt2_from_pt_x(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2);

void parse_rect_from_landmark(const std::vector<cv::Point2f>& pts, float scale, float vx_ratio, float vy_ratio, rect_info_t& rect_info);
void estimate_similar_transform_from_pts(const std::vector<cv::Point2f>& pts,cv::Mat& M, int dsize, float scale, float vx_ratio, float vy_ratio);

void transform_pts(const std::vector<cv::Point2f>& pts,std::vector<cv::Point2f>& dst_pts, const cv::Mat& M);
void transform_img(const cv::Mat& img, cv::Mat& out, const cv::Mat& M, int dsize_h, int dsize_w);
void transform_img(const cv::Mat& img, cv::Mat& out, const cv::Mat& M, int dsize);
cv::Mat get_rotation_matrix(float pitch_, float yaw_, float roll_);
void transform_keypoint(kp_info_t& kp_info, std::vector<cv::Point3f>& kp_transformed);
float calculate_distance_ratio(std::vector<cv::Point2f>& lmk, int idx1, int idx2, int idx3, int idx4, float eps);
