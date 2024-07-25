#ifndef LIVEPORTRAIT_H_
#define LIVEPORTRAIT_H_
#include <memory>
#include <net.h>
#include "landmark.h"
#include "scrfd.h"
#include "common.h"
class LivePortrait
{
public:
    LivePortrait() = default;
    ~LivePortrait();
    int load_model(const char* model_path);
    int run_single_iamge(const cv::Mat& source, const cv::Mat& mask_crop, cv::Mat& out, 
        float lip_close_ratio = 0.5f, float eye_close_ratio = 0.5f, float head_pitch_ratio = 0.f,
        float head_yaw_ratio = 0.f, float head_roll_ratio = 0.f);
    int run_multi_image(const cv::Mat& source, const cv::Mat& mask_crop, const std::vector<cv::Mat>& driving_rgb_lst, const std::string& save_path);
private:
    int prepare_driving(const std::vector<cv::Mat>& driving_rgb_lst, std::vector<template_dct_t>& template_dct_lst);
    int prepare_retargeting(const cv::Mat& img, const cv::Mat& mask_crop, ret_dct_t& crop_info, 
        ncnn::Mat& f_s, std::vector<cv::Point3f>& x_s,  cv::Mat& R_s, cv::Mat& R_d, kp_info_t& x_s_info, cv::Mat& mask_ori,
        float head_pitch_ratio = 0.f, float head_yaw_ratio = 0.f, float head_roll_ratio = 0.f);
    int crop_image(const cv::Mat& img, const std::vector<cv::Point2f>& pts, ret_dct_t& ret_dct, int dsize, float scale, float vx_ratio, float vy_ratio);
    void crop_source_image(const cv::Mat& img, crop_cfg_t& crop_cfg, ret_dct_t& crop_info);
    void get_kp_info(const cv::Mat& img, kp_info_t& kp_info);
    int extract_feature_3d(const cv::Mat& img, ncnn::Mat& feature_3d);
    float calc_lip_close_ratio(std::vector<cv::Point2f>& lmk);
    void calc_eye_close_ratio(std::vector<cv::Point2f>& lmk, std::vector<float>& c_s_eyes);
    int retarget_lip(std::vector<cv::Point3f>& kp_source, std::vector<float>& lip_close_ratio, std::vector<cv::Point3f>& delta);
    int retarget_eye(std::vector<cv::Point3f>& kp_source, std::vector<float>& eye_close_ratio, std::vector<cv::Point3f>& delta);
    void calc_lmks_from_cropped_video(const std::vector<cv::Mat>& driving_rgb_crop_lst, std::vector<std::vector<cv::Point2f>>& lmk_crop);
    void calc_driving_ratio(std::vector<std::vector<cv::Point2f>>& driving_lmk_lst,
        std::vector<std::vector<float>>& input_eye_ratio_lst,std::vector<float>& input_lip_ratio_lst);
    void make_motion_template(std::vector<cv::Mat>& I_d_lst, std::vector<template_dct_t>& template_dct_lst);
    cv::Mat prepare_paste_back(const cv::Mat& mask_crop, cv::Mat& crop_M_c2o, int dsize_w, int dsize_h);
    void stitching(std::vector<cv::Point3f>& kp_source, std::vector<cv::Point3f>& kp_driving, std::vector<cv::Point3f>& kp_driving_new);
    int spade_generator(ncnn::Mat& feature, ncnn::Mat& out);
    void paste_back(const cv::Mat& img_crop, const cv::Mat& M_c2o, const cv::Mat& img_ori, const cv::Mat& mask_ori, cv::Mat& result);

    void warp_decode(ncnn::Mat& feature_3d, std::vector<cv::Point3f>& kp_source_v, std::vector<cv::Point3f>& kp_driving_v, ncnn::Mat& mesh, cv::Mat& out_img);
    int warping_motion(ncnn::Mat& feat, ncnn::Mat& kp_driving, ncnn::Mat& kp_source, ncnn::Mat& mesh, ncnn::Mat& out);
    void calc_combined_lip_ratio(std::vector<cv::Point2f>& source_lmk, std::vector<float>& combined_lip_ratio, float input_lip_ratio);
    void calc_combined_eye_ratio(std::vector<cv::Point2f>& source_lmk, std::vector<float>& combined_eye_ratio, float input_eye_ratio);
private:
    kp_info_t x_s_info_;
    cv::Mat R_s_;
    cv::Mat R_d_;
    ret_dct_t crop_info_;
    ncnn::Mat f_s_;
    cv::Mat mask_ori_;
    std::vector<cv::Point3f> x_s_;
private:
    std::shared_ptr<SCRFD> face_detect_;
    std::shared_ptr<Landmark> face_landmark_106_;
    std::shared_ptr<Landmark> face_landmark_203_;
private:
    ncnn::Net warp_module_;
    ncnn::Net motion_extrator_module_;
    ncnn::Net appearance_feature_extractor_;
    ncnn::Net stitching_retargeting_lip_;
    ncnn::Net stitching_retargeting_;
    ncnn::Net stitching_retargeting_eye_;
    ncnn::Net spade_generator_module_;
    ncnn::Net warping_module2_;
    const float norm_[3] = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
};

#endif // LIVEPORTRAIT_H_
