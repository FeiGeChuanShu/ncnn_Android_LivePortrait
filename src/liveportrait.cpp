#include "liveportrait.h"
#include "layer.h"
#include "net.h"
#include "layer_type.h"

class GridSampleN : public ncnn::Layer {
public:
    virtual int create_pipeline(const ncnn::Option& opt) {
        {
            gridsample = ncnn::create_layer_cpu(ncnn::LayerType::GridSample);
            // set param
            ncnn::ParamDict pd;
            pd.set(0, 1);// sample_type
            pd.set(1, 1);// padding_mode
            pd.set(2, 0);// align_corner
            pd.set(3, 0);// permute_fusion

            gridsample->load_param(pd);

            gridsample->load_model(ncnn::ModelBinFromMatArray(0));

            gridsample->create_pipeline(opt);
        }
        {
            concat = ncnn::create_layer_cpu(ncnn::LayerType::Concat);
            // set param
            ncnn::ParamDict pd;
            pd.set(0, 0);// axis
        
            concat->load_param(pd);

            concat->load_model(ncnn::ModelBinFromMatArray(0));

            concat->create_pipeline(opt);
        }



        return 0;
    }
    virtual int destroy_pipeline(const ncnn::Option& opt){
        if (gridsample){
            gridsample->destroy_pipeline(opt);
            delete gridsample;
            gridsample = 0;
        }
        if (concat) {
            concat->destroy_pipeline(opt);
            delete concat;
            concat = 0;
        }

        return 0;
    }

    GridSampleN() {
        one_blob_only = false;

    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {
        const ncnn::Mat& grid = bottom_blobs[1];
        const ncnn::Mat& input = bottom_blobs[0];
        
        int num = grid.c;
        int elempack = input.elempack;
        size_t elemsize = input.elemsize;

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(64 * 64, 16, 4, 22, elemsize, elempack, opt.blob_allocator);

        std::vector<ncnn::Mat> tops(num);
#pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < num; ++i) {
            std::vector<ncnn::Mat> bottoms(2);
            bottoms[0] = input;
            ncnn::Mat tmp = grid.channel(i).reshape(3, 64, 64, 16);
            bottoms[1] = tmp;

            std::vector<ncnn::Mat> top(1);
            gridsample->forward(bottoms, top, opt);

            tops[i] = top[0].reshape(64 * 64, 16, 4, 1);
        }

        concat->forward(tops, top_blobs, opt);
        return 0;
    }
private:
    Layer* gridsample;
    Layer* concat;
};
DEFINE_LAYER_CREATOR(GridSampleN)
LivePortrait::~LivePortrait(){
    warp_module_.clear();
    motion_extrator_module_.clear();
    appearance_feature_extractor_.clear();
    stitching_retargeting_lip_.clear();
    stitching_retargeting_.clear();
    stitching_retargeting_eye_.clear();
    spade_generator_module_.clear();
    warping_module2_.clear();
    
}
int LivePortrait::load_model(const char* model_path)
{
    std::string model_dir = model_path;
    if (model_dir.find_last_of('/') == std::string::npos &&
         model_dir.find_last_of('\\') == std::string::npos) {
        model_dir += "/";
    } 

    warp_module_.register_custom_layer("GridSampleN", GridSampleN_layer_creator);
    warp_module_.load_param((model_dir + "warp.param").c_str());
    warp_module_.load_model((model_dir + "warp.bin").c_str());
    
    motion_extrator_module_.load_param((model_dir + "motion_extractor.ncnn.param").c_str());
    motion_extrator_module_.load_model((model_dir + "motion_extractor.ncnn.bin").c_str());

    appearance_feature_extractor_.load_param((model_dir + "appearance_feature_extractor.ncnn.param").c_str());
    appearance_feature_extractor_.load_model((model_dir + "appearance_feature_extractor.ncnn.bin").c_str());

    stitching_retargeting_lip_.load_param((model_dir + "stitching_retargeting_lip.param").c_str());
    stitching_retargeting_lip_.load_model((model_dir + "stitching_retargeting_lip.bin").c_str());

    stitching_retargeting_eye_.load_param((model_dir + "stitching_retargeting_eye.param").c_str());
    stitching_retargeting_eye_.load_model((model_dir + "stitching_retargeting_eye.bin").c_str());

    stitching_retargeting_.load_param((model_dir + "stitching_retargeting.param").c_str());
    stitching_retargeting_.load_model((model_dir + "stitching_retargeting.bin").c_str());

    spade_generator_module_.load_param((model_dir + "spade_generator.param").c_str());
    spade_generator_module_.load_model((model_dir + "spade_generator.bin").c_str());

    face_detect_ = std::make_shared<SCRFD>((model_dir + "scrfd_500m_kps-opt2.param").c_str(), (model_dir + "scrfd_500m_kps-opt2.bin").c_str());
    face_landmark_106_ = std::make_shared<Landmark>((model_dir + "106.param").c_str(), (model_dir + "106.bin").c_str());
    face_landmark_203_ = std::make_shared<Landmark>((model_dir + "203.param").c_str(), (model_dir + "203.bin").c_str());

    return 0;
}
int LivePortrait::prepare_retargeting(const cv::Mat& img, const cv::Mat& mask_crop, ret_dct_t& crop_info, 
    ncnn::Mat& f_s, std::vector<cv::Point3f>& x_s, cv::Mat& R_s, cv::Mat& R_d, kp_info_t& x_s_info, 
    cv::Mat& mask_ori, float head_pitch_ratio, float head_yaw_ratio, float head_roll_ratio){
    cv::Mat src;
    cv::cvtColor(img, src, cv::COLOR_BGR2RGB);

    crop_cfg_t crop_cfg{512, 2.3, 0.f, -0.125};
    crop_source_image(src, crop_cfg, crop_info);

    get_kp_info(crop_info.img_crop_256, x_s_info);

    //headpose
    float x_s_info_user_pitch = x_s_info.pitch +  head_pitch_ratio;
    float x_s_info_user_yaw = x_s_info.yaw +  head_yaw_ratio;
    float x_s_info_user_roll = x_s_info.roll +  head_roll_ratio;

    R_s = get_rotation_matrix(x_s_info.pitch, x_s_info.yaw, x_s_info.roll);
    R_d = get_rotation_matrix(x_s_info_user_pitch, x_s_info_user_yaw, x_s_info_user_roll);
    
    extract_feature_3d(crop_info.img_crop_256, f_s);

    transform_keypoint(x_s_info, x_s);

    mask_ori = prepare_paste_back(mask_crop, crop_info.M_c2o, src.cols, src.rows);
    
    return 0;
}
int LivePortrait::run_single_iamge(const cv::Mat& source, const cv::Mat& mask_crop, 
    cv::Mat& out, float lip_close_ratio, float eye_close_ratio, float head_pitch_ratio,
    float head_yaw_ratio, float head_roll_ratio){

    prepare_retargeting(source, mask_crop, crop_info_, f_s_, x_s_, R_s_, R_d_, x_s_info_, mask_ori_,
        head_pitch_ratio, head_yaw_ratio, head_roll_ratio);
    fprintf(stderr, "prepare retargeting done\n");

    cv::Mat R_d_new = (R_d_ * R_s_.t()) * R_s_;

    std::vector<cv::Point3f> x_d_new;
    for (size_t j = 0; j < x_s_info_.kp.size(); ++j) {
        float x = (x_s_info_.kp[j].x * R_d_new.at<float>(0, 0) +
            x_s_info_.kp[j].y * R_d_new.at<float>(1, 0) + 
            x_s_info_.kp[j].z * R_d_new.at<float>(2, 0) + x_s_info_.exp[j].x) * x_s_info_.scale + x_s_info_.t[0];
        float y = (x_s_info_.kp[j].x * R_d_new.at<float>(0, 1) +
            x_s_info_.kp[j].y * R_d_new.at<float>(1, 1) +
            x_s_info_.kp[j].z * R_d_new.at<float>(2, 1) + x_s_info_.exp[j].y) * x_s_info_.scale + x_s_info_.t[1];
        float z = (x_s_info_.kp[j].x * R_d_new.at<float>(0, 2) +
            x_s_info_.kp[j].y * R_d_new.at<float>(1, 2) +
            x_s_info_.kp[j].z * R_d_new.at<float>(2, 2) + x_s_info_.exp[j].z) * x_s_info_.scale + x_s_info_.t[2];
        x_d_new.emplace_back(x, y, z);
    }

    ncnn::Mat mesh;
    make_coordinate_grid(16, 64, 64, mesh);

    std::vector<cv::Point3f> lip_delta_before_animation;
    std::vector<float> combined_lip_ratio_tensor;
    calc_combined_lip_ratio(crop_info_.lmk_crop, combined_lip_ratio_tensor, lip_close_ratio);
    retarget_lip(x_s_, combined_lip_ratio_tensor, lip_delta_before_animation);
    fprintf(stderr, "retarget lip done\n");

    std::vector<cv::Point3f> eye_delta_before_animation;
    std::vector<float> combined_eye_ratio_tensor;
    calc_combined_eye_ratio(crop_info_.lmk_crop, combined_eye_ratio_tensor, eye_close_ratio);
    retarget_eye(x_s_, combined_eye_ratio_tensor, eye_delta_before_animation);
    fprintf(stderr, "retarget eye done\n");
    
    std::vector<cv::Point3f> x_d_i_new(x_s_.size());
    for(size_t i = 0; i < x_d_new.size(); ++i){
        x_d_i_new[i] = x_d_new[i] + lip_delta_before_animation[i] + eye_delta_before_animation[i];
    }
    std::vector<cv::Point3f> kp_driving;
    stitching(x_s_, x_d_i_new, kp_driving);

    cv::Mat I_p_i;
    warp_decode(f_s_, x_s_, kp_driving, mesh, I_p_i);
    fprintf(stderr, "warp done\n");
    
    //cv::Mat I_p_pstbk;
    paste_back(I_p_i, crop_info_.M_c2o, source, mask_ori_, out);

    return 0;
}

int LivePortrait::prepare_driving(const std::vector<cv::Mat>& driving_rgb_lst, 
    std::vector<template_dct_t>& template_dct_lst){
    std::vector<std::vector<cv::Point2f>> driving_lmk_crop_lst;
    calc_lmks_from_cropped_video(driving_rgb_lst, driving_lmk_crop_lst);

    std::vector<cv::Mat> driving_rgb_crop_256x256_lst(driving_rgb_lst.size());
    for (size_t i = 0; i < driving_rgb_lst.size(); ++i) {
        cv::resize(driving_rgb_lst[i], driving_rgb_crop_256x256_lst[i], cv::Size(256, 256));
    }
    
    // std::vector<std::vector<float>> c_d_eyes_lst;
    // std::vector<float> c_d_lip_lst;
    // calc_driving_ratio(driving_lmk_crop_lst, c_d_eyes_lst, c_d_lip_lst);
    
    make_motion_template(driving_rgb_crop_256x256_lst, template_dct_lst);
    return 0;
}

int LivePortrait::run_multi_image(const cv::Mat& source, const cv::Mat& mask_crop, 
    const std::vector<cv::Mat>& driving_rgb_lst, const std::string& save_path){
    prepare_retargeting(source, mask_crop, crop_info_, f_s_, x_s_, R_s_, R_d_, x_s_info_, mask_ori_);
    fprintf(stderr, "prepare retargeting done\n");

    std::vector<template_dct_t> template_dct_lst;
    prepare_driving(driving_rgb_lst, template_dct_lst);
    fprintf(stderr, "prepare driving done\n");

    ncnn::Mat mesh;
    make_coordinate_grid(16, 64, 64, mesh);

    bool flag_lip_zero = true;
    std::vector<cv::Point3f> lip_delta_before_animation;
    std::vector<float> combined_lip_ratio_tensor_before_animation;
    calc_combined_lip_ratio(crop_info_.lmk_crop, combined_lip_ratio_tensor_before_animation, 0.f);

    if (combined_lip_ratio_tensor_before_animation[0] < 0.03)
        flag_lip_zero = false;
    else
        retarget_lip(x_s_, combined_lip_ratio_tensor_before_animation, lip_delta_before_animation);

    fprintf(stderr, "retargeting done\n");

    cv::Mat R_d_0;
    template_dct_t x_d_0_info;
    for (size_t i = 0; i < driving_rgb_lst.size(); ++i) {

        std::cout << "[frame]: " << i + 1 << "/" << driving_rgb_lst.size() << std::endl;
        template_dct_t x_d_i_info = template_dct_lst[i];
        cv::Mat R_d_i = x_d_i_info.R_d;
        if (i == 0) {
            R_d_0 = R_d_i;
            x_d_0_info.exp.assign(x_d_i_info.exp.begin(), x_d_i_info.exp.end());
            x_d_0_info.t.assign(x_d_i_info.t.begin(), x_d_i_info.t.end());
            x_d_0_info.R_d = x_d_i_info.R_d.clone();
            x_d_0_info.scale = x_d_i_info.scale;
        }


        cv::Mat R_new = (R_d_i * R_d_0.t()) * R_s_;
        std::vector<cv::Point3f> delta_new;
        for (size_t j = 0; j < x_s_info_.exp.size(); ++j) {
            delta_new.emplace_back(x_s_info_.exp[j] + (x_d_i_info.exp[j] - x_d_0_info.exp[j]));
        }
        float scale_new = x_s_info_.scale * (x_d_i_info.scale / x_d_0_info.scale);
        std::vector<float> t_new(3, 0.f);
        for (size_t j = 0; j < x_s_info_.t.size() - 1; ++j) {
            t_new[j] = x_s_info_.t[j] + (x_d_i_info.t[j] - x_d_0_info.t[j]);
        }

        std::vector<cv::Point3f> x_d_i_new;
        for (size_t j = 0; j < x_s_info_.kp.size(); ++j) {
            float x = (x_s_info_.kp[j].x * R_new.at<float>(0, 0) +
                x_s_info_.kp[j].y * R_new.at<float>(1, 0) + 
                x_s_info_.kp[j].z * R_new.at<float>(2, 0) + delta_new[j].x) * scale_new + t_new[0];
            float y = (x_s_info_.kp[j].x * R_new.at<float>(0, 1) +
                x_s_info_.kp[j].y * R_new.at<float>(1, 1) +
                x_s_info_.kp[j].z * R_new.at<float>(2, 1) + delta_new[j].y) * scale_new + t_new[1];
            float z = (x_s_info_.kp[j].x * R_new.at<float>(0, 2) +
                x_s_info_.kp[j].y * R_new.at<float>(1, 2) +
                x_s_info_.kp[j].z * R_new.at<float>(2, 2) + delta_new[j].z) * scale_new + t_new[2];
            x_d_i_new.emplace_back(x, y, z);
        }

        std::vector<cv::Point3f> kp_driving;
        stitching(x_s_, x_d_i_new, kp_driving);
        if (flag_lip_zero) {
            for (size_t j = 0; j < kp_driving.size(); ++j) {
                kp_driving[j] = kp_driving[j] + lip_delta_before_animation[j];
            }
        }

        cv::Mat I_p_i;
        warp_decode(f_s_, x_s_, kp_driving, mesh, I_p_i);

        cv::Mat I_p_pstbk;
        paste_back(I_p_i, crop_info_.M_c2o, source, mask_ori_, I_p_pstbk);
        cv::imwrite(save_path + "/"+std::to_string(i)+".jpg", I_p_pstbk);
    }

    return 0;
}
int LivePortrait::crop_image(const cv::Mat& img, const std::vector<cv::Point2f>& pts, 
    ret_dct_t& ret_dct, int dsize, float scale, float vx_ratio, float vy_ratio) {
    
    cv::Mat M_inv;
    estimate_similar_transform_from_pts(pts, M_inv, dsize, scale, vx_ratio, vy_ratio);

    cv::Mat img_crop;
    transform_img(img, img_crop, M_inv, dsize);
    transform_pts(pts, ret_dct.pt_crop, M_inv);

    ret_dct.img_crop = img_crop.clone();
    ret_dct.M_o2c = M_inv.clone();
    ret_dct.M_c2o = M_inv.inv();
    
    return 0;
}

void LivePortrait::crop_source_image(const cv::Mat& img, crop_cfg_t& crop_cfg, ret_dct_t& crop_info) {
    std::vector<FaceObject> face_objs;
    face_detect_->detect(img, face_objs);

    std::vector<cv::Point2f> lmk;
    face_landmark_106_->detect(img, cv::Rect(face_objs[0].rect), lmk, 192);

    crop_image(img, lmk, crop_info, crop_cfg.dsize, crop_cfg.scale, crop_cfg.vx_ratio, crop_cfg.vy_ratio);

    //landmark_run
    {
        int dsize = 224;
        float scale = 1.5;
        float vx_ratio = 0.f;
        float vy_ratio = -0.1;
        ret_dct_t crop_dct;
        crop_image(img, lmk, crop_dct, dsize, scale, vx_ratio, vy_ratio);

        std::vector<cv::Point2f> lmk1;
        face_landmark_203_->detect(crop_dct.img_crop, cv::Rect(0, 0, crop_dct.img_crop.cols, crop_dct.img_crop.rows), lmk1, dsize);

        transform_pts(lmk1, crop_info.lmk_crop, crop_dct.M_c2o);
    }

    cv::resize(crop_info.img_crop, crop_info.img_crop_256, cv::Size(256, 256), 0, 0, cv::INTER_AREA);
    for (size_t i = 0; i < crop_info.lmk_crop.size(); ++i) {
        crop_info.lmk_crop_256.emplace_back(crop_info.lmk_crop[i].x * 256.f / crop_cfg.dsize, crop_info.lmk_crop[i].y * 256.f / crop_cfg.dsize);
    }
    
}


void LivePortrait::get_kp_info(const cv::Mat& img, kp_info_t& kp_info) {
    ncnn::Extractor ex = motion_extrator_module_.create_extractor();
    
    ncnn::Mat input = ncnn::Mat::from_pixels(
        img.data, ncnn::Mat::PIXEL_RGB,
        img.cols, img.rows);

    input.substract_mean_normalize(0, norm_);

    ex.input("in0", input);
    ncnn::Mat out_pitch;
    ex.extract("out0", out_pitch);
    ncnn::Mat out_yaw;
    ex.extract("out1", out_yaw);
    ncnn::Mat out_roll;
    ex.extract("out2", out_roll);
    ncnn::Mat out_t;
    ex.extract("out3", out_t);
    ncnn::Mat out_exp;
    ex.extract("out4", out_exp);
    ncnn::Mat out_scale;
    ex.extract("out5", out_scale);
    ncnn::Mat out_kp;
    ex.extract("out6", out_kp);
    
    kp_info.pitch = 0;
    kp_info.roll = 0;
    kp_info.yaw = 0;

    //2degree
    float* pitch_ptr = (float*)out_pitch.data;
    float* yaw_ptr = (float*)out_yaw.data;
    float* roll_ptr = (float*)out_roll.data;
    for (int i = 0; i < 66; ++i) {
        kp_info.pitch += pitch_ptr[i] * i;
        kp_info.yaw += yaw_ptr[i] * i;
        kp_info.roll += roll_ptr[i] * i;
    }
    kp_info.pitch = kp_info.pitch * 3 - 97.5;
    kp_info.yaw = kp_info.yaw * 3 - 97.5;
    kp_info.roll = kp_info.roll * 3 - 97.5;

    kp_info.scale = ((float*)out_scale.data)[0];
    kp_info.t.assign((float*)out_t.data, (float*)out_t.data + out_t.w);

    float* exp_ptr = (float*)out_exp.data;
    float* kp_ptr = (float*)out_kp.data;
    for (int i = 0; i < 21; ++i) {
        kp_info.exp.emplace_back(exp_ptr[i * 3], exp_ptr[i * 3 + 1], exp_ptr[i * 3 + 2]);
        kp_info.kp.emplace_back(kp_ptr[i * 3], kp_ptr[i * 3 + 1], kp_ptr[i * 3 + 2]);
    }

}

int LivePortrait::extract_feature_3d(const cv::Mat& img, ncnn::Mat& feature_3d){
    ncnn::Extractor ex = appearance_feature_extractor_.create_extractor();
    ncnn::Mat input = ncnn::Mat::from_pixels(
        img.data, ncnn::Mat::PIXEL_RGB,
        img.cols, img.rows);

    input.substract_mean_normalize(0, norm_);
    ex.input("in0", input);

    ex.extract("out0", feature_3d);
    return 0;
}


int LivePortrait::retarget_lip(std::vector<cv::Point3f>& kp_source, 
    std::vector<float>& lip_close_ratio, std::vector<cv::Point3f>& delta){
    ncnn::Extractor ex = stitching_retargeting_lip_.create_extractor();

    ncnn::Mat in0(65, 1);
    float* in_ptr = (float*)in0.data;
    for (int i = 0; i < 21; ++i) {
        in_ptr[i*3] = kp_source[i].x;
        in_ptr[i * 3 + 1] = kp_source[i].y;
        in_ptr[i * 3 + 2] = kp_source[i].z;
    }
    in_ptr[63] = lip_close_ratio[0];
    in_ptr[64] = lip_close_ratio[1];

    ex.input("input", in0);

    ncnn::Mat out;
    ex.extract("output", out);
    delta.clear();
    float* ptr = out.channel(0);
    for (int i = 0; i < 21; ++i) {
        delta.emplace_back(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
    }

    return 0;
}

int LivePortrait::warping_motion(ncnn::Mat& feat, ncnn::Mat& kp_driving, 
    ncnn::Mat& kp_source, ncnn::Mat& mesh, ncnn::Mat& out){
    ncnn::Extractor ex = warp_module_.create_extractor();

    ncnn::Mat zeros(64 * 64, 16, 1);
    zeros.fill(0.f);

    ex.input("zeros", zeros);
    ex.input("feat", feat);
    ex.input("mesh", mesh);
    ex.input("kp_driving", kp_driving);
    ex.input("kp_source", kp_source);

    ex.extract("out0", out);

    return 0;
}

int LivePortrait::retarget_eye(std::vector<cv::Point3f>& kp_source, 
    std::vector<float>& eye_close_ratio, std::vector<cv::Point3f>& delta){
    ncnn::Extractor ex = stitching_retargeting_eye_.create_extractor();

    ncnn::Mat in0(66, 1);
    float* in_ptr = (float*)in0.data;
    for (int i = 0; i < 21; ++i) {
        in_ptr[i*3] = kp_source[i].x;
        in_ptr[i * 3 + 1] = kp_source[i].y;
        in_ptr[i * 3 + 2] = kp_source[i].z;
    }
    in_ptr[63] = eye_close_ratio[0];
    in_ptr[64] = eye_close_ratio[1];
    in_ptr[65] = eye_close_ratio[2];

    ex.input("input", in0);

    ncnn::Mat out;
    ex.extract("output", out);

    float* ptr = out.channel(0);
    delta.clear();
    for (int i = 0; i < 21; ++i) {
        delta.emplace_back(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2]);
    }
    return 0;
}


float LivePortrait::calc_lip_close_ratio(std::vector<cv::Point2f>& lmk) {
    return calculate_distance_ratio(lmk, 90, 102, 48, 66, 0.000001);
}
void LivePortrait::calc_eye_close_ratio(std::vector<cv::Point2f>& lmk, std::vector<float>& c_s_eyes) {
    float lefteye_close_ratio = calculate_distance_ratio(lmk, 6, 18, 0, 12, 0.000001);
    float righteye_close_ratio = calculate_distance_ratio(lmk, 30, 42, 24, 36, 0.000001);
    c_s_eyes.push_back(lefteye_close_ratio);
    c_s_eyes.push_back(righteye_close_ratio);
}
void LivePortrait::calc_combined_lip_ratio(std::vector<cv::Point2f>& source_lmk, 
    std::vector<float>& combined_lip_ratio, float input_lip_ratio) {
    float c_s_lip = calc_lip_close_ratio(source_lmk);
    float c_d_lip_i = input_lip_ratio;
    combined_lip_ratio.push_back(c_s_lip);
    combined_lip_ratio.push_back(c_d_lip_i);
}

void LivePortrait::calc_combined_eye_ratio(std::vector<cv::Point2f>& source_lmk, 
    std::vector<float>& combined_eye_ratio, float input_eye_ratio) {
    std::vector<float> c_s_eyes;
    calc_eye_close_ratio(source_lmk, c_s_eyes);
    float c_d_eyes_i = input_eye_ratio;
    combined_eye_ratio.push_back(c_s_eyes[0]);
    combined_eye_ratio.push_back(c_s_eyes[1]);
    combined_eye_ratio.push_back(c_d_eyes_i);
}
void LivePortrait::calc_lmks_from_cropped_video(const std::vector<cv::Mat>& driving_rgb_crop_lst, 
    std::vector<std::vector<cv::Point2f>>& lmk_crop) {

    trajectory_t trajectory{ -1, -1, {} };
    int dsize = 224;
    float scale = 1.5;
    float vx_ratio = 0.f;
    float vy_ratio = -0.1;
    for (size_t i = 0; i < driving_rgb_crop_lst.size(); ++i) {
        if (i == 0 || trajectory.start == -1) {
            
            std::vector<FaceObject> face_objs;
            face_detect_->detect(driving_rgb_crop_lst[i], face_objs);

            std::vector<cv::Point2f> lmk;
            face_landmark_106_->detect(driving_rgb_crop_lst[i], cv::Rect(face_objs[0].rect), lmk, 192);
            
            std::vector<cv::Point2f> lmk_transformed;
            //landmark_run
            {
                ret_dct_t crop_dct;
                crop_image(driving_rgb_crop_lst[i], lmk, crop_dct, dsize, scale, vx_ratio, vy_ratio);

                std::vector<cv::Point2f> lmk1;
                face_landmark_203_->detect(crop_dct.img_crop, cv::Rect(0, 0, crop_dct.img_crop.cols, crop_dct.img_crop.rows), lmk1, dsize);
                transform_pts(lmk1, lmk_transformed, crop_dct.M_c2o);
            }

            trajectory.start = i;
            trajectory.end = i;
            trajectory.lmk_lst.push_back(lmk_transformed);
        }
        else {
            std::vector<cv::Point2f> lmk_transformed;
            //landmark_run
            {
                ret_dct_t crop_dct;
                crop_image(driving_rgb_crop_lst[i], trajectory.lmk_lst.back(), crop_dct, dsize, scale, vx_ratio, vy_ratio);

                std::vector<cv::Point2f> lmk1;
                face_landmark_203_->detect(crop_dct.img_crop, cv::Rect(0, 0, crop_dct.img_crop.cols, crop_dct.img_crop.rows), lmk1, dsize);
                transform_pts(lmk1, lmk_transformed, crop_dct.M_c2o);
            }
            trajectory.end = i;
            trajectory.lmk_lst.push_back(lmk_transformed);
        }
    }

    lmk_crop.resize(trajectory.lmk_lst.size());
    std::copy(trajectory.lmk_lst.begin(), trajectory.lmk_lst.end(), lmk_crop.begin());
}

void LivePortrait::calc_driving_ratio(std::vector<std::vector<cv::Point2f>>& driving_lmk_lst,
    std::vector<std::vector<float>>& input_eye_ratio_lst, std::vector<float>& input_lip_ratio_lst) {
    input_lip_ratio_lst.resize(driving_lmk_lst.size());
    for (size_t i = 0; i < driving_lmk_lst.size(); ++i) {
        std::vector<float> input_eye_ratio;
        calc_eye_close_ratio(driving_lmk_lst[i], input_eye_ratio);
        input_eye_ratio_lst.push_back(input_eye_ratio);
        input_lip_ratio_lst[i] = calc_lip_close_ratio(driving_lmk_lst[i]);
    }
}

void LivePortrait::make_motion_template(std::vector<cv::Mat>& I_d_lst, 
    std::vector<template_dct_t>& template_dct_lst) {
    for (size_t i = 0; i < I_d_lst.size(); ++i) {
        cv::Mat I_d_i = I_d_lst[i];

        kp_info_t x_d_i_info{ 0.0f, 0.0f, 0.0f,{},{} };
        get_kp_info(I_d_i, x_d_i_info);

        cv::Mat R_d_i = get_rotation_matrix(x_d_i_info.pitch, x_d_i_info.yaw, x_d_i_info.roll);

        template_dct_t temp_dct;
        temp_dct.scale = x_d_i_info.scale;
        temp_dct.R_d = R_d_i.clone();
        temp_dct.exp.assign(x_d_i_info.exp.begin(), x_d_i_info.exp.end());
        temp_dct.t.assign(x_d_i_info.t.begin(), x_d_i_info.t.end());
        template_dct_lst.push_back(temp_dct);
    }
}

cv::Mat LivePortrait::prepare_paste_back(const cv::Mat& mask_crop, cv::Mat& crop_M_c2o, 
    int dsize_w, int dsize_h) {
    cv::Mat mask_ori;
    transform_img(mask_crop, mask_ori, crop_M_c2o, dsize_h, dsize_w);
    mask_ori.convertTo(mask_ori, CV_32F, 1 / 255.f);
    return mask_ori;
}

void LivePortrait::stitching(std::vector<cv::Point3f>& kp_source, std::vector<cv::Point3f>& kp_driving, 
    std::vector<cv::Point3f>& kp_driving_new) {
    ncnn::Extractor ex = stitching_retargeting_.create_extractor();

    ncnn::Mat in0(126, 1);
    float* in_ptr = (float*)in0.data;
    for (int i = 0; i < 21; ++i) {
        in_ptr[i * 3] = kp_source[i].x;
        in_ptr[i * 3 + 1] = kp_source[i].y;
        in_ptr[i * 3 + 2] = kp_source[i].z;

        in_ptr[i * 3 + 63] = kp_driving[i].x;
        in_ptr[i * 3 + 63 + 1] = kp_driving[i].y;
        in_ptr[i * 3 + 63 + 2] = kp_driving[i].z;
    }
    
    ex.input("input", in0);

    ncnn::Mat delta;
    ex.extract("output", delta);

    kp_driving_new.resize(21);
    float* delta_ptr = (float*)delta.data;
    for (int i = 0; i < 21; ++i) {
        kp_driving_new[i].x = kp_driving[i].x + delta_ptr[i * 3] + delta_ptr[63];
        kp_driving_new[i].y = kp_driving[i].y + delta_ptr[i * 3 + 1] + delta_ptr[64];
        kp_driving_new[i].z = kp_driving[i].z + delta_ptr[i * 3 + 2];
    }

}

int LivePortrait::spade_generator(ncnn::Mat& feature, ncnn::Mat& out){
    ncnn::Extractor ex = spade_generator_module_.create_extractor();
    ex.input("input", feature);
    ex.extract("output", out);
    return 0;
}

void LivePortrait::paste_back(const cv::Mat& img_crop, const cv::Mat& M_c2o, 
    const cv::Mat& img_ori, const cv::Mat& mask_ori, cv::Mat& result) {
    transform_img(img_crop, result, M_c2o, img_ori.rows, img_ori.cols);
#pragma omp parallel for num_threads(4)
    for (int h = 0; h < result.rows; ++h) {
        cv::Vec3b* src_ptr = result.ptr<cv::Vec3b>(h);
        const float* mask_ptr = mask_ori.ptr<float>(h);
        const cv::Vec3b* img_ptr = img_ori.ptr<cv::Vec3b>(h);
        for (int w = 0; w < result.cols; ++w) {
            unsigned char R = src_ptr[w][0] * mask_ptr[w] + img_ptr[w][2] * (1.f - mask_ptr[w]);
            unsigned char G = src_ptr[w][1] * mask_ptr[w] + img_ptr[w][1] * (1.f - mask_ptr[w]);
            unsigned char B = src_ptr[w][2] * mask_ptr[w] + img_ptr[w][0] * (1.f - mask_ptr[w]);
            src_ptr[w][0] = B;
            src_ptr[w][1] = G;
            src_ptr[w][2] = R;
        }
    }
}

void LivePortrait::warp_decode(ncnn::Mat& feature_3d, std::vector<cv::Point3f>& kp_source_v, 
    std::vector<cv::Point3f>& kp_driving_v, ncnn::Mat& mesh,  cv::Mat& out_img)
{
    ncnn::Mat kp_driving(3, 21);
    ncnn::Mat kp_source(3, 21);
    float* kp_driving_ptr = (float*)kp_driving.data;
    float* kp_source_ptr = (float*)kp_source.data;
    for (int i = 0; i < 21; ++i) {
        kp_driving_ptr[i * 3] = kp_driving_v[i].x;
        kp_driving_ptr[i * 3 + 1] = kp_driving_v[i].y;
        kp_driving_ptr[i * 3 + 2] = kp_driving_v[i].z;
        kp_source_ptr[i * 3] = kp_source_v[i].x;
        kp_source_ptr[i * 3 + 1] = kp_source_v[i].y;
        kp_source_ptr[i * 3 + 2] = kp_source_v[i].z;
    }
    
    ncnn::Mat warp_feature;
    warping_motion(feature_3d, kp_driving, kp_source, mesh, warp_feature);

    ncnn::Mat out;
    spade_generator(warp_feature, out);

    cv::Mat I_p_i(cv::Size(512, 512), CV_32FC3, (void*)out.data);

    I_p_i *= 255;
    I_p_i.convertTo(out_img, CV_8UC3, 1.f);
}