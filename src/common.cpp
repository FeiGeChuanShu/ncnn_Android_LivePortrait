#include <numeric>
#include "common.h"
#include "cpu.h"
cv::Rect adjust_boundingBox(const cv::Rect& boundingBox, int img_w, int img_h) {
    auto w = boundingBox.width;
    auto h = boundingBox.height;
    cv::Rect box = boundingBox;
    
    box.x -= static_cast<int>(0.1 * w);
    box.y -= static_cast<int>(0.05 * h);

    box.width += static_cast<int>(0.2 * w);
    box.height += static_cast<int>(0.1 * h);

    if (box.width < box.height) {
        auto dx = (box.height - box.width);
        box.x -= dx / 2;
        box.width += dx;
    }
    else {
        auto dy = (box.width - box.height);
        box.y -= dy / 2;
        box.height += dy;
    }

    box.x = std::max(0, box.x);
    box.y = std::max(0, box.y);
    box.width = box.x + box.width < img_w ? box.width : img_w - box.x - 1;
    box.height = box.y + box.height < img_h ? box.height : img_h - box.y - 1;
    return box;
}

void concat(const std::vector<ncnn::Mat>& bottoms, ncnn::Mat& c, int axis){
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Concat");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, axis);// axis

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}

void reshape(const ncnn::Mat& in, ncnn::Mat& out, int w, int h, int d, int c){
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_packing_layout = true;
    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, w);// w
    pd.set(1, h);// h
    pd.set(11,d);// d
    pd.set(2, c);// c
    pd.set(3, 0);// 
    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in.elempack * in.w;
        if (dims == 2) elemcount = in.elempack * in.h;
        if (dims == 3) elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        if (op->support_packing)
        {
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx()))
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
        }

        if (in.elempack != dst_elempack)
        {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }
    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}


int make_coordinate_grid(int d, int h, int w, ncnn::Mat& mesh) {
    ncnn::Mat xx = ncnn::Mat(w, h, d);
    ncnn::Mat yy = ncnn::Mat(w, h, d);
    ncnn::Mat zz = ncnn::Mat(w, h, d);
    for (int i = 0; i < d; i++){
        for (int j = 0; j < h; j++){
            float* x_ptr = xx.channel(i).row(j);
            for (int k = 0; k < w; k++){
                x_ptr[k] = 2.f * (float)k / ((float)w - 1.f) - 1.f;

                float* y_ptr = yy.channel(i).row(k);
                y_ptr[j] = 2.f * (float)k / ((float)h - 1.f) - 1.f;
            }
        }
        zz.channel(i).fill(2.f * (float)i / ((float)d - 1.f) - 1.f);
    }

    std::vector<ncnn::Mat> bottoms = { xx.reshape(1, w, h, d), yy.reshape(1, w, h, d), zz.reshape(1, w, h, d) };
    mesh.create(3, w, h, 16);
    concat(bottoms, mesh, 3);
    reshape(mesh, mesh, 3, 64 * 64, 16, 1);
    return 0;
}


void parse_pt2_from_pt106(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2) {

    cv::Point2f pt_left_eye, pt_right_eye;
    pt_left_eye.x = (pts[33].x + pts[35].x + pts[40].x + pts[39].x) / 4;
    pt_left_eye.y = (pts[33].y + pts[35].y + pts[40].y + pts[39].y) / 4;
    pt_right_eye.x = (pts[87].x + pts[89].x + pts[94].x + pts[93].x) / 4;
    pt_right_eye.y = (pts[87].y + pts[89].y + pts[94].y + pts[93].y) / 4;

    cv::Point2f pt_center_eye = (pt_left_eye + pt_right_eye) / 2;
    cv::Point2f pt_center_lip = (pts[52] + pts[61]) / 2;

    pt2.push_back(pt_center_eye);
    pt2.push_back(pt_center_lip);
}
void parse_pt2_from_pt203(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2) {

    cv::Point2f pt_left_eye, pt_right_eye;
    pt_left_eye.x = (pts[0].x + pts[6].x + pts[12].x + pts[18].x) / 4;
    pt_left_eye.y = (pts[0].y + pts[6].y + pts[12].y + pts[18].y) / 4;
    pt_right_eye.x = (pts[24].x + pts[30].x + pts[36].x + pts[42].x) / 4;
    pt_right_eye.y = (pts[24].y + pts[30].y + pts[36].y + pts[42].y) / 4;

    cv::Point2f pt_center_eye = (pt_left_eye + pt_right_eye) / 2;
    cv::Point2f pt_center_lip = (pts[48] + pts[66]) / 2;

    pt2.push_back(pt_center_eye);
    pt2.push_back(pt_center_lip);
}


void parse_pt2_from_pt_x(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& pt2) {
    if (pts.size() == 106) {
        parse_pt2_from_pt106(pts, pt2);
    }
    else if (pts.size() == 203) {
        parse_pt2_from_pt203(pts, pt2);
    }
}


void parse_rect_from_landmark(const std::vector<cv::Point2f>& pts, float scale, float vx_ratio, float vy_ratio, rect_info_t& rect_info) {
    std::vector<cv::Point2f> pt2;
    parse_pt2_from_pt_x(pts, pt2);

    cv::Point2f uy = pt2[1] - pt2[0];
    float l = std::sqrt(uy.x * uy.x + uy.y * uy.y);
    if (l < 0.001)
        uy = cv::Point2f(0, 1);
    else
        uy /= l;
    cv::Point2f ux = cv::Point2f(uy.y, -uy.x);

    float angle = std::acos(ux.x);
    if (ux.y < 0)
        angle = -angle;

    cv::Point2f center0;
    center0.x = std::accumulate(pts.begin(), pts.end(), 0.f, [](float cur, const cv::Point2f & pt){ return cur + pt.x; }) / pts.size();
    center0.y = std::accumulate(pts.begin(), pts.end(), 0.f, [](float cur, const cv::Point2f & pt) { return cur + pt.y; }) / pts.size();

    std::vector<cv::Point2f> rpts;
    for (size_t i = 0; i < pts.size(); ++i) {
        float x = (pts[i].x - center0.x) * ux.x + (pts[i].y - center0.y) * ux.y;
        float y = (pts[i].x - center0.x) * uy.x + (pts[i].y - center0.y) * uy.y;
        rpts.emplace_back(x, y);
    }
    cv::Rect_<float> box = cv::boundingRect(rpts);
    cv::Point2f center1;
    center1.x = box.x + box.width / 2;
    center1.y = box.y + box.height / 2;

    cv::Point2f size;
    size.x = std::max(box.width, box.height) * scale;
    size.y = size.x;

    cv::Point2f center;
    center.x = center0.x + ux.x * center1.x + uy.x * center1.y;
    center.y = center0.y + ux.y * center1.x + uy.y * center1.y;
    center.x = center.x + ux.x * (vx_ratio * size.x) + uy.x * (vy_ratio * size.x);
    center.y = center.y + ux.y * (vx_ratio * size.y) + uy.y * (vy_ratio * size.y);

    rect_info.center = center;
    rect_info.size = size;
    rect_info.angle = angle;

}

void estimate_similar_transform_from_pts(const std::vector<cv::Point2f>& pts,cv::Mat& M, int dsize, float scale, float vx_ratio, float vy_ratio) {

    rect_info_t rect_info;
    parse_rect_from_landmark(pts, scale, vx_ratio, vy_ratio, rect_info);

    float s = (float)dsize / rect_info.size.x;
    cv::Point2f tgt_center;
    tgt_center.x = (float)dsize / 2;
    tgt_center.y = (float)dsize / 2;

    float costheta = std::cos(rect_info.angle);
    float sintheta = std::sin(rect_info.angle);

    float cx = rect_info.center.x;
    float cy = rect_info.center.y;

    float tcx = tgt_center.x;
    float tcy = tgt_center.y;

    M = (cv::Mat_<float>(3, 3) << s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy),
                                -s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy), 0, 0, 1);
}

void transform_pts(const std::vector<cv::Point2f>& pts, std::vector<cv::Point2f>& dst_pts, const cv::Mat& M) {
    for (size_t i = 0; i < pts.size(); ++i) {
        float x = pts[i].x * M.at<float>(0, 0) + pts[i].y * M.at<float>(0, 1) + M.at<float>(0, 2);
        float y = pts[i].x * M.at<float>(1, 0) + pts[i].y * M.at<float>(1, 1) + M.at<float>(1, 2);
        dst_pts.emplace_back(x, y);
    }
}
void transform_img(const cv::Mat& img, cv::Mat& out, const cv::Mat& M, int dsize_h, int dsize_w) {

    cv::warpPerspective(img, out, M, cv::Size(dsize_w, dsize_h));
}
void transform_img(const cv::Mat& img, cv::Mat& out, const cv::Mat& M, int dsize) {

    cv::warpPerspective(img, out, M, cv::Size(dsize, dsize) );
}

cv::Mat get_rotation_matrix(float pitch_, float yaw_, float roll_) {
    float pitch = pitch_ / 180 * PI;
    float yaw = yaw_ / 180 * PI;
    float roll = roll_ / 180 * PI;

    cv::Mat rot_x = (cv::Mat_<float>(3, 3) << 1.f, 0.f, 0.f,
        0.f, std::cos(pitch), -std::sin(pitch),
        0.f, std::sin(pitch), std::cos(pitch));
    cv::Mat rot_y = (cv::Mat_<float>(3, 3) << std::cos(yaw), 0.f, std::sin(yaw),
        0.f, 1.f, 0.f,
        -std::sin(yaw), 0.f, std::cos(yaw));
    cv::Mat rot_z = (cv::Mat_<float>(3, 3) << std::cos(roll), -std::sin(roll), 0.f,
        std::sin(roll), std::cos(roll), 0.f,
        0.f, 0.f, 1.f);

    cv::Mat rot = rot_z * rot_y * rot_x;

    return rot.t();
}

void transform_keypoint(kp_info_t& kp_info, std::vector<cv::Point3f>& kp_transformed) {
    kp_transformed.clear();
    cv::Mat R_s = get_rotation_matrix(kp_info.pitch, kp_info.yaw, kp_info.roll);
    int num_kp = kp_info.kp.size();
    for (int i = 0; i < num_kp; ++i) {
        float x = kp_info.kp[i].x * R_s.at<float>(0, 0) +
            kp_info.kp[i].y * R_s.at<float>(1, 0) +
            kp_info.kp[i].z * R_s.at<float>(2, 0) +
            kp_info.exp[i].x;
        float y = kp_info.kp[i].x * R_s.at<float>(0, 1) +
            kp_info.kp[i].y * R_s.at<float>(1, 1) +
            kp_info.kp[i].z * R_s.at<float>(2, 1) +
            kp_info.exp[i].y;
        float z = kp_info.kp[i].x * R_s.at<float>(0, 2) +
            kp_info.kp[i].y * R_s.at<float>(1, 2) +
            kp_info.kp[i].z * R_s.at<float>(2, 2) +
            kp_info.exp[i].z;
        kp_transformed.emplace_back(x * kp_info.scale + kp_info.t[0], y * kp_info.scale + kp_info.t[1], z * kp_info.scale);
    }

}

float calculate_distance_ratio(std::vector<cv::Point2f>& lmk, int idx1, int idx2, int idx3, int idx4, float eps) {
    float d1 = std::sqrt((lmk[idx1].x - lmk[idx2].x) * (lmk[idx1].x - lmk[idx2].x) + (lmk[idx1].y - lmk[idx2].y) * (lmk[idx1].y - lmk[idx2].y));
    float d2 = std::sqrt((lmk[idx3].x - lmk[idx4].x) * (lmk[idx3].x - lmk[idx4].x) + (lmk[idx3].y - lmk[idx4].y) * (lmk[idx3].y - lmk[idx4].y));
    return d1 / (d2 + eps);
}
