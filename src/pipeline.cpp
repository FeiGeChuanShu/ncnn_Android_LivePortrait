#include "pipeline.h"
#include <sys/stat.h>
#include <dirent.h>
Pipeline::~Pipeline(){

}
int Pipeline::init(const char* model_dir)
{
    if(liveportrait_ == nullptr)
        liveportrait_ = std::make_shared<LivePortrait>();
    else
        liveportrait_.reset(new LivePortrait());

    int ret = liveportrait_->load_model(model_dir);

    return ret;
}
int Pipeline::load_driving_data(const std::string& driving_path, std::vector<cv::Mat>& driving)
{
    struct stat infos;
    if(stat(driving_path.c_str(), &infos) != 0)
        return -1;
    else if(infos.st_mode & S_IFDIR){
        
        std::vector<std::pair<int,std::string>> driving_names;
        DIR* dir = opendir(driving_path.c_str());
        if (!dir) {
            return -1;
        }

        struct dirent* ent;
        while ((ent = readdir(dir)) != NULL) {
            std::string entry_name(ent->d_name);
            if (entry_name == "." || entry_name == "..") continue;

            if (entry_name.size() > 4 &&
                entry_name.compare(entry_name.size() - 4, 4, ".jpg") == 0) {

                driving_names.emplace_back(stoi(entry_name.substr(0, entry_name.length()-4)), driving_path + "/" + entry_name);
            }
        }
        closedir(dir);
        std::sort(driving_names.begin(), driving_names.end(), 
            [](const std::pair<int, std::string>& n1, const std::pair<int, std::string>& n2)
                {return n1.first < n2.first;});
        for(auto name : driving_names){
            cv::Mat frame = cv::imread(name.second);
            cv::Mat frame_rgb;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
            driving.push_back(frame_rgb);
        }
    }
    else if(infos.st_mode & S_IFREG){
        cv::VideoCapture cap(driving_path);
        while (true){
            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
                break;
            cv::Mat frame_rgb;
            cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
            driving.push_back(frame_rgb);
        }
        cap.release();
    }
    else
        return -1;
    

    return 0;
}
int Pipeline::run(pipeline_config_t& pipe_config)
{
    cv::Mat source = cv::imread(pipe_config.source_path);
    if (source.empty()) {
        fprintf(stderr, "source is empty\n");
        return -1;
    }

    cv::Mat mask = cv::imread(pipe_config.mask_template_path, cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        fprintf(stderr, "mask is empty\n");
        return -1;
    }

    int ret = 0;
    if (pipe_config.mode == 0) {
        cv::Mat out;
        ret = liveportrait_->run_single_iamge(source, mask, out, pipe_config.lip_close_ratio, pipe_config.eye_close_ratio);
        if(ret < 0)
            return -1;
        cv::imwrite(pipe_config.save_path + "/retargeting.jpg", out);
    }
    else if(pipe_config.mode == 1){
        std::vector<cv::Mat> driving;
        ret = load_driving_data(pipe_config.driving_path, driving);
        if(driving.size() == 0){
            fprintf(stderr, "no driving data loaded\n");
            return -1;
        }
        fprintf(stderr, "load %d frames\n", driving.size());
        
        ret = liveportrait_->run_multi_image(source, mask, driving, pipe_config.save_path);
    }
    else
        return -1;

    return 0;
}