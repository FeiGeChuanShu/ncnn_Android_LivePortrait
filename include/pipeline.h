#ifndef PIPELINE_H_
#define PIPELINE_H_
#include <memory>
#include "liveportrait.h"

typedef struct _pipeline_config{
    retargeting_info_t retargeting_info;
    int mode;
    std::string mask_template_path;
    std::string source_path;
    std::string driving_path;
    std::string save_path;
}pipeline_config_t;

class Pipeline
{
public:
    Pipeline() = default;
    ~Pipeline();
    int init(const char* model_dir);
    int run(pipeline_config_t& pipe_config);
private:
    int load_driving_data(const std::string&  driving_path, std::vector<cv::Mat>& driving);
    std::shared_ptr<LivePortrait> liveportrait_ = nullptr;
};

#endif // PIPELINE_H_
