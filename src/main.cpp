#include "pipeline.h"

int main()
{
    Pipeline pipe;
    int ret = pipe.init("../model/");
    if(ret < 0){
        fprintf(stderr, "pipeline init failed\n");
        return -1;
    }
    fprintf(stderr, "init success\n");

    pipeline_config_t pipe_config;
    pipe_config.eye_close_ratio = 0.35f;
    pipe_config.lip_close_ratio = 0.35f;
    pipe_config.mode = 0; //0 for single image retarget lip and eye
                          //1 for single source image driving by video/images
    pipe_config.save_path = "./out/";
    pipe_config.driving_path = "../data/";
    pipe_config.source_path = "../1.jpg";
    pipe_config.mask_template_path = "../mask_template.png";
    pipe.run(pipe_config );


    return 0;

}