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
    //retargeting info for mode=0
    //eye and lip close
    pipe_config.retargeting_info.eye_close_ratio = 0.35f;    //[0., 1.0]
    pipe_config.retargeting_info.lip_close_ratio = 0.f;      //[0., 1.0]
    //head pose rotation
    pipe_config.retargeting_info.head_pitch_variation = 0.f; //[-15.0, 15.0]
    pipe_config.retargeting_info.head_yaw_variation = 0.f;   //[-15.0, 15.0]
    pipe_config.retargeting_info.head_roll_variation = 0.f;  //[-15.0, 15.0]
    //head pose translation
    pipe_config.retargeting_info.mov_x = 0.1f;                //[-0.19, 0.19]
    pipe_config.retargeting_info.mov_y = 0.f;                //[-0.19, 0.19]
    pipe_config.retargeting_info.mov_z = 1.f;                //[0.9, 1.2]
    //lip pouting
    pipe_config.retargeting_info.lip_variation_zero = 0.f;   //[-0.09, 0.09]
    //lip pursing
    pipe_config.retargeting_info.lip_variation_one = 10.f;    //[-20.0, 15.0]
    //lip grin
    pipe_config.retargeting_info.lip_variation_two = 0.f;    //[0., 15.0]
    //lip close <-> open
    pipe_config.retargeting_info.lip_variation_three = 0.f;  //[-90.0, 120.0]

    pipe_config.retargeting_info.smile = 0.f;                //[-0.3, 1.3]
    pipe_config.retargeting_info.wink = 0.f;                 //[0., 39.0]
    pipe_config.retargeting_info.eyebrow = 0.f;              //[-30.0, 30.0]
    pipe_config.retargeting_info.eyeball_direction_x = 20.f;  //[-30.0, 30.0]
    pipe_config.retargeting_info.eyeball_direction_y = 0.f;  //[-63.0, 63.0]
    pipe_config.mode = 0; //0 for single image retarget lip and eye
                          //1 for single source image driving by video/images
    pipe_config.save_path = "./out/";
    pipe_config.driving_path = "../data/";
    pipe_config.source_path = "../1.jpg";
    pipe_config.mask_template_path = "../mask_template.png";
    pipe.run(pipe_config );


    return 0;

}