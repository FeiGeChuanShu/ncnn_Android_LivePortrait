# LivePortrait_ncnn
a naive example of LivePortrait infer by ncnn

## Run mode :  
1.Retargeting(lip open/eye open/head pose editing)  
2.Source image driving by video/images  

PS: infer time: 28s/driving image
## Time profile  
```
op type                    avg time(ms)      %
Conv3d(no simd/vulkan)     27168.23         96.24%
other op                   1059.54          3.76%
total time:                28227.77
```

## Result
### 1.Retargeting  
![](result/retargeting_res.jpg)  
### 2.Source image driving by video  
![](result/d14_res2.gif)  
![](result/d14_res.gif)  

## Reference  
1.https://github.com/KwaiVGI/LivePortrait  
2.https://github.com/Tencent/ncnn  

