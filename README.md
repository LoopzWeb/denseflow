# denseflow
use opencv3.0 to compute denseflow
## inference
>1.https://github.com/wanglimin/dense_flow
>I first use this code to get dense flow.But I met something wrong when make this project,because the author uses opencv2 and I only install opencv3 in my computer.There are a lot of differences between opencv2 and opencv3.I warpped the code with the inference2.
>2.https://github.com/Katou2/Optical_Flow_GPU_Opencv3
## sample
Usage:
>./denseFlow_gpu -f=test.avi -x=tmp/flow_x -y=tmp/flow_x -i=tmp/image -b=20 -t=1 -d=0 -s=1
>test.avi: input video
>tmp: folder containing RGB images and optical flow images
