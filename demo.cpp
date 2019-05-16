#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
//#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/core/cvstd.hpp"

#include <stdio.h>
#include <iostream>

using namespace cv;
//sing namespace cv::gpu;
using namespace cv::cuda;


static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

int main(int argc, char** argv){
	// IO operation
	const char* keys =
		{
		    "{h help|   |print this messages}"
			"{f vidFile      | ex2.avi | filename of video }"
			"{x xFlowFile    | flow_x | filename of flow x component }"
			"{y yFlowFile    | flow_y | filename of flow x component }"
			"{i imgFile      | flow_i | filename of flow image}"
			"{b bound | 15 | specify the maximum of optical flow}"
			"{t type | 0 | specify the optical flow algorithm;\n0:FarnebackOpticalFlow\n1:OpticalFlowDual_TVL1\n2:BroxOpticalFlow }"
			"{d device_id    | 0  | set gpu id}"
			"{s step  | 1 | specify the step for frame sampling}"
		};

	CommandLineParser cmd(argc, argv, keys);
	if (cmd.get<bool>("h"))
	{
		cout << "Usage: compute_flow [options]" << endl;
		cout << "Avaible options:" << endl;
		cmd.printMessage();
		return 0;
	}
	string vidFile = cmd.get<std::string>("f");
	string xFlowFile = cmd.get<std::string>("xFlowFile");
	string yFlowFile = cmd.get<std::string>("yFlowFile");
	string imgFile = cmd.get<std::string>("imgFile");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
    //print args
    cout<<"videofilename:"<<vidFile<<endl;
    cout<<"filename of flow_x:"<<xFlowFile<<endl;
    cout<<"filename of flow_y:"<<yFlowFile<<endl;
    cout<<"filename of image:"<<imgFile<<endl;
    cout<<"bound:"<<bound<<endl;
    cout<<"image:"<<imgFile<<endl;
    cout<<"type:"<<type<<endl;
    cout<<"device_id:"<<device_id<<endl;
    cout<<"step:"<<step<<endl;


	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y,flow_cpu,planes[3];
	GpuMat frame_0, frame_1, flow_u, flow_v,flow_gpu;

	setDevice(device_id);
	//FarnebackOpticalFlow alg_farn;
	Ptr<cuda::FarnebackOpticalFlow> alg_farn=cuda::FarnebackOpticalFlow::create();
	//OpticalFlowDual_TVL1_GPU alg_tvl1;
	Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1=cuda::OpticalFlowDual_TVL1::create();
	//BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
	Ptr<cuda::BroxOpticalFlow> alg_brox=cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

	while(true) {
		capture >> frame;
		cout<<"frame_num:"<<frame_num<<endl;
		if(frame.empty())
			break;
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			//?
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while (step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;
		}

		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

               //  Mat prev_grey_, grey_;
               //  resize(prev_grey, prev_grey_, Size(453, 342));
               //  resize(grey, grey_, Size(453, 342));
        //GpuMat
		frame_0.upload(prev_grey);
		frame_1.upload(grey);


        // GPU optical flow
		switch(type){
		case 0:
			//alg_farn(frame_0,frame_1,flow_u,flow_v);
			alg_farn->calc(frame_0,frame_1,flow_gpu);
			break;
		case 1:
			//alg_tvl1(frame_0,frame_1,flow_u,flow_v);
			alg_tvl1->calc(frame_0,frame_1,flow_gpu);
			break;
		case 2:
			GpuMat d_frame0f, d_frame1f;
	        frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
	        frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
			//alg_brox(d_frame0f, d_frame1f, flow_u,flow_v);
			alg_brox->calc(d_frame0f,d_frame1f,flow_gpu);
			break;
		}

		//flow_u.download(flow_x);
		//flow_v.download(flow_y);
		flow_gpu.download(flow_cpu);
		cv::split(flow_cpu,planes);
		flow_x=planes[0];
		flow_y=planes[1];

		// Output optical flow
		Mat imgX(flow_x.size(),CV_8UC1);
		Mat imgY(flow_y.size(),CV_8UC1);
		convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%05d.jpg",int(frame_num));

		// Mat imgX_, imgY_, image_;
		// resize(imgX,imgX_,cv::Size(340,256));
		// resize(imgY,imgY_,cv::Size(340,256));
		// resize(image,image_,cv::Size(340,256));

		imwrite(xFlowFile + tmp,imgX);
		imwrite(yFlowFile + tmp,imgY);
		imwrite(imgFile + tmp, image);

		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1){
			capture >> frame;
			step_t--;
		}
	}
	return 0;
}
