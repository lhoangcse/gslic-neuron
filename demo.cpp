// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include <time.h>
#include <stdio.h>

#include "gSLICr_Lib/gSLICr.h"
#include "NVTimer.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include <fstream>

using namespace std;
using namespace cv;


void load_image(const Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<Vec3b>(y, x)[2];
		}
}

void load_image(const gSLICr::UChar4Image* inimg, Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}

bool load_neuron_image(
    const char* file,
    gSLICr::NeuronImage*& img_out,
    gSLICr::Vector4i& img_size,
    int w, int h, int d, int t)
{
    int size[4] = { w, h, d, t };
    img_size = size;

    // allocate image and fill buffer from binary file
    img_out = new gSLICr::NeuronImage(img_size, true, false);

    ifstream is(file, ios::binary);

    // get length of file
    is.seekg(0, ios::end);
    int length = is.tellg();
    is.seekg(0, ios::beg);

    AssertExitB(length == img_out->dataSize * sizeof(gSLICr::neur));

    gSLICr::neur* buffer = img_out->GetData(MEMORYDEVICE_CPU);
    is.read((char*)buffer, length);
    is.close();

    return true;
}


int main()
{
#ifdef USE_FAKE_DATA
    const char* red_file   = "D:\\Git\\gslic-neuron\\data\\fake\\neuron-red-avg.bin";
    const char* green_file = "D:\\Git\\gslic-neuron\\data\\fake\\neuron-green-norm.bin";
#else
    const char* red_file   = "D:\\Git\\gslic-neuron\\data\\real\\neuron-red-avg.bin";
    const char* green_file = "D:\\Git\\gslic-neuron\\data\\real\\neuron-green-norm.bin";
#endif
    int w = IMAGE_WIDTH, h = IMAGE_HEIGHT, d = IMAGE_DEPTH, t = 1;
    gSLICr::NeuronImage* red_img;
    gSLICr::Vector4i red_img_size;
    AssertExitI(load_neuron_image(red_file, red_img, red_img_size, w, h, d, t));

    t = IMAGE_TIME;
    gSLICr::NeuronImage* green_img;
    gSLICr::Vector4i green_img_size;
    AssertExitI(load_neuron_image(green_file, green_img, green_img_size, w, h, d, t));

    AssertExitI(red_img_size.x == green_img_size.x);
    AssertExitI(red_img_size.y == green_img_size.y);
    AssertExitI(red_img_size.z == green_img_size.z);

	// gSLICr settings
	gSLICr::objects::settings my_settings;
    my_settings.img_size_green = green_img_size;
    my_settings.img_size_red = red_img_size;
    my_settings.no_segs = 75;
	my_settings.spixel_size = 16;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 10;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

    //Mat display_red_img(IMAGE_HEIGHT, IMAGE_WIDTH,
    //    CV_32SC1, red_img->GetData(MEMORYDEVICE_CPU));
    //imshow("Original Red Image", display_red_img);
    //waitKey(0);

	//// gSLICr takes gSLICr::UChar4Image as input and out put
	//gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	//gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);

	//Size s(my_settings.img_size.x, my_settings.img_size.y);
	//Mat oldFrame, frame;
	//Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);

    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
    
    //oldFrame = imread("photo.jpg");
    //resize(oldFrame, frame, s);
	//load_image(frame, in_img);
    //imshow("image", frame);

    sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
    gSLICr_engine->Process_Frame(red_img, green_img);
    sdkStopTimer(&my_timer); 
    cout<<"\rsegmentation in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
        
	//gSLICr_engine->Draw_Segmentation_Result(out_img);

	//load_image(out_img, boundry_draw_frame);
    //imshow("segmentation", boundry_draw_frame);
    
    //waitKey(0);
    
	//destroyAllWindows();
    return 0;
}
