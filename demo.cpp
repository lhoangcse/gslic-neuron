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

    AssertExitB(is.is_open());

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

bool write_seg_res(const char* file, const gSLICr::IntImage4D* seg_res)
{
    const int* seg_index = seg_res->GetData(MEMORYDEVICE_CPU);

    ofstream os(file, ios::binary);
    AssertExitB(os.is_open());

    os.write((char*)seg_index, seg_res->dataSize * sizeof(int));
    os.close();

    return true;
}


int main()
{
    string base_dir = "D:\\Git\\gslic-neuron\\data\\";
    string red_name = "neuron-red-avg.bin";
    string green_name = "neuron-green-norm.bin";
    string seg_name = "seg_res.bin";
#ifdef USE_FAKE_DATA
    base_dir = base_dir + "fake\\";
#else
    base_dir = base_dir + "real\\";
#endif
    string red_file   = base_dir + red_name;
    string green_file = base_dir + green_name;
    string seg_file   = base_dir + seg_name;
    int w = IMAGE_WIDTH, h = IMAGE_HEIGHT, d = IMAGE_DEPTH, t = 1;
    gSLICr::NeuronImage* red_img;
    gSLICr::Vector4i red_img_size;
    AssertExitI(load_neuron_image(red_file.c_str(), red_img, red_img_size, w, h, d, t));

    t = IMAGE_TIME;
    gSLICr::NeuronImage* green_img;
    gSLICr::Vector4i green_img_size;
    AssertExitI(load_neuron_image(green_file.c_str(), green_img, green_img_size, w, h, d, t));

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
	my_settings.no_iters = 5;
	my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
    my_settings.seg_method = gSLICr::GIVEN_NUM; // or gSLICr::GIVEN_NUM for given number
	my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step

	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(my_settings);

    StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
    
    sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
    gSLICr_engine->Process_Frame(red_img, green_img);
    sdkStopTimer(&my_timer); 
    cout<<"\rsegmentation in:["<<sdkGetTimerValue(&my_timer)<<"]ms"<<flush;
        
    const gSLICr::IntImage4D* seg_res = gSLICr_engine->Get_Seg_Res();
    AssertExitI(write_seg_res(seg_file.c_str(), seg_res));

    return 0;
}
