// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine.h"

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;


seg_engine::seg_engine(const objects::settings& in_settings)
{
	gSLICr_settings = in_settings;
}


seg_engine::~seg_engine()
{
    delete src_img_red;
    delete src_img_green;
	delete idx_img;
	delete spixel_map;
}

void seg_engine::Perform_Segmentation(NeuronImage* in_img_red, NeuronImage* in_img_green)
{
    src_img_red->SetFrom(in_img_red, ORUtils::MemoryBlock<neur>::CPU_TO_CUDA);
    src_img_green->SetFrom(in_img_green, ORUtils::MemoryBlock<neur>::CPU_TO_CUDA);
    
	Init_Cluster_Centers();
    //Test_Init_Clusters();

	Find_Center_Association();
    //Test_Display_Pixel_Membership();

	for (int i = 0; i < gSLICr_settings.no_iters; i++)
	{
		Update_Cluster_Center();
        //Test_Display_Pixel_Membership();

		Find_Center_Association();
        //Test_Display_Pixel_Membership();
    }

    if (gSLICr_settings.do_enforce_connectivity)
    {
        Enforce_Connectivity();
        //Test_Enforce_Connectivity();
    }
    //Test_Display_Pixel_Membership();
    cudaThreadSynchronize();
}

int seg_engine::Test_Init_Clusters()
{
    ORcudaSafeCall(cudaPeekAtLastError());
    ORcudaSafeCall(cudaDeviceSynchronize());

    spixel_map->UpdateHostFromDevice();

    const spixel_info* sdata = spixel_map->GetData(MEMORYDEVICE_CPU);
    printf("\n");
    for (size_t i = 0; i < spixel_map->dataSize; i++)
    {
        printf("i = %d, xyz = %f %f %f, n = %d, id = %d, red = %f, ",
            i,
            sdata[i].center.x,
            sdata[i].center.y,
            sdata[i].center.z,
            sdata[i].no_pixels,
            sdata[i].id,
            sdata[i].red_color);
        printf("green = ");
        for (size_t j = 0; j < IMAGE_TIME; j++)
        {
            printf("%f ", sdata[i].green_color[j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}

int seg_engine::Test_Display_Pixel_Membership()
{
    ORcudaSafeCall(cudaPeekAtLastError());
    ORcudaSafeCall(cudaDeviceSynchronize());

    idx_img->UpdateHostFromDevice();

    Vector4i img_size = idx_img->noDims;
    int max2d_display = 50;
    int max3d_display = 2;
    size_t z_step = img_size.x * img_size.y;
    const int* data = idx_img->GetData(MEMORYDEVICE_CPU);
    printf("\n");
    for (size_t k = 0; k < min(max3d_display, img_size.z); k++)
    {
        printf("--------------- Z = %d ---------------\n", k);
        for (size_t j = 0; j < min(max2d_display, img_size.y); j++)
        {
            for (size_t i = 0; i < min(max2d_display, img_size.x); i++)
            {
                printf("%02d;", data[k * z_step + j * img_size.x + i]);
            }
            printf("\n");
        }
    }
    
    printf("\n");

    return 0;
}