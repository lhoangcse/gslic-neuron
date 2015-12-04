// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_core_engine.h"
#include <fstream>

using namespace gSLICr;
using namespace std;

gSLICr::engines::core_engine::core_engine(const objects::settings& in_settings)
{
	slic_seg_engine = new seg_engine_GPU(in_settings);
}

gSLICr::engines::core_engine::~core_engine()
{
		delete slic_seg_engine;
}

void gSLICr::engines::core_engine::Process_Frame(NeuronImage* in_img_red, NeuronImage* in_img_green)
{
    slic_seg_engine->Perform_Segmentation(in_img_red, in_img_green);
}

const IntImage4D * gSLICr::engines::core_engine::Get_Seg_Res()
{
	return slic_seg_engine->Get_Seg_Mask();
}
