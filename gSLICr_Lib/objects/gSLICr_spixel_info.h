// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "../gSLICr_defines.h"

namespace gSLICr
{
	namespace objects
	{
		struct spixel_info
		{
			Vector3f center;
			Vector4f color_info;
            neur red_color;
			int id;
			int no_pixels;
		};
	}

	typedef ORUtils::Image4D<objects::spixel_info> SpixelMap;
}