// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "../gSLICr_defines.h"
#include "../objects/gSLICr_settings.h"
#include "../objects/gSLICr_spixel_info.h"

#define AssertExit(cond) if(!cond) { printf("File %s, line %i, assert failed.\n", __FILE__, __LINE__); return -1; }

namespace gSLICr
{
	namespace engines
	{
		class seg_engine
		{
		protected:

			// normalizing distances
			float max_color_dist;
			float max_xyz_dist;

			// images
            NeuronImage *src_img_red; // original source image
            NeuronImage *src_img_green;
            IntImage4D *idx_img;

			// superpixel map
			SpixelMap* spixel_map;
			int spixel_size;

			objects::settings gSLICr_settings;

			virtual void Init_Cluster_Centers() = 0;
			virtual void Find_Center_Association() = 0;
			virtual void Update_Cluster_Center() = 0;
			virtual void Enforce_Connectivity() = 0;

		public:

			seg_engine(const objects::settings& in_settings );
			virtual ~seg_engine();

            const IntImage4D* Get_Seg_Mask() const {
				idx_img->UpdateHostFromDevice();
				return idx_img;
			};

            void Perform_Segmentation(NeuronImage* in_img_red, NeuronImage* in_img_green);
			virtual void Draw_Segmentation_Result(UChar4Image* out_img){};

        public:
            int Test_Init_Clusters();
            int Test_Find_Center();
            virtual int Test_Update_Clusters() { return 0; }
            virtual int Test_Enforce_Connectivity() { return 0; }
        };
	}
}

