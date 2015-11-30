// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine.h"

namespace gSLICr
{
	namespace engines
	{
		class seg_engine_GPU : public seg_engine
		{
		private:

			int no_grid_per_center;
            SpixelMap* accum_map;
            IntImage4D* tmp_idx_img;

		protected:
			void Init_Cluster_Centers();
			void Find_Center_Association();
			void Update_Cluster_Center();
			void Enforce_Connectivity();

		public:

			seg_engine_GPU(const objects::settings& in_settings);
			~seg_engine_GPU();

			void Draw_Segmentation_Result(UChar4Image* out_img);

        public:
            int Test_Update_Clusters();
            int Test_Enforce_Connectivity();
		};
	}
}

