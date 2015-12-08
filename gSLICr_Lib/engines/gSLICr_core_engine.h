// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_seg_engine_GPU.h"


namespace gSLICr
{
    namespace engines
    {
        class core_engine
        {
        private:

            seg_engine* slic_seg_engine;

        public:

            core_engine(const objects::settings& in_settings);
            ~core_engine();

            // Function to segment in_img
            void Process_Frame(NeuronImage* in_img_red, NeuronImage* in_img_green);

            // Function to get the pointer to the segmented mask image
            const IntImage4D * Get_Seg_Res();
        };
    }
}

