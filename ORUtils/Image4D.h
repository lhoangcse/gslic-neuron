#pragma once


#include "MemoryBlock.h"

#ifndef __METALC__

namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type
	*/
	template <typename T>
	class Image4D : public MemoryBlock < T >
	{
	public:
		/** Size of the image in pixels. */
		Vector4<int> noDims;

		/** Initialize an empty image of the given size, either
		on CPU only or on both CPU and GPU.
		*/
        Image4D(Vector4<int> noDims, bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(noDims.x * noDims.y * noDims.z * noDims.w, allocate_CPU, allocate_CUDA, metalCompatible)
		{
			this->noDims = noDims;
		}

        Image4D(bool allocate_CPU, bool allocate_CUDA, bool metalCompatible = true)
			: MemoryBlock<T>(1, allocate_CPU, allocate_CUDA, metalCompatible)
		{
            this->noDims = Vector4<int>(1, 1, 1, 1);  //TODO - make nicer
		}

        Image4D(Vector4<int> noDims, MemoryDeviceType memoryType)
            : MemoryBlock<T>(noDims.x * noDims.y * noDims.z * noDims.w, memoryType)
		{
			this->noDims = noDims;
		}

		// Suppress the default copy constructor and assignment operator
        Image4D(const Image4D&);
        Image4D& operator=(const Image4D&);
	};
}

#endif
