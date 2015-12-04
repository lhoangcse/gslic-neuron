// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once

#include "../ORUtils/PlatformIndependence.h"
#include "../ORUtils/Vector.h"
#include "../ORUtils/Matrix.h"
#include "../ORUtils/Image.h"
#include "../ORUtils/Image4D.h"
#include "../ORUtils/MathUtils.h"
#include "../ORUtils/MemoryBlock.h"

//------------------------------------------------------
// 
// Compile time GPU Settings, don't touch it!
//
//------------------------------------------------------

#ifndef BLOCK_DIM
#define BLOCK_DIM		8
#endif

//#define USE_FAKE_DATA

#ifdef USE_FAKE_DATA
#define IMAGE_WIDTH 20
#define IMAGE_HEIGHT 20
#define IMAGE_DEPTH 5
#define IMAGE_TIME 10
#else
#define IMAGE_WIDTH 120
#define IMAGE_HEIGHT 120
#define IMAGE_DEPTH 17
#define IMAGE_TIME 310
#endif

namespace gSLICr
{
	//------------------------------------------------------
	// 
	// math defines
	//
	//------------------------------------------------------

	typedef unsigned char uchar;
	typedef unsigned short ushort;
	typedef unsigned int uint;
	typedef unsigned long ulong;
    typedef float neur;

	typedef class ORUtils::Matrix3<float> Matrix3f;
	typedef class ORUtils::Matrix4<float> Matrix4f;

	typedef class ORUtils::Vector2<short> Vector2s;
	typedef class ORUtils::Vector2<int> Vector2i;
	typedef class ORUtils::Vector2<float> Vector2f;
	typedef class ORUtils::Vector2<double> Vector2d;

	typedef class ORUtils::Vector3<short> Vector3s;
	typedef class ORUtils::Vector3<double> Vector3d;
	typedef class ORUtils::Vector3<int> Vector3i;
	typedef class ORUtils::Vector3<uint> Vector3ui;
	typedef class ORUtils::Vector3<uchar> Vector3u;
	typedef class ORUtils::Vector3<float> Vector3f;

	typedef class ORUtils::Vector4<float> Vector4f;
	typedef class ORUtils::Vector4<int> Vector4i;
	typedef class ORUtils::Vector4<short> Vector4s;
	typedef class ORUtils::Vector4<uchar> Vector4u;

	//------------------------------------------------------
	// 
	// image defines
	//
	//------------------------------------------------------

	typedef  ORUtils::Image<Vector4f> Float4Image;
	typedef  ORUtils::Image<int> IntImage;
    typedef  ORUtils::Image<Vector4u> UChar4Image;
    
    typedef  ORUtils::Image4D<int> IntImage4D;
    typedef  ORUtils::Image4D<neur> NeuronImage;


	//------------------------------------------------------
	// 
	// Other defines
	//
	//------------------------------------------------------

	typedef enum 
	{
		CIELAB = 0,
		XYZ,
		RGB
	} COLOR_SPACE;

	typedef enum
	{
		GIVEN_NUM = 0,
		GIVEN_SIZE

	} SEG_METHOD;


}


#ifndef DEBUGBREAK
#define DEBUGBREAK \
	{ \
	int ryifrklaeybfcklarybckyar=0; \
	ryifrklaeybfcklarybckyar++; \
	}
#endif
