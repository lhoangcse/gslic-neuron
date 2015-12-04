// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "../gSLICr_defines.h"
#include "../objects/gSLICr_spixel_info.h"

_CPU_AND_GPU_CODE_ inline void rgb2xyz(const gSLICr::Vector4u& pix_in, gSLICr::Vector4f& pix_out)
{
	float _b = (float)pix_in.x * 0.0039216f;
	float _g = (float)pix_in.y * 0.0039216f;
	float _r = (float)pix_in.z * 0.0039216f;

	pix_out.x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
	pix_out.y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
	pix_out.z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

}

_CPU_AND_GPU_CODE_ inline void rgb2CIELab(const gSLICr::Vector4u& pix_in, gSLICr::Vector4f& pix_out)
{
	float _b = (float)pix_in.x * 0.0039216f;
	float _g = (float)pix_in.y * 0.0039216f;
	float _r = (float)pix_in.z * 0.0039216f;

	float x = _r*0.412453f + _g*0.357580f + _b*0.180423f;
	float y = _r*0.212671f + _g*0.715160f + _b*0.072169f;
	float z = _r*0.019334f + _g*0.119193f + _b*0.950227f;

	float epsilon = 0.008856f;	//actual CIE standard
	float kappa = 903.3f;		//actual CIE standard

	float Xr = 0.950456f;	//reference white
	float Yr = 1.0f;		//reference white
	float Zr = 1.088754f;	//reference white

	float xr = x / Xr;
	float yr = y / Yr;
	float zr = z / Zr;

	float fx, fy, fz;
	if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
	else				fx = (kappa*xr + 16.0f) / 116.0f;
	if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
	else				fy = (kappa*yr + 16.0f) / 116.0f;
	if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
	else				fz = (kappa*zr + 16.0f) / 116.0f;

	pix_out.x = 116.0f*fy - 16.0f;
	pix_out.y = 500.0f*(fx - fy);
	pix_out.z = 200.0f*(fy - fz);
}

_CPU_AND_GPU_CODE_ inline void cvt_img_space_shared(const gSLICr::Vector4u* inimg, gSLICr::Vector4f* outimg, const gSLICr::Vector2i& img_size, int x, int y, const gSLICr::COLOR_SPACE& color_space)
{
	int idx = y * img_size.x + x;

	switch (color_space)
	{
	case gSLICr::RGB:
		outimg[idx].x = inimg[idx].x;
		outimg[idx].y = inimg[idx].y;
		outimg[idx].z = inimg[idx].z;
		break;
	case gSLICr::XYZ:
		rgb2xyz(inimg[idx], outimg[idx]);
		break;
	case gSLICr::CIELAB:
		rgb2CIELab(inimg[idx], outimg[idx]);
		break;
	}
}

_CPU_AND_GPU_CODE_ inline void init_cluster_centers_shared(
    const gSLICr::neur* in_img_red, const gSLICr::neur* in_img_green,
    gSLICr::objects::spixel_info* out_spixel,
    gSLICr::Vector4i map_size, gSLICr::Vector4i img_size, int spixel_size,
    int x, int y, int z)
{
	int cluster_idx = z * map_size.x * map_size.y + y * map_size.x + x;
    int time_step = img_size.x * img_size.y * img_size.z;

	int img_x = x * spixel_size + spixel_size / 2;
	int img_y = y * spixel_size + spixel_size / 2;
    int img_z = z * spixel_size + spixel_size / 2;

	img_x = img_x >= img_size.x ? (x * spixel_size + img_size.x) / 2 : img_x;
    img_y = img_y >= img_size.y ? (y * spixel_size + img_size.y) / 2 : img_y;
    img_z = img_z >= img_size.z ? (z * spixel_size + img_size.z) / 2 : img_z;
    int img_idx = img_z * img_size.y * img_size.x + img_y * img_size.x + img_x;

	// TODO: go one step towards gradients direction
    //printf("x=%d,y=%d,z=%d,ix=%d,iy=%d,iz=%d,c=%d\n", x, y, z, img_x, img_y, img_z, cluster_idx);

	out_spixel[cluster_idx].id = cluster_idx;
    out_spixel[cluster_idx].center = 
        gSLICr::Vector3f((float)img_x, (float)img_y, (float)img_z);
	//out_spixel[cluster_idx].color_info = inimg[img_idx];
    for (int i = 0; i < IMAGE_TIME; i++)
    {
        out_spixel[cluster_idx].green_color[i] = in_img_green[img_idx + i * time_step];
    }
    out_spixel[cluster_idx].red_color = in_img_red[img_idx];
	out_spixel[cluster_idx].no_pixels = 0;
}

_CPU_AND_GPU_CODE_ inline float compute_slic_distance(
    const gSLICr::neur* in_img_red, const gSLICr::neur* in_img_green,
    gSLICr::Vector4i img_size, int idx_img,
    int x, int y, int z, const gSLICr::objects::spixel_info& center_info,
    float weight, float normalizer_xy, float normalizer_color)
{
    int t = 0; // time slice index
    float dcolor = fabsf(in_img_red[idx_img] - center_info.red_color);

    float dxyz = sqrtf(
        (x - center_info.center.x) * (x - center_info.center.x) +
        (y - center_info.center.y) * (y - center_info.center.y) +
        (z - center_info.center.z) * (z - center_info.center.z));

    float dt = 0;
    size_t time_step = img_size.x * img_size.y * img_size.z;
    size_t ctr_img_idx = center_info.center.z * img_size.y * img_size.x +
                         center_info.center.y * img_size.x +
                         center_info.center.x;
    for (t = 0; t < img_size.w; t++)
    {
        dt += in_img_green[idx_img + t * time_step] * center_info.green_color[t];
    }
    dt = 1 - dt;

	float retval = dcolor * normalizer_color + weight * dxyz * normalizer_xy + dt;

    //if (idx_img == 0)
    //{
    //    printf("xyz %d %d %d, c-xyz %f %f %f, dcolor %f,dxyz %f,dt %f,retval %f, green ",
    //        x, y, z, center_info.center.x, center_info.center.y, center_info.center.z,
    //        dcolor, dxyz, dt, retval);
    //    for (t = 0; t < img_size.w; t++)
    //    {
    //        printf("%f ", center_info.green_color[t]);
    //    }
    //    printf("\n");
    //}

	return retval;
}

_CPU_AND_GPU_CODE_ inline void find_center_association_shared(
    const gSLICr::neur* in_img_red, const gSLICr::neur* in_img_green,
    const gSLICr::objects::spixel_info* in_spixel_map, int* out_idx_img,
    gSLICr::Vector4i map_size, gSLICr::Vector4i img_size, int spixel_size,
    float weight, int x, int y, int z, float max_xyz_dist, float max_color_dist)
{
	int idx_img = z * img_size.x * img_size.y + y * img_size.x + x;

	int ctr_x = x / spixel_size;
    int ctr_y = y / spixel_size;
    int ctr_z = z / spixel_size;

	int minidx = -1;
	float dist = 999999.9999f;

	// search 3x3x3 neighborhood
	for (int i = -1; i <= 1; i++)
    for (int j = -1; j <= 1; j++)
    for (int k = -1; k <= 1; k++)
	{
		int ctr_x_check = ctr_x + j;
        int ctr_y_check = ctr_y + i;
        int ctr_z_check = ctr_z + k;
        if (ctr_x_check >= 0 && ctr_x_check < map_size.x && 
            ctr_y_check >= 0 && ctr_y_check < map_size.y &&
            ctr_z_check >= 0 && ctr_z_check < map_size.z)
		{
            int ctr_idx = ctr_z_check * map_size.x * map_size.y + ctr_y_check * map_size.x + ctr_x_check;
			float cdist = compute_slic_distance(
                in_img_red, in_img_green, img_size, idx_img,
                x, y, z, in_spixel_map[ctr_idx], weight,
                max_xyz_dist, max_color_dist);
			if (cdist < dist)
			{
				dist = cdist;
				minidx = in_spixel_map[ctr_idx].id;
			}
		}
	}

	if (minidx >= 0) out_idx_img[idx_img] = minidx;
}

_CPU_AND_GPU_CODE_ inline void draw_superpixel_boundry_shared(const int* idx_img, gSLICr::Vector4u* sourceimg, gSLICr::Vector4u* outimg, gSLICr::Vector2i img_size, int x, int y)
{
	int idx = y * img_size.x + x;

	if (idx_img[idx] != idx_img[idx + 1]
	 || idx_img[idx] != idx_img[idx - 1]
	 || idx_img[idx] != idx_img[(y - 1)*img_size.x + x]
	 || idx_img[idx] != idx_img[(y + 1)*img_size.x + x])
	{
		outimg[idx] = gSLICr::Vector4u(0,0,255,0);
	}
	else
	{
		outimg[idx] = sourceimg[idx];
	}
}

_CPU_AND_GPU_CODE_ inline void finalize_reduction_result_shared(
    const gSLICr::objects::spixel_info* accum_map,
    gSLICr::objects::spixel_info* spixel_list,
    gSLICr::Vector4i map_size, int no_blocks_per_spixel, int x, int y, int z)
{
	int spixel_idx = z * map_size.y * map_size.x + y * map_size.x + x;
    size_t t = 0;
    int no_pixels = 0;

	spixel_list[spixel_idx].center = gSLICr::Vector3f(0, 0, 0);
    spixel_list[spixel_idx].red_color = 0;
	spixel_list[spixel_idx].no_pixels = 0;
    for (t = 0; t < IMAGE_TIME; t++)
    {
        spixel_list[spixel_idx].green_color[t] = 0;
    }

	for (int i = 0; i < no_blocks_per_spixel; i++)
	{
		int accum_list_idx = spixel_idx * no_blocks_per_spixel + i;

		spixel_list[spixel_idx].center += accum_map[accum_list_idx].center;
        spixel_list[spixel_idx].red_color += accum_map[accum_list_idx].red_color;
		spixel_list[spixel_idx].no_pixels += accum_map[accum_list_idx].no_pixels;
        for (t = 0; t < IMAGE_TIME; t++)
        {
            spixel_list[spixel_idx].green_color[t] +=
                accum_map[accum_list_idx].green_color[t];
        }
	}

	if (spixel_list[spixel_idx].no_pixels != 0)
	{
        no_pixels = spixel_list[spixel_idx].no_pixels;
        spixel_list[spixel_idx].center /= (float)no_pixels;
        spixel_list[spixel_idx].red_color /= (float)no_pixels;
        for (t = 0; t < IMAGE_TIME; t++)
        {
            spixel_list[spixel_idx].green_color[t] /= (float)no_pixels;
        }
    }
}

_CPU_AND_GPU_CODE_ inline void supress_local_lable(const int* in_idx_img,
    int* out_idx_img, gSLICr::Vector4i img_size, int x, int y, int z)
{
    int idx_img = z * img_size.y * img_size.x + y * img_size.x + x;
    int clabel = in_idx_img[idx_img];

	// don't suppress boundary
    if (x <= 1 || x >= img_size.x - 2 ||
        y <= 1 || y >= img_size.y - 2 ||
        z <= 1 || z >= img_size.z - 2)
	{ 
        out_idx_img[idx_img] = clabel;
		return; 
	}

	int diff_count = 0;
	int diff_label = -1;

    for (int k = -2; k <= 2; k++)
	for (int j = -2; j <= 2; j++)
    for (int i = -2; i <= 2; i++)
	{
        int idx = (z + k) * img_size.y * img_size.x + (y + j) * img_size.x + (x + i);
        int nlabel = in_idx_img[idx];
        if (nlabel != clabel)
		{
            diff_label = nlabel;
			diff_count++;
		}
	}

	if (diff_count >= 16)
        out_idx_img[idx_img] = diff_label;
	else
        out_idx_img[idx_img] = clabel;
}