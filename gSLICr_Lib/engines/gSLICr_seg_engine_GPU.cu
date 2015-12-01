// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#include "gSLICr_seg_engine_GPU.h"
#include "gSLICr_seg_engine_shared.h"

using namespace std;
using namespace gSLICr;
using namespace gSLICr::objects;
using namespace gSLICr::engines;

// ----------------------------------------------------
//
//	kernel function defines
//
// ----------------------------------------------------

__global__ void Enforce_Connectivity_device(
    const int* in_idx_img, int* out_idx_img, Vector4i img_size);

__global__ void Init_Cluster_Centers_device(
    const neur* inimg, spixel_info* out_spixel,
    Vector4i map_size, Vector4i img_size, int spixel_size);

__global__ void Find_Center_Association_device(
    const neur* in_img_red, const neur* in_img_green, const spixel_info* in_spixel_map,
    int* out_idx_img, Vector4i map_size, Vector4i img_size, int spixel_size,
    float weight, float max_xyz_dist, float max_color_dist);

__global__ void Update_Cluster_Center_device(
    const neur* inimg, const int* in_idx_img,
    spixel_info* accum_map, Vector4i map_size, Vector4i img_size,
    int spixel_size, int no_blocks_per_line);

__global__ void Finalize_Reduction_Result_device(
    const spixel_info* accum_map, spixel_info* spixel_list,
    Vector4i map_size, int no_blocks_per_spixel);

__global__ void Draw_Segmentation_Result_device(const int* idx_img, Vector4u* sourceimg, Vector4u* outimg, Vector2i img_size);

// ----------------------------------------------------
//
//	host function implementations
//
// ----------------------------------------------------

seg_engine_GPU::seg_engine_GPU(const settings& in_settings) : seg_engine(in_settings)
{
    src_img_red = new NeuronImage(in_settings.img_size_red, true, true);
    src_img_green = new NeuronImage(in_settings.img_size_green, true, true);

    idx_img = new IntImage4D(in_settings.img_size_red, true, true);
    tmp_idx_img = new IntImage4D(in_settings.img_size_red, true, true);

	if (in_settings.seg_method == GIVEN_NUM)
	{
		float cluster_size = (float)(src_img_red->dataSize) / (float)(in_settings.no_segs);
		spixel_size = (int)ceil(pow(cluster_size, 1.0 / 3.0));
	}
	else
	{
		spixel_size = in_settings.spixel_size;
	}
	
	int spixel_per_col = (int)ceil((float)in_settings.img_size_red.x / spixel_size);
	int spixel_per_row = (int)ceil((float)in_settings.img_size_red.y / spixel_size);
    int spixel_per_dep = (int)ceil((float)in_settings.img_size_red.z / spixel_size);
	
    Vector4i map_size = Vector4i(spixel_per_col, spixel_per_row, spixel_per_dep, 1);
	spixel_map = new SpixelMap(map_size, true, true);

    // cluster center search space is in 3S x 3S x 3S neighbors
    // because each pixel may belong to a maximum of 27 clusters
    // (on the opposite side but equivalently, the original algorithm
    // lets each cluster center assign membership to pixel within a 2S x 2S region)
    float total_pixel_to_search = (float)(spixel_size * spixel_size * spixel_size * 27);

    // If each thread block is BLOCK_DIM x BLOCK_DIM x BLOCK_DIM, then
    // the number of blocks to search over the above space is:
    no_grid_per_center = (int)ceil(total_pixel_to_search / 
        (float)(BLOCK_DIM * BLOCK_DIM * BLOCK_DIM));

	map_size.x *= no_grid_per_center;
    accum_map = new SpixelMap(map_size, true, true);

	// normalizing factors
    // maximum xyz distance between any two points within a cluster
    // is 3 * (S^2) where S = size of super pixel
    max_xyz_dist = 1.0f / (sqrtf(3) * spixel_size);
    max_color_dist = 1.0f; // TODO: adjust color normalization to [0,1]
}

gSLICr::engines::seg_engine_GPU::~seg_engine_GPU()
{
    delete accum_map;
}

void gSLICr::engines::seg_engine_GPU::Init_Cluster_Centers()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
	neur* img_ptr = src_img_red->GetData(MEMORYDEVICE_CUDA);

	Vector4i map_size = spixel_map->noDims;
    Vector4i img_size = src_img_red->noDims;

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(
        (int)ceil((float)map_size.x / (float)blockSize.x),
        (int)ceil((float)map_size.y / (float)blockSize.y),
        (int)ceil((float)map_size.z / (float)blockSize.z));

	Init_Cluster_Centers_device <<< gridSize, blockSize >>>(
        img_ptr, spixel_list, map_size, img_size, spixel_size);
}

void gSLICr::engines::seg_engine_GPU::Find_Center_Association()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
    neur* red_ptr = src_img_red->GetData(MEMORYDEVICE_CUDA);
    neur* green_ptr = src_img_green->GetData(MEMORYDEVICE_CUDA);
    int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

    Vector4i map_size = spixel_map->noDims;

    // use full 4-d size from green image to compute distance temporally
    Vector4i img_size = src_img_green->noDims;

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(
        (int)ceil((float)img_size.x / (float)blockSize.x),
        (int)ceil((float)img_size.y / (float)blockSize.y),
        (int)ceil((float)img_size.z / (float)blockSize.z));

	Find_Center_Association_device<<< gridSize, blockSize >>>(
        red_ptr, green_ptr, spixel_list, idx_ptr, map_size, img_size, spixel_size,
        gSLICr_settings.coh_weight, max_xyz_dist, max_color_dist);
}

void gSLICr::engines::seg_engine_GPU::Update_Cluster_Center()
{
	spixel_info* accum_map_ptr = accum_map->GetData(MEMORYDEVICE_CUDA);
	spixel_info* spixel_list_ptr = spixel_map->GetData(MEMORYDEVICE_CUDA);
    neur* img_ptr = src_img_red->GetData(MEMORYDEVICE_CUDA);
    int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

	Vector4i map_size = spixel_map->noDims;
    Vector4i img_size = src_img_green->noDims;

	int no_blocks_per_line = spixel_size * 3 / BLOCK_DIM;

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(map_size.x, map_size.y, map_size.z * no_grid_per_center);

	Update_Cluster_Center_device<<< gridSize, blockSize >>>(
        img_ptr, idx_ptr, accum_map_ptr, map_size, img_size,
        spixel_size, no_blocks_per_line);

	dim3 gridSize3(map_size.x, map_size.y, map_size.z);

    Finalize_Reduction_Result_device<<< gridSize3, blockSize >>>(
        accum_map_ptr, spixel_list_ptr, map_size, no_grid_per_center);
}

void gSLICr::engines::seg_engine_GPU::Enforce_Connectivity()
{
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);
	int* tmp_idx_ptr = tmp_idx_img->GetData(MEMORYDEVICE_CUDA);
	Vector4i img_size = idx_img->noDims;

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(
        (int)ceil((float)img_size.x / (float)blockSize.x),
        (int)ceil((float)img_size.y / (float)blockSize.y),
        (int)ceil((float)img_size.z / (float)blockSize.z));

	Enforce_Connectivity_device<<< gridSize, blockSize >>>(idx_ptr, tmp_idx_ptr, img_size);
	Enforce_Connectivity_device<<< gridSize, blockSize >>>(tmp_idx_ptr, idx_ptr, img_size);
}

void gSLICr::engines::seg_engine_GPU::Draw_Segmentation_Result(UChar4Image* out_img)
{
//	Vector4u* inimg_ptr = source_img->GetData(MEMORYDEVICE_CUDA);
//	Vector4u* outimg_ptr = out_img->GetData(MEMORYDEVICE_CUDA);
//	int* idx_img_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);
//	
//	Vector2i img_size = idx_img->noDims;
//
//	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
//	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));
//
//	Draw_Segmentation_Result_device<<<gridSize,blockSize>>>(idx_img_ptr, inimg_ptr, outimg_ptr, img_size);
//	out_img->UpdateHostFromDevice();
}



// ----------------------------------------------------
//
//	device function implementations
//
// ----------------------------------------------------

__global__ void Draw_Segmentation_Result_device(const int* idx_img, Vector4u* sourceimg, Vector4u* outimg, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x > img_size.x - 2 || y > img_size.y - 2) return;

	draw_superpixel_boundry_shared(idx_img, sourceimg, outimg, img_size, x, y);
}

__global__ void Init_Cluster_Centers_device(
    const neur* inimg, spixel_info* out_spixel,
    Vector4i map_size, Vector4i img_size, int spixel_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x > map_size.x - 1 || y > map_size.y - 1 || z > map_size.z - 1)
        return;

    init_cluster_centers_shared(inimg, out_spixel, map_size, img_size, spixel_size, x, y, z);
}

__global__ void Find_Center_Association_device(
    const neur* in_img_red, const neur* in_img_green, const spixel_info* in_spixel_map,
    int* out_idx_img, Vector4i map_size, Vector4i img_size, int spixel_size, float weight,
    float max_xyz_dist, float max_color_dist)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x > img_size.x - 1 || y > img_size.y - 1 || z > img_size.z - 1)
        return;

    find_center_association_shared(
        in_img_red, in_img_green, in_spixel_map, out_idx_img, map_size,
        img_size, spixel_size, weight, x, y, z, max_xyz_dist, max_color_dist);
}

__global__ void Update_Cluster_Center_device(
    const neur* inimg, const int* in_idx_img, spixel_info* accum_map,
    Vector4i map_size, Vector4i img_size, int spixel_size, int no_blocks_per_line)
{
	int local_id = threadIdx.z * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x +
                   threadIdx.x;

    // shared memory to store the aggregate values of all pixels
    // that belong to this cluster
    __shared__ neur color_shared[BLOCK_DIM * BLOCK_DIM * BLOCK_DIM];
    __shared__ Vector3f xyz_shared[BLOCK_DIM * BLOCK_DIM * BLOCK_DIM];
    __shared__ int count_shared[BLOCK_DIM * BLOCK_DIM * BLOCK_DIM];
	__shared__ bool should_add; 

	color_shared[local_id] = 0;
    xyz_shared[local_id] = Vector3f(0, 0, 0);
	count_shared[local_id] = 0;
	should_add = false;
	__syncthreads();

	int no_blocks_per_spixel = gridDim.z / map_size.z;

    // The z dimension of the grid is itself a [map.z x no_blocks_per_spixel] grid
    // so we need to extract the corresponding components separately
    int blockIdx_z = blockIdx.z / no_blocks_per_spixel;
    int blockIdx_grid = blockIdx.z % no_blocks_per_spixel;

    int spixel_id = blockIdx_z * map_size.y * map_size.x +
                    blockIdx.y * map_size.x +
                    blockIdx.x;

    //if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx_z == 0)
    //{
    //    printf("%d ", local_id);
    //}

    //if (local_id == 0)
    //{
    //    printf("blockDim: x = %d, y = %d, z = %d\n", blockDim.x, blockDim.y, blockDim.z);
    //    printf("gridDim:  x = %d, y = %d, z = %d\n", gridDim.x, gridDim.y, gridDim.z);
    //    printf("no_blocks_per_spixel = %d\n", no_blocks_per_spixel);
    //}
	// compute the relative position in the search window
    int block_x = blockIdx_grid % no_blocks_per_line;
    int block_y = (blockIdx_grid / no_blocks_per_line) % no_blocks_per_line;
    int block_z = blockIdx_grid / (no_blocks_per_line * no_blocks_per_line);

    //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    //{
    //    printf("blockIdx.z = %d,blockIdx.y = %d,blockIdx.x = %d\n",
    //        blockIdx.z, blockIdx.y, blockIdx.x);
    //    //printf("block_x = %d,block_y = %d,block_z = %d\n", block_x, block_y, block_z);
    //}

    //if (block_z == 0)
    //{
    //    printf("hello");
    //}

	int x_offset = block_x * BLOCK_DIM + threadIdx.x;
    int y_offset = block_y * BLOCK_DIM + threadIdx.y;
    int z_offset = block_z * BLOCK_DIM + threadIdx.z;

    if (x_offset < spixel_size * 3 &&
        y_offset < spixel_size * 3 &&
        z_offset < spixel_size * 3)
	{
		// compute the start of the search window
		int x_start = blockIdx.x * spixel_size - spixel_size;	
        int y_start = blockIdx.y * spixel_size - spixel_size;
        int z_start = blockIdx_z * spixel_size - spixel_size;

		int x_img = x_start + x_offset;
        int y_img = y_start + y_offset;
        int z_img = z_start + z_offset;

		if (x_img >= 0 && x_img < img_size.x &&
            y_img >= 0 && y_img < img_size.y &&
            z_img >= 0 && z_img < img_size.z)
		{
            int img_idx = z_img * img_size.y * img_size.x + y_img * img_size.x + x_img;
            // if this pixel belongs to the current cluster then add it to the list
			if (in_idx_img[img_idx] == spixel_id)
			{
                //if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx_z == 0)
                //{
                //    printf("spid %d, lid %d, img_idx %d, x %d, y %d, z %d, color %f\n",
                //        spixel_id, local_id, img_idx, x_img, y_img, z_img, inimg[img_idx]);
                //}
				color_shared[local_id] = inimg[img_idx];
                xyz_shared[local_id] = Vector3f(x_img, y_img, z_img);
				count_shared[local_id] = 1;
				should_add = true;
			}
		}
	}
	__syncthreads();

	if (should_add)
	{
        //if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx_z == 0 &&
        //    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        //{
        //    printf("------------------------\n");
        //    for (size_t i = 0; i < 512; i++)
        //    {
        //        printf("(x = %f, y = %f, z = %f) ",
        //            xyz_shared[i].x, xyz_shared[i].y, xyz_shared[i].z);
        //    }
        //    printf("\n");
        //}
        __syncthreads();

        // parallel reduction in local memory using unrolled loops
        if (local_id < 256)
        {
            color_shared[local_id] += color_shared[local_id + 256];
            xyz_shared[local_id] += xyz_shared[local_id + 256];
            count_shared[local_id] += count_shared[local_id + 256];
        }
        __syncthreads();

		if (local_id < 128)
		{
			color_shared[local_id] += color_shared[local_id + 128];
            xyz_shared[local_id] += xyz_shared[local_id + 128];
			count_shared[local_id] += count_shared[local_id + 128];
		}
		__syncthreads();

		if (local_id < 64)
		{
			color_shared[local_id] += color_shared[local_id + 64];
            xyz_shared[local_id] += xyz_shared[local_id + 64];
			count_shared[local_id] += count_shared[local_id + 64];
		}
		__syncthreads();

		if (local_id < 32)
		{
			color_shared[local_id] += color_shared[local_id + 32];
			color_shared[local_id] += color_shared[local_id + 16];
			color_shared[local_id] += color_shared[local_id + 8];
			color_shared[local_id] += color_shared[local_id + 4];
			color_shared[local_id] += color_shared[local_id + 2];
			color_shared[local_id] += color_shared[local_id + 1];

			xyz_shared[local_id] += xyz_shared[local_id + 32];
			xyz_shared[local_id] += xyz_shared[local_id + 16];
			xyz_shared[local_id] += xyz_shared[local_id + 8];
			xyz_shared[local_id] += xyz_shared[local_id + 4];
			xyz_shared[local_id] += xyz_shared[local_id + 2];
			xyz_shared[local_id] += xyz_shared[local_id + 1];

			count_shared[local_id] += count_shared[local_id + 32];
			count_shared[local_id] += count_shared[local_id + 16];
			count_shared[local_id] += count_shared[local_id + 8];
			count_shared[local_id] += count_shared[local_id + 4];
			count_shared[local_id] += count_shared[local_id + 2];
			count_shared[local_id] += count_shared[local_id + 1];
		}
	}
	__syncthreads();

	if (local_id == 0)
	{
        int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx_grid;
        accum_map[accum_map_idx].center = xyz_shared[0];
        accum_map[accum_map_idx].red_color = color_shared[0];
        accum_map[accum_map_idx].no_pixels = count_shared[0];

        //if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx_z == 0 &&
        //    threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        //{
        //    printf("accum_idx %f, x %f, y %f, z %f, color %f, count %d\n",
        //        accum_map_idx, 
        //        xyz_shared[0].x, xyz_shared[0].y, xyz_shared[0].z,
        //        color_shared[0],
        //        count_shared[0]);
        //}
	}


}

__global__ void Finalize_Reduction_Result_device(
    const spixel_info* accum_map, spixel_info* spixel_list,
    Vector4i map_size, int no_blocks_per_spixel)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x > map_size.x - 1 || y > map_size.y - 1 || z > map_size.z - 1)
        return;

	finalize_reduction_result_shared(accum_map, spixel_list, map_size,
        no_blocks_per_spixel, x, y, z);
}

__global__ void Enforce_Connectivity_device(const int* in_idx_img, int* out_idx_img,
    Vector4i img_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x > img_size.x - 1 || y > img_size.y - 1 || z > img_size.z - 1)
        return;

	supress_local_lable(in_idx_img, out_idx_img, img_size, x, y, z);
}

int seg_engine_GPU::Test_Update_Clusters()
{
    ORcudaSafeCall(cudaPeekAtLastError());
    ORcudaSafeCall(cudaDeviceSynchronize());

    accum_map->UpdateHostFromDevice();

    Vector4i accum_img_size = accum_map->noDims;
    int max_display = 40;
    const spixel_info* data = accum_map->GetData(MEMORYDEVICE_CPU);

    size_t total_pixels = 0;
    for (size_t i = 0; i < accum_map->dataSize; i++)
    {
        total_pixels += data[i].no_pixels;
        if (data[i].center.x > 0)
        {
            int xx = 0;
            xx++;
        }
    }

    printf("\n");
    for (size_t j = 0; j < min(max_display, accum_img_size.y); j++)
    {
        printf("--------- j = %d ---------\n");
        for (size_t i = 0; i < min(max_display, accum_img_size.x); i++)
        {
            const spixel_info& d = data[j * accum_img_size.x + i];
            printf("i = %d, id = %d, color = %f, x = %f, y = %f, z = %f, no_pix = %d\n",
                i, d.id, d.red_color, d.center.x, d.center.y, d.center.z, d.no_pixels);
        }
        printf("\n");
    }

    printf("\n");

    return 0;
}

int seg_engine_GPU::Test_Enforce_Connectivity()
{
    ORcudaSafeCall(cudaPeekAtLastError());
    ORcudaSafeCall(cudaDeviceSynchronize());

    idx_img->UpdateHostFromDevice();

    Vector4i idx_img_size = idx_img->noDims;
    int* data = idx_img->GetData(MEMORYDEVICE_CPU);

    int* clusters = (int*)calloc(2000, sizeof(int));
    for (size_t i = 0; i < idx_img->dataSize; i++)
    {
        clusters[data[i]]++;
    }

    size_t total_pixels = 0;
    for (size_t i = 0; i < 2000; i++)
    {
        total_pixels += clusters[i];
    }

    free(clusters);
    return 0;
}