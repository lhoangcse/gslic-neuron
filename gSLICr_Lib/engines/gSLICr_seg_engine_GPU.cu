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
    const neur* in_img_red, const neur* in_img_green, spixel_info* out_spixel,
    Vector4i map_size, Vector4i img_size, int spixel_size);

__global__ void Find_Center_Association_device(
    const neur* in_img_red, const neur* in_img_green, const spixel_info* in_spixel_map,
    int* out_idx_img, Vector4i map_size, Vector4i img_size, int spixel_size,
    float weight, float max_xyz_dist, float max_color_dist);

__global__ void Update_Cluster_Center_device(
    const neur* in_img_red, const neur* in_img_green, const int* in_idx_img,
    spixel_info* accum_map, Vector4i map_size, Vector4i img_size,
    int spixel_size, int no_blocks_per_line);

__global__ void Finalize_Reduction_Result_device(
    const spixel_info* accum_map, spixel_info* spixel_list,
    Vector4i map_size, int no_blocks_per_spixel);

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
    max_color_dist = 550.f; // empirical max difference in red channel intensities
}

gSLICr::engines::seg_engine_GPU::~seg_engine_GPU()
{
    delete accum_map;
}

void gSLICr::engines::seg_engine_GPU::Init_Cluster_Centers()
{
    spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
    neur* red_ptr = src_img_red->GetData(MEMORYDEVICE_CUDA);
    neur* green_ptr = src_img_green->GetData(MEMORYDEVICE_CUDA);

    Vector4i map_size = spixel_map->noDims;
    Vector4i img_size = src_img_green->noDims;

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize(
        (int)ceil((float)map_size.x / (float)blockSize.x),
        (int)ceil((float)map_size.y / (float)blockSize.y),
        (int)ceil((float)map_size.z / (float)blockSize.z));

    Init_Cluster_Centers_device <<< gridSize, blockSize >>>(
        red_ptr, green_ptr, spixel_list, map_size, img_size, spixel_size);
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
    neur* red_ptr = src_img_red->GetData(MEMORYDEVICE_CUDA);
    neur* green_ptr = src_img_green->GetData(MEMORYDEVICE_CUDA);
    int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

    Vector4i map_size = spixel_map->noDims;
    Vector4i img_size = src_img_green->noDims;

    int no_blocks_per_line = ceil(spixel_size * 3.0f / BLOCK_DIM);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize(map_size.x, map_size.y, map_size.z * no_grid_per_center);

    Update_Cluster_Center_device<<< gridSize, blockSize >>>(
        red_ptr, green_ptr, idx_ptr, accum_map_ptr, map_size, img_size,
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


// ----------------------------------------------------
//
//	device function implementations
//
// ----------------------------------------------------

__global__ void Init_Cluster_Centers_device(
    const neur* in_img_red, const neur* in_img_green, spixel_info* out_spixel,
    Vector4i map_size, Vector4i img_size, int spixel_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    if (x > map_size.x - 1 || y > map_size.y - 1 || z > map_size.z - 1)
        return;

    init_cluster_centers_shared(in_img_red, in_img_green, out_spixel,
        map_size, img_size, spixel_size, x, y, z);
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
    const neur* in_img_red, const neur* in_img_green, const int* in_idx_img,
    spixel_info* accum_map, Vector4i map_size, Vector4i img_size, int spixel_size,
    int no_blocks_per_line)
{
    int local_id = threadIdx.z * blockDim.x * blockDim.y +
                   threadIdx.y * blockDim.x +
                   threadIdx.x;
    
    // each block of threads is a cubic with size BLOCK_DIM
    const int block_size = BLOCK_DIM * BLOCK_DIM * BLOCK_DIM;
    // size of reduction block
    int reduce_block_size = 0;
    // each thread processes 16 time values at a time
    const int t_size = 16;
    // size of each 3D image
    const int time_step = img_size.x * img_size.y * img_size.z;
    // time index
    size_t t = 0;
    size_t tt = 0;
    size_t t_offset = 0;

    // shared memory to store the aggregate values of all pixels
    // that belong to this cluster
    __shared__ neur color_shared[block_size];
    __shared__ neur green_shared[t_size * block_size];
    __shared__ Vector3f xyz_shared[block_size];
    __shared__ int count_shared[block_size];
    __shared__ int pixel_idx[block_size];
    __shared__ bool should_add; 

    color_shared[local_id] = 0;
    xyz_shared[local_id] = Vector3f(0, 0, 0);
    count_shared[local_id] = 0;
    pixel_idx[local_id] = -1;

    for (tt = 0; tt < t_size; tt++)
    {
        green_shared[local_id * t_size + tt] = 0;
    }

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
    int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx_grid;

    // compute the relative position in the search window
    int block_x = blockIdx_grid % no_blocks_per_line;
    int block_y = (blockIdx_grid / no_blocks_per_line) % no_blocks_per_line;
    int block_z = blockIdx_grid / (no_blocks_per_line * no_blocks_per_line);

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
                color_shared[local_id] = in_img_red[img_idx];
                xyz_shared[local_id] = Vector3f(x_img, y_img, z_img);
                count_shared[local_id] = 1;
                pixel_idx[local_id] = img_idx;

                should_add = true;
            }
        }
    }
    __syncthreads();

    if (should_add)
    {
        // load and add up green channels
        int img_idx = pixel_idx[local_id];
        for (t = 0; t < IMAGE_TIME; t += t_size)
        {
            // Each thread loads t_size number of pixels at a time
            // It's important here that all threads must execute the same
            // code block leading up to __syncthreads
            for (tt = 0; tt < t_size; tt++)
            {
                if (t + tt >= IMAGE_TIME)
                {
                    // even if finished processing image, set value to zero
                    // so sum-reduction step is trivial
                    green_shared[local_id + tt * block_size] = 0;
                }
                else
                {
                    green_shared[local_id + tt * block_size] = (img_idx >= 0) ?
                        in_img_green[img_idx + (t + tt) * time_step] : 0;
                }
            }
            __syncthreads();
            // sum-reduce across threads
            // IMPORTANT: make sure block_size is power of 2 !!!
            reduce_block_size = block_size / 2;
            while (reduce_block_size > 0)
            {
                // each thread adds the following pair of pixels:
                // (i                 , i + reduce_block_size                 )
                // (i +     block_size, i + reduce_block_size +     block_size)
                // (i + 2 * block_size, i + reduce_block_size + 2 * block_size)
                // . . . . . . . . . . . . . . . .
                if (local_id < reduce_block_size)
                {
                    for (tt = 0; tt < t_size; tt++)
                    {
                        t_offset = local_id + tt * block_size;
                        green_shared[t_offset] += green_shared[t_offset + reduce_block_size];
                    }
                }
                reduce_block_size = reduce_block_size / 2;

                __syncthreads();
            }

            // output final sum for t_size pixels at a time
            if (local_id < t_size)
            {
                if (t + local_id >= IMAGE_TIME) break;

                // TODO: bank conflict in reading from green_shared
                // as t_size threads are accessing the same bank. However
                // global memory writes to accum_map are coalesced.
                accum_map[accum_map_idx].green_color[t + local_id] =
                    green_shared[local_id * block_size];
            }
            __syncthreads();
        }

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
        accum_map[accum_map_idx].center = xyz_shared[0];
        accum_map[accum_map_idx].red_color = color_shared[0];
        accum_map[accum_map_idx].no_pixels = count_shared[0];
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