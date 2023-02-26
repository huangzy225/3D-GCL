#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

typedef long long ll;

// Kernel functions
template <typename scalar_t>
__global__ void vis_2_onehot_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> visible_faces_2, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> vis_2,
    const int image_height,
    const int image_width)
{
    int batch_id = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float face_id = vis_2[batch_id][x][y];
    int face_id_int = int(face_id);
    if (face_id_int != -1){
        visible_faces_2[batch_id][face_id_int] = 1.;
    }
}


std::vector<torch::Tensor> vis_2_onehot_cuda(torch::Tensor vis_2,
                                             const int face_num){
    const int warp_size = 32;
    const dim3 blocks(warp_size, warp_size, 1);     // determine size of grids, blocks, threads

    const auto batch_size = vis_2.size(0);
    const auto image_height = vis_2.size(1);
    const auto image_width = vis_2.size(2);

    const auto h_grid = (image_height + warp_size - 1) / warp_size;
    const auto w_grid = (image_width + warp_size - 1) / warp_size;
    const dim3 grids(h_grid ,w_grid, batch_size);             // determine size of grids, blocks, threads

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, vis_2.device().index());
    auto vis_face_2_input = torch::zeros({batch_size, face_num}, options);

    AT_DISPATCH_FLOATING_TYPES(vis_2.type(), "vis_2_onehot_cuda", ([&] {
        vis_2_onehot_cuda_kernel<scalar_t><<<grids, blocks>>>(
            vis_face_2_input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            vis_2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            image_height,
            image_width);
    }));
    
    return {vis_face_2_input};
}


// face-wise calculation
template <typename scalar_t>
__global__ void cal_corr_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> vis_1, 
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> face, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> verts_1, 
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> verts_2,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> visible_faces_2, 
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> corr_map,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> corr_mask, 
    const int image_height,
    const int image_width)
{
    const int batch_id = blockIdx.z;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(y >= image_width || x >= image_height)  
        return;    

    //pts_1: 1x2
    int pts_1_x = x;            // [0, 191]; col
    int pts_1_y = y;            // [0, 255]; row

    int face_id = int(vis_1[batch_id][x][y]);   
    
    if(face_id == -1){
        corr_mask[batch_id][x][y] = 2;
    }

    if (face_id != -1){
        const int vert_ia = int(face[face_id][0]);   
        const int vert_ib = int(face[face_id][1]);    
        const int vert_ic = int(face[face_id][2]); 
        // tri_1/2: 3x2      
        const auto tri_1a0 = verts_1[batch_id][vert_ia][0];
        const auto tri_1a1 = verts_1[batch_id][vert_ia][1];

        const auto tri_1b0 = verts_1[batch_id][vert_ib][0];
        const auto tri_1b1 = verts_1[batch_id][vert_ib][1];

        const auto tri_1c0 = verts_1[batch_id][vert_ic][0];
        const auto tri_1c1 = verts_1[batch_id][vert_ic][1];


        const auto tri_2a0 = verts_2[batch_id][vert_ia][0];
        const auto tri_2a1 = verts_2[batch_id][vert_ia][1];

        const auto tri_2b0 = verts_2[batch_id][vert_ib][0];
        const auto tri_2b1 = verts_2[batch_id][vert_ib][1];

        const auto tri_2c0 = verts_2[batch_id][vert_ic][0];
        const auto tri_2c1 = verts_2[batch_id][vert_ic][1];


        // get barycentric_coords
        const auto d00 = (tri_1b0 - tri_1a0) * (tri_1b0 - tri_1a0) +  
                         (tri_1b1 - tri_1a1) * (tri_1b1 - tri_1a1);
        const auto d01 = (tri_1b0 - tri_1a0) * (tri_1c0 - tri_1a0) +
                         (tri_1b1 - tri_1a1) * (tri_1c1 - tri_1a1);
        const auto d11 = (tri_1c0 - tri_1a0) * (tri_1c0 - tri_1a0) + 
                         (tri_1c1 - tri_1a1) * (tri_1c1 - tri_1a1);
        const auto d20 = (pts_1_y - tri_1a0) * (tri_1b0 - tri_1a0) + 
                         (pts_1_x - tri_1a1) * (tri_1b1 - tri_1a1);
        const auto d21 = (pts_1_y - tri_1a0) * (tri_1c0 - tri_1a0) + 
                         (pts_1_x - tri_1a1) * (tri_1c1 - tri_1a1);

        const auto denom = d00*d11 - d01*d01;
        const auto v = (d11*d20 - d01*d21) / denom;
        const auto w = (d00*d21 - d01*d20) / denom;
        const auto u = 1. - v - w;

        const auto pts_2_x = tri_2a0 * u + tri_2b0 * v + tri_2c0 * w;
        const auto pts_2_y = tri_2a1 * u + tri_2b1 * v + tri_2c1 * w;

        corr_map[batch_id][x][y][0] = pts_2_x;
        corr_map[batch_id][x][y][1] = pts_2_y;

        if(abs(visible_faces_2[batch_id][face_id]-1) < 0.001){
            corr_mask[batch_id][x][y] = 0.;
        }
    }
}


std::vector<torch::Tensor> cal_corr_cuda(torch::Tensor vis_1, 
                                         torch::Tensor face,
                                         torch::Tensor verts_1,
                                         torch::Tensor verts_2,
                                         torch::Tensor visible_faces_2){
    const int warp_size = 32;
    const dim3 blocks(warp_size, warp_size, 1);     // determine size of grids, blocks, threads

    const auto batch_size = vis_1.size(0);
    const auto image_height = vis_1.size(1);
    const auto image_width = vis_1.size(2);

    const auto h_grid = (image_height + warp_size - 1) / warp_size;
    const auto w_grid = (image_width + warp_size - 1) / warp_size;
    const dim3 grids(h_grid, w_grid, batch_size);             // determine size of grids, blocks, threads

    auto option_map = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, vis_1.device().index());
    auto corr_map = torch::zeros({batch_size, image_height, image_width, 2}, option_map);

    auto option_mask = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, vis_1.device().index());
    auto corr_mask = torch::ones({batch_size, image_height, image_width}, option_mask);
    
    AT_DISPATCH_FLOATING_TYPES(vis_1.type(), "cal_corr_cuda", ([&] {
        cal_corr_cuda_kernel<scalar_t><<<grids, blocks>>>(
            vis_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            face.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            verts_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            verts_2.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            visible_faces_2.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            corr_map.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            corr_mask.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            image_height,
            image_width);
    }));
    
    return {corr_map, corr_mask};
}
