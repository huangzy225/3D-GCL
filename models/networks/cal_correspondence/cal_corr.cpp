#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

// forward declearation
std::vector<torch::Tensor> cal_corr_cuda(torch::Tensor vis_1, 
                                         torch::Tensor face,
                                         torch::Tensor verts_1,
                                         torch::Tensor verts_2,
                                         torch::Tensor visible_faces_2);

std::vector<torch::Tensor> vis_2_onehot_cuda(torch::Tensor vis_2,
                                             const int face_num);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> cal_corr(torch::Tensor vis_1, 
                                    torch::Tensor face,
                                    torch::Tensor verts_1,
                                    torch::Tensor verts_2,
                                    torch::Tensor visible_faces_2)
{
    CHECK_INPUT(vis_1);
    CHECK_INPUT(face);
    CHECK_INPUT(verts_1);
    CHECK_INPUT(verts_2);
    CHECK_INPUT(visible_faces_2);

    return cal_corr_cuda(
        vis_1,
        face,
        verts_1,
        verts_2,
        visible_faces_2);
}

std::vector<torch::Tensor> vis_2_onehot(torch::Tensor vis_2,
                                       const int face_num)
{
    CHECK_INPUT(vis_2);
    
    return vis_2_onehot_cuda(vis_2, face_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cal_corr", &cal_corr, "calculate corralation map and mask. (CUDA)");
  m.def("vis_2_onehot", &vis_2_onehot, "pad the visible map to a same size. (CUDA)");
}