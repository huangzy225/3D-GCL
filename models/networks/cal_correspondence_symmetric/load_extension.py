import torch
import pickle
import os
import time
import numpy as np
from torch.utils.cpp_extension import load


SMPL_Faces = np.load('./smpl/smpl_faces.npy')

data_root = '/media/homee/Data/Datasets/Zalora_256_192'
source = '24_2AC5DAAD353D9EGS_2026295_superdry-6070-5926202-1.jpg'
target = '24_1FD83AAEFCB574GS_2374720_superdry-9883-0274732-1.jpg'

def calc_corr_python_cuda(verts_A, vis_A, verts_B, vis_B, faces):
    '''
    Compute for each pixel (x,y) in img_1 the corresponding pixel (u,v) in img_2.
    Input:
        pred_1: HMR prediction for image 1. see calc_correspondence_from_smpl.
        pred_2: HMR prediction for image 2
        faces: list, each element is a triangle face represented by 3 vertex indices (i_a, i_b, i_c)
    Output:
        corr_map: (img_size, img_size, 2). corr_map[y,x] = (u,v)
        corr_mask: (img_size, img_size). The value is one of:
            0: human pixel with correspondence in img_1
            1: human pixel without correspondece in img_1
            2: background pixel
    '''
    faces = torch.from_numpy(faces.astype(np.int64)).to(vis_A.device)# .to(torch.long)
    n, h, w = vis_A.shape
    corr_maps = []
    corr_masks = []
    for verts_1, vis_1, verts_2, vis_2 in zip(verts_A, vis_A, verts_B, vis_B):
        verts_1, verts_2 = verts_1[:, :2], verts_2[:, :2]
        invisible = 4294967295
        # common visible face indices
        visible_face_1 = torch.unique(vis_1, sorted=True) # pred_1['visibility'])
        visible_face_2 = torch.unique(vis_2, sorted=True) # pred_2['visibility'])
        if visible_face_1[-1] == invisible:
            visible_face_1 = visible_face_1[:-1]
        # visible_face_1 = visible_face_1[visible_face_1 != invisible]
        common_face = np.intersect1d(visible_face_1.detach().cpu().numpy(), 
                                     visible_face_2.detach().cpu().numpy(), assume_unique=True)

        # corr_map and corr_mask    
        corr_map = torch.zeros(size=(h, w, 2), dtype=torch.float32, device=vis_1.device)
        corr_mask = torch.ones(size=(h, w), dtype=torch.uint8, device=vis_1.device)
        corr_mask[vis_1==invisible] = 2
        yy, xx = torch.meshgrid(torch.arange(h, device=vis_1.device), torch.arange(w, device=vis_1.device))
        end_time = time.time()
        for face_id in visible_face_1:
            vis_mask = (vis_1==face_id)
            pts_1 = torch.stack([xx[vis_mask], yy[vis_mask]]).t() #(N, 2)
            # barycentric coordinate transformation
            vert_ids = faces[face_id] #[i_a, i_b, i_c]
            tri_1 = verts_1[vert_ids] #(3, 2)
            tri_2 = verts_2[vert_ids]
            pts_bc = get_barycentric_coords_cuda(pts_1, tri_1) #(N, 3)
            pts_2 = torch.mm(pts_bc, tri_2) #(N, 2)

            corr_map[vis_mask] = pts_2
            if face_id in visible_face_2:
                corr_mask[vis_mask] = 0
        last_time = time.time()
        print("Time cost: ", last_time - end_time)
        print(torch.max(corr_map), torch.mean(corr_map), torch.min(corr_map))
        corr_map[:, :, 0] += 32
        corr_map = torch.nn.functional.pad(corr_map,(0,0, 32,32, 0,0),'constant',value=0)
        corr_mask = torch.nn.functional.pad(corr_mask.unsqueeze(2),(0,0, 32,32),'constant',value=0)

        corr_maps.append(corr_map)
        corr_masks.append(corr_mask)

    corr_maps = torch.stack(corr_maps)
    corr_masks = torch.stack(corr_masks)
    return corr_maps, corr_masks


def get_barycentric_coords_cuda(pts, triangle):
    '''
    Compute the barycentric coordinates of a set of points respect to a given triangle.
    Input:
        pts: (N, 2), points coordinates in original space
        triangle: (3, 2), triangle vertices
    Output:
        pts_bc: (N, 3), barycentirc coordinates
    '''
    a, b, c = triangle
    v0 = b-a
    v1 = c-a
    v2 = pts - a
    d00 = v0.dot(v0) # scalar
    d01 = v0.dot(v1) # scalar
    d11 = v1.dot(v1) # scalar
    d20 = torch.mm(v2, v0.unsqueeze(1)) # (N,)
    d21 = torch.mm(v2, v1.unsqueeze(1)) # (N,)
    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    u = 1. - v - w
    return torch.cat([u,v,w], dim=1)


def calc_correspondence_from_smpl_internal_cuda(verts_1, vis_1, verts_2, vis_2, faces):
    '''
    Compute for each pixel (x,y) in img_1 the corresponding pixel (u,v) in img_2.
    Input:
        pred_1: HMR prediction for image 1. see calc_correspondence_from_smpl.
        pred_2: HMR prediction for image 2
        faces: list, each element is a triangle face represented by 3 vertex indices (i_a, i_b, i_c)
    Output:
        corr_map: (img_size, img_size, 2). corr_map[y,x] = (u,v)
        corr_mask: (img_size, img_size). The value is one of:
            0: human pixel with correspondence in img_1
            1: human pixel without correspondece in img_1
            2: background pixel
    '''
    faces = torch.from_numpy(faces.astype(np.float32)).to(vis_1.device)# .to(torch.long)
    bc, h, w = vis_1.shape

    start_time = time.time()
    verts_1, verts_2 = verts_1[:, :, :2], verts_2[:, :, :2]
    invisible = -1 # 4294967295

    visible_faces_2 = torch.zeros(size=(bc, faces.size(0)), dtype=torch.float32, device=vis_1.device)
    pad_vis_func = cal_corr.pad_visible_map
    pad_vis_func(visible_faces_2, vis_2)
    # for i in range(bc):
    #     visible_face_ls_2 = torch.unique(vis_2[i], sorted=True).to(torch.long)
    #     visible_faces_2[i, visible_face_ls_2] = 1.
  
    corr_maps = torch.zeros(size=(bc, h, w, 2), dtype=torch.float32, device=vis_1.device)
    corr_masks = torch.ones(size=(bc, h, w), dtype=torch.float32, device=vis_1.device)
    cal_corr_func = cal_corr.cal_corr

    cal_corr_func(vis_1, faces, verts_1, verts_2, visible_faces_2, corr_maps, corr_masks)
    end_time = time.time()
    print("Time cost: ", end_time - start_time)
    print(torch.max(corr_maps), torch.mean(corr_maps), torch.min(corr_maps))

    corr_maps[:, :, 0] += 32
    corr_maps = torch.nn.functional.pad(corr_maps,(0,0, 32,32, 0,0),'constant',value=0)
    corr_masks = torch.nn.functional.pad(corr_masks.unsqueeze(3),(0,0,32,32),'replicate')

    import matplotlib.pyplot as plt
    vis_mask = corr_masks[0, :, :, 0].detach().cpu().numpy()
    plt.imshow(vis_mask)
    plt.show()

    return corr_maps, corr_masks


if __name__ == '__main__':
    with open(os.path.join(data_root, 'smpl', source.replace('.jpg', '.pkl')), "rb") as f:
        smpl_A = pickle.load(f)
    verts_A = smpl_A['pred_vertices_img']
    vis_A = smpl_A['visibility']

    with open(os.path.join(data_root, 'smpl', target.replace('.jpg', '.pkl')), "rb") as f:
        smpl_B = pickle.load(f)
    verts_B = smpl_B['pred_vertices_img']
    vis_B = smpl_B['visibility']

    # test python code
    verts_A1 = torch.from_numpy(np.array(verts_A)).cuda()
    vis_A1 = torch.from_numpy(np.array(vis_A, dtype=np.int64)).cuda()

    verts_B1 = torch.from_numpy(np.array(verts_B)).cuda()
    vis_B1 = torch.from_numpy(np.array(vis_B, dtype=np.int64)).cuda()

    verts_A1 = verts_A1.unsqueeze(0)
    vis_A1 = vis_A1.unsqueeze(0)
    verts_B1 = verts_B1.unsqueeze(0)
    vis_B1 = vis_B1.unsqueeze(0)

    calc_corr_python_cuda(verts_A1, vis_A1, verts_B1, vis_B1, SMPL_Faces)

    # test cuda code
    valid_index_A = (vis_A == 4294967295)
    vis_A = vis_A.astype(np.float32)
    vis_A[valid_index_A] = -1

    valid_index_B = (vis_B == 4294967295)
    vis_B = vis_B.astype(np.float32)
    vis_B[valid_index_B] = -1

    verts_A = torch.from_numpy(np.array(verts_A)).cuda()
    vis_A = torch.from_numpy(np.array(vis_A)).cuda()

    verts_B = torch.from_numpy(np.array(verts_B)).cuda()
    vis_B = torch.from_numpy(np.array(vis_B)).cuda()

    verts_A = verts_A.unsqueeze(0)
    vis_A = vis_A.unsqueeze(0)
    verts_B = verts_B.unsqueeze(0)
    vis_B = vis_B.unsqueeze(0)

    cal_corr = load(name='cal_corr', sources=['cal_corr.cpp', 'cal_corr_cuda.cu'])
    calc_correspondence_from_smpl_internal_cuda(verts_A, vis_A, verts_B, vis_B, SMPL_Faces)