# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import cv2
import math
import json
import pickle
import numpy as np
from PIL import Image, ImageOps
import pycocotools.mask as maskUtils
from dataset.base_dataset import BaseDataset, get_params, get_transform

                   
class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true', help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.input_size = opt.input_size
        h, w = self.input_size.split('_')
        self.padding_size = (eval(h) - eval(w)) // 2
        label_paths, image_paths, semantic_paths = self.get_paths(opt)
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
        self.label_paths = label_paths
        self.image_paths = image_paths
        self.semantic_paths = semantic_paths
        size = len(self.label_paths)
        self.dataset_size = size
        self.ref_dict, self.train_test_folder = self.get_ref(opt)
        self.keypoints_pool = {}

        self.smpl_faces = np.load('./smpl/smpl_faces.npy')
        self.smpl_faces = np.concatenate((self.smpl_faces, np.array([6890,6890,6890], dtype=np.uint32)[np.newaxis,...]))

        self.vertices_sem_idx = np.load('./smpl/vertices_semantic_idxs.npy')
        self.vertices_sem_idx = np.concatenate((self.vertices_sem_idx, np.array([-1], dtype=np.int32)))
        
        self.symmetric_table = np.load('./smpl/vertices_symm_idxs.npy')
        self.symmetric_table = np.concatenate((self.symmetric_table, np.array([-1], dtype=np.int))).astype(np.int32)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        semantic_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, semantic_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path, params1):
        label = Image.open(path)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc
        # 'unknown' is opt.label_nc
        return label_tensor

    def get_densepose(self, path, left_padding, right_padding):
        with open(path, "rb") as f:
            densepose = pickle.load(f)
        x0, y0, x1, y1 = densepose['pred_boxes_XYXY']
        # for key in densepose.keys():
        #     densepose[key] = densepose[key].detach().cpu().numpy()
        # uv = densepose['pred_densepose_uv']
        label = densepose['pred_densepose_label'][np.newaxis,...].astype(np.float32) * 10
        uv = densepose['pred_densepose_uv'].astype(np.float32)
        label_uv = np.concatenate([label, uv], axis=0)
        h, w = self.input_size.split('_')
        uv_h, uv_w = label_uv.shape[-2], label_uv.shape[-1]
        pad_w = int(x0), eval(w) - uv_w - int(x0) 
        pad_h = int(y0), eval(h) - uv_h - int(y0) 
        pad_uv = np.pad(label_uv, ((0,0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), 'constant', constant_values=(0,0))
        pad_uv = np.pad(pad_uv,((0,0), (0,0), (left_padding, right_padding)),'constant',constant_values=(0,0))
        return pad_uv

    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask

    def get_hand_mask(self, hand_keypoints):
        # 肩，肘，腕
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((h,w,1),dtype=np.float32)
        bottom_mask = np.ones((h,w,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            # 对上半部分进行膨胀操作，消除两部分之间的空隙
            kernel = np.ones((25,25),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((15,15),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask
        
    def get_palm(self, keypoints, parsing, left_padding, right_padding):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_keypoints[:,0] += left_padding
        right_hand_keypoints[:,0] += left_padding
        # parsing = np.pad(parsing, ((0,0),(left_padding,right_padding),(0,0)), 'constant', constant_values=(0,0))

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask[...,np.newaxis], left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask[...,np.newaxis], right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    def get_smpl_semantic_region(self, smpl_person, left_padding, right_padding):
        visibility = smpl_person['visibility'].copy()
        visibility[visibility == 4294967295] = len(self.smpl_faces)-1
        visibility[visibility == 4294901760] = len(self.smpl_faces)-1
        # out of bound
        visibility[visibility >= 13777] = len(self.smpl_faces)-1

        vertices = self.smpl_faces[visibility].astype(np.int32)
        
        # visible_vertices = self.vertices_sem_idx[vertices[...,0]]
        # arm_region_idx = np.array([0, 2, 8, 9, 12, 13, 15, 16, 17, 22], dtype=np.uint8)
        # arm_region = np.isin(visible_vertices, arm_region_idx)
        # for ii in range(1,3):
        #     arm_region = np.logical_or(arm_region, 
        #                     np.isin(self.vertices_sem_idx[vertices[...,ii]], arm_region_idx))
        
        # body_region_idx = np.array([20, 6, 7, 21, 1, 23], dtype=np.uint8)
        # body_region = np.isin(visible_vertices, body_region_idx)
        # for ii in range(1,3):
        #     body_region = np.logical_or(body_region, 
        #                     np.isin(self.vertices_sem_idx[vertices[...,ii]], body_region_idx))
        
        # # kernel = np.ones((4, 4), np.uint8)
        # # arm_region= cv2.erode(arm_region.astype(np.uint8), kernel, iterations=1)
        # # body_region= cv2.erode(body_region.astype(np.uint8), kernel, iterations=1)

        # smpl_sem = (arm_region * -1 + body_region * 1)[...,np.newaxis]
        # smpl_sem = np.pad(smpl_sem,((0,0),(left_padding, right_padding), (0,0)),'constant',constant_values=(0,0))

        return None, vertices


    def get_joints(self, keypoints_path, affine_matrix=None, coeffs=None, name='query'):
        with open(keypoints_path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18,3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        canvas = np.zeros((512, 512, 3), dtype=np.uint8)
        # drop circle on each keypoints
        cycle_radius = 5 
        for i in range(18):
            joint = keypoints[i]
            if joint[2] < 0.1:
                continue
            x, y = int(joint[0]), int(joint[1])
            x = x + 32
            cv2.circle(canvas, (int(x), int(y)), cycle_radius, colors[i], thickness=-1)
        
        # draw eclipse on each connection
        stickwidth = 5 
        joints = []
        for i in range(17):
            index = np.array(limbSeq[i]) - 1
            cur_canvas = canvas.copy()
            if (keypoints[index.astype(int), 2] < 0.1).any():
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue

            Y = keypoints[index.astype(int), 0] + 32                 # add offset as line 118 do
            X = keypoints[index.astype(int), 1] 
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        # rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)        # this sentence throw error, confusing.
        pose = canvas
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = (np.clip((im_dist/3), 0, 255).astype(np.uint8))[...,np.newaxis]
            ims_dist = im_dist if e == 1 else np.concatenate([ims_dist, im_dist], axis=2)
            e += 1
        # tensor_pose = transform_label(pose)
        color_joints = np.concatenate((pose, ims_dist), axis=2)
        return keypoints, color_joints


    def get_mask_from_kps(self, kps, img_h, img_w):
        rles = maskUtils.frPyObjects(kps, img_h, img_w)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)[...,np.newaxis].astype(np.float32)
        mask = mask * 255.0
        return mask

    def get_rectangle_mask(self, a, b, c, d, img_h, img_w):
        x1, y1 = a + (b-d)/4,   b + (c-a)/4
        x2, y2 = a - (b-d)/4,   b - (c-a)/4

        x3, y3 = c + (b-d)/4,   d + (c-a)/4
        x4, y4 = c - (b-d)/4,   d - (c-a)/4

        kps  = [x1,y1,x2,y2]

        v0_x, v0_y = c-a,   d-b
        v1_x, v1_y = x3-x1, y3-y1
        v2_x, v2_y = x4-x1, y4-y1

        cos1 = (v0_x*v1_x+v0_y*v1_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v1_x*v1_x+v1_y*v1_y))
        cos2 = (v0_x*v2_x+v0_y*v2_y) / (math.sqrt(v0_x*v0_x+v0_y*v0_y)*math.sqrt(v2_x*v2_x+v2_y*v2_y))

        if cos1<cos2:
            kps.extend([x3,y3,x4,y4])
        else:
            kps.extend([x4,y4,x3,y3])

        kps = np.array(kps).reshape(1,-1).tolist()
        mask = self.get_mask_from_kps(kps, img_h=img_h, img_w=img_w)

        return mask
    
    def get_hand_mask(self, hand_keypoints):
        # 肩，肘，腕
        s_x,s_y,s_c = hand_keypoints[0]
        e_x,e_y,e_c = hand_keypoints[1]
        w_x,w_y,w_c = hand_keypoints[2]

        h, w = 512, 512
        up_mask = np.ones((h,w,1),dtype=np.float32)
        bottom_mask = np.ones((h,w,1),dtype=np.float32)
        if s_c > 0.1 and e_c > 0.1:
            up_mask = self.get_rectangle_mask(s_x, s_y, e_x, e_y, h, w)
            # 对上半部分进行膨胀操作，消除两部分之间的空隙
            kernel = np.ones((25,25),np.uint8)
            up_mask = cv2.dilate(up_mask,kernel,iterations=1)
            up_mask = (up_mask > 0).astype(np.float32)[...,np.newaxis]
        if e_c > 0.1 and w_c > 0.1:
            bottom_mask = self.get_rectangle_mask(e_x, e_y, w_x, w_y, h, w)
            kernel = np.ones((15,15),np.uint8)
            bottom_mask = cv2.dilate(bottom_mask,kernel,iterations=1)
            bottom_mask = (bottom_mask > 0).astype(np.float32)[...,np.newaxis]

        return up_mask, bottom_mask

    def get_palm_mask(self, hand_mask, hand_up_mask, hand_bottom_mask):
        inter_up_mask = ((hand_mask + hand_up_mask) == 2).astype(np.float32)
        hand_mask = hand_mask - inter_up_mask
        inter_bottom_mask = ((hand_mask+hand_bottom_mask)==2).astype(np.float32)
        palm_mask = hand_mask - inter_bottom_mask

        return palm_mask

    def get_palm(self, keypoints, parsing, left_padding, right_padding):
        left_hand_keypoints = keypoints[[5,6,7],:].copy()
        right_hand_keypoints = keypoints[[2,3,4],:].copy()

        left_hand_keypoints[:,0] += left_padding
        right_hand_keypoints[:,0] += left_padding

        left_hand_up_mask, left_hand_botton_mask = self.get_hand_mask(left_hand_keypoints)
        right_hand_up_mask, right_hand_botton_mask = self.get_hand_mask(right_hand_keypoints)

        # mask refined by parsing
        left_hand_mask = (parsing == 14).astype(np.float32)
        right_hand_mask = (parsing == 15).astype(np.float32)
        left_palm_mask = self.get_palm_mask(left_hand_mask, left_hand_up_mask, left_hand_botton_mask)
        right_palm_mask = self.get_palm_mask(right_hand_mask, right_hand_up_mask, right_hand_botton_mask)
        palm_mask = ((left_palm_mask + right_palm_mask) > 0).astype(np.uint8)

        return palm_mask

    def __getitem__(self, index):
        left_padding = right_padding = self.padding_size
        h, w = self.input_size.split('_')
        params1 = get_params(self.opt, (w,h))

        # label_path = self.label_paths[index]
        # label_path = label_path 
        # label_tensor, keypoints = self.get_label_tensor(label_path)
        
        image_path = self.image_paths[index]
        image_path = image_path
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.expand(image, border=(left_padding,0,right_padding,0), fill=(255,255,255))       # padding, left,top,right,bottom
        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        semantic_path = self.semantic_paths[index]
        semantic_path = semantic_path 
        parsing = cv2.imread(semantic_path)[..., 0:1]
        parsing = np.array(parsing[:, :, 0], dtype=np.uint8)
        parsing = np.pad(parsing,((0,0),(left_padding, right_padding)),'constant',constant_values=(0,0))

        tar_densepose_path = image_path.replace('image', 'densepose').replace('.png', '.pkl')
        target_uv = self.get_densepose(tar_densepose_path, left_padding, right_padding)

        tar_smpl_path = image_path.replace("/image/", "/smpl/").replace(".png", ".pkl")
        with open(tar_smpl_path, "rb") as f:
            target_smpl = pickle.load(f)
            target_smpl['visibility'] = target_smpl['visibility'].astype(np.int64)
        target_smpl_sem, target_vertices = self.get_smpl_semantic_region(target_smpl, left_padding, right_padding)

        # reference
        ref_name = self.image_ref_paths[index]
        path_ref = os.path.join(self.opt.dataroot, 'image', ref_name)
        image_ref = Image.open(path_ref).convert('RGB')
        image_ref = ImageOps.expand(image_ref, border=(left_padding,0,right_padding,0), fill=(255,255,255))       # padding, left,top,right,bottom
        
        # label_ref_path = self.imgpath_to_labelpath(path_ref)
        # label_ref_tensor, ref_keypoints = self.get_label_tensor(label_ref_path)
        transform_image = get_transform(self.opt, params1)
        ref_tensor = transform_image(image_ref) 

        semantic_ref_path = path_ref.replace('image', 'parsing').replace('.png','_label.png')
        parsing_ref = cv2.imread(semantic_ref_path)[..., 0:1]
        parsing_ref = np.array(parsing_ref[:, :, 0], dtype=np.uint8)
        parsing_ref = np.pad(parsing_ref,((0,0),(left_padding, right_padding)),'constant',constant_values=(0,0))

        src_densepose_path = path_ref.replace('image', 'densepose').replace('.png', '.pkl')
        source_uv = self.get_densepose(src_densepose_path, left_padding, right_padding)
        
        src_smpl_path = path_ref.replace("/image/", "/smpl/").replace(".png", ".pkl")
        with open(src_smpl_path, "rb") as f:
            source_smpl = pickle.load(f)
            source_smpl['visibility'] = source_smpl['visibility'].astype(np.int64)

        source_smpl_sem, source_vertices = self.get_smpl_semantic_region(source_smpl, left_padding, right_padding)

        src_visible_table = np.zeros(shape=(6891,), dtype=np.uint8)
        visible_vertices = list(set(source_vertices.flatten()))
        src_visible_table[visible_vertices] = 1
        src_visible_table[-1] = 0

        target_sym_vertices = []
        tar_symmetric_vert = self.symmetric_table[target_vertices[...,0]]
        target_sym_vertices.append(tar_symmetric_vert)
        visible_sym_region = src_visible_table[tar_symmetric_vert]
        for ii in range(1,3):
            tar_symmetric_vert = self.symmetric_table[target_vertices[...,ii]]
            target_sym_vertices.append(tar_symmetric_vert)
            visible_sym_region = np.logical_and(visible_sym_region, 
                                        src_visible_table[tar_symmetric_vert])

        input_dict = {
                    'image': image_tensor,
                    'path': image_path,
                    'ref': ref_tensor,
                    'parsing_array': parsing,
                    'parsing_ref_array': parsing_ref,
                    'source_vertices_img' : source_smpl['pred_vertices_img'],
                    'source_visibility' : source_smpl['visibility'],
                    'source_densepose' : source_uv,
                    'target_vertices_img' : target_smpl['pred_vertices_img'],
                    'target_visibility' : target_smpl['visibility'],
                    'target_densepose': target_uv,
                    'symmetric_mask': visible_sym_region,
                    'index': index,
                    }
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path
