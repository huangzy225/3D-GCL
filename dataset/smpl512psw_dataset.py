# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import math
import json
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from PIL import Image, ImageOps
from dataset.pix2pix512psw_dataset import Pix2pixDataset
from dataset.base_dataset import get_params, get_transform


bparts = [
        ["lshoulder","lhip","rhip","rshoulder"],
        ["lshoulder", "rshoulder", "cnose"],
        ["lshoulder","lelbow"],
        ["lelbow", "lwrist"],
        ["rshoulder","relbow"],
        ["relbow", "rwrist"]]
        # ["lhip", "lknee"],
        # ["lknee", "lankle"],
        # ["rhip", "rknee"],
        # ["rknee", "rankle"]]

order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
        'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 
        'lankle', 'reye', 'leye', 'rear', 'lear']

class SMPL512pswDataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        parser.set_defaults(load_size=550)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(label_nc=20)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.add_argument('--extend_flat', action='store_true')
        parser.add_argument('--extend_aug', action='store_true')

        return parser

    def get_paths(self, opt):
        img_res = opt.input_size 
        self.img_h, self.img_w= img_res.split('_')
        if opt.phase == 'test':
            pairs_file = pd.read_csv('./data/test_psw512.csv')
        else:
            pairs_file = pd.read_csv('./data/train_psw512.csv')

        self.pairs = []
        self.sources = {}
        print('Loading data pairs ...')
        for i in range(len(pairs_file)):
            pair = [pairs_file.iloc[i]['from'], pairs_file.iloc[i]['to']]
            self.pairs.append(pair)

        image_paths = []
        self.image_ref_paths = []
        label_paths = []
        semantic_paths = []
        self.code = []
        for i in range(len(self.pairs)):
            src_im, tar_im = self.pairs[i]
            name = os.path.join(opt.dataroot, 'image', tar_im)
            image_paths.append(name)
            label_path = name.replace('image', 'keypoints').replace('.png', '_keypoints.json')
            label_paths.append(os.path.join(label_path))
            semantic_path = name.replace('image', 'parsing').replace('.png', '_label.png')
            semantic_paths.append(os.path.join(semantic_path))
            self.code.append(0)
            self.image_ref_paths.append(os.path.join(opt.dataroot, 'image', src_im))

        return label_paths, image_paths, semantic_paths

    def get_ref_no_back(self, opt):
        ref_dict = {}
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref_vgg(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open('./data/deepfashion_ref{}_{}.txt'.format(extra, self.img_h)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = [it for it in items[1:]]
            else:
                val = [items[-1]]
            ref_dict[key.replace('\\',"/")] = [v.replace('\\',"/") for v in val]
        train_test_folder = ('', '')
        return ref_dict, train_test_folder

    def get_ref(self, opt):
        if opt.video_like:
            return self.get_ref_no_back(opt)
        else:
            return self.get_ref_vgg(opt)

    def get_label_tensor(self, path, name='query'):
        with open(path, 'r') as f:
            keypoints_data = json.load(f)
        if len(keypoints_data['people']) == 0:
            keypoints = np.zeros((18, 3))
        else:
            keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        self.keypoints_pool[name] = keypoints
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        canvas = np.zeros((self.opt.load_size, self.opt.load_size, 3), dtype=np.uint8)
        # drop circle on each keypoints
        cycle_radius = 5 * self.opt.load_size // 256
        for i in range(18):
            joint = keypoints[i]
            if joint[2] < 0.1:
                continue
            x, y = int(joint[0]), int(joint[1])
            x = x + self.padding_size
            cv2.circle(canvas, (int(x), int(y)), cycle_radius, colors[i], thickness=-1)
        
        # draw eclipse on each connection
        stickwidth = 5 * self.opt.load_size // 256
        joints = []
        for i in range(17):
            index = np.array(limbSeq[i]) - 1
            cur_canvas = canvas.copy()
            if (keypoints[index.astype(int), 2] < 0.1).any():
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue

            Y = keypoints[index.astype(int), 0] + self.padding_size                  # add offset as line 118 do
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
        pose = Image.fromarray(canvas).resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)
        params = get_params(self.opt, pose.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist/3), 0, 255).astype(np.uint8)
            tensor_dist = transform_img(Image.fromarray(im_dist))
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1
        tensor_pose = transform_label(pose)
        label_tensor = torch.cat((tensor_pose, tensors_dist), dim=0)
        
        return label_tensor, keypoints, params

    def imgpath_to_labelpath(self, path):
        label_path = path.replace('/image/', '/keypoints/').replace('.jpg', '_keypoints.json')
        return label_path

    def labelpath_to_imgpath(self, path):
        img_path = path.replace('/pose/', '/img/').replace('_{}.txt', '.jpg')
        return img_path

    def valid_joints(self, joint):
        return (joint >= 0.1).all()

    def get_part_corner(self, joints, bpart, wh, o_w, o_h, ar = 1.0):
        bpart_indices = [order.index(b) for b in bpart]
        part_src = np.float32(joints[bpart_indices][:, :2])
        # fall backs
        if not self.valid_joints(joints[bpart_indices][:, 2]):
            if bpart[0] == "lhip" and bpart[1] == "lknee":      
                bpart = ["lhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "rhip" and bpart[1] == "rknee":   
                bpart = ["rhip"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])
            # elif bpart[0] == "lknee" and bpart[1] == "lankle":
            #     bpart = ["lknee"]
            #     bpart_indices = [order.index(b) for b in bpart]
            #     part_src = np.float32(joints[bpart_indices][:,:2])
            # elif bpart[0] == "rknee" and bpart[1] == "rankle":
            #     bpart = ["rknee"]
            #     bpart_indices = [order.index(b) for b in bpart]
            #     part_src = np.float32(joints[bpart_indices][:,:2])
            elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose":
                bpart = ["lshoulder", "rshoulder", "rshoulder"]
                bpart_indices = [order.index(b) for b in bpart]
                part_src = np.float32(joints[bpart_indices][:,:2])

        if not self.valid_joints(joints[bpart_indices][:, 2]):
                return None
        part_src[:, 0] = part_src[:, 0] + abs(o_h - o_w) // 2                    # correct axis by adding pad size 

        if part_src.shape[0] == 1:
            # leg fallback
            a = part_src[0]
            b = np.float32([a[0],o_h - 1])
            part_src = np.float32([a,b])

        if part_src.shape[0] == 4:
            pass
        elif part_src.shape[0] == 3:
            # lshoulder, rshoulder, cnose
            if bpart == ["lshoulder", "rshoulder", "rshoulder"]:
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                if normal[1] > 0.0:
                    normal = -normal

                a = part_src[0] + normal
                b = part_src[0]
                c = part_src[1]
                d = part_src[1] + normal
                part_src = np.float32([a,b,c,d])
            else:
                assert bpart == ["lshoulder", "rshoulder", "cnose"]
                neck = 0.5*(part_src[0] + part_src[1])
                neck_to_nose = part_src[2] - neck
                part_src = np.float32([neck + 2*neck_to_nose, neck])

                # segment box
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1],segment[0]])
                alpha = 1.0 / 2.0
                a = part_src[0] + alpha*normal
                b = part_src[0] - alpha*normal
                c = part_src[1] - alpha*normal
                d = part_src[1] + alpha*normal
                #part_src = np.float32([a,b,c,d])
                part_src = np.float32([b,c,d,a])
        else:
            assert part_src.shape[0] == 2

            segment = part_src[1] - part_src[0]
            normal = np.array([-segment[1],segment[0]])
            alpha = ar / 2.0
            a = part_src[0] + alpha*normal
            b = part_src[0] - alpha*normal
            c = part_src[1] - alpha*normal
            d = part_src[1] + alpha*normal
            part_src = np.float32([a,b,c,d])
        return part_src

    def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask, \
                     pose, box_factor):
        h, w = upper_img.shape[:2]
        o_h, o_w = h, w
        h = h // 2**box_factor
        w = w // 2**box_factor
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)

        bparts = [
                ["lshoulder","lhip","rhip","rshoulder"],
                ["lshoulder", "rshoulder", "cnose"],
                ["lshoulder","lelbow"],
                ["lelbow", "lwrist"],
                ["rshoulder","relbow"],
                ["relbow", "rwrist"],
                ["lhip", "lknee"],
                ["lknee", "lankle"],
                ["rhip", "rknee"],
                ["rknee", "rankle"]]

        order = ['cnose', 'cneck', 'rshoulder', 'relbow', 'rwrist', 'lshoulder', 
                'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 
                'lankle', 'reye', 'leye', 'rear', 'lear']
        ar = 0.5

        part_imgs = list()
        part_stickmen = list()
        M_invs = list()
        denorm_hand_masks = list()
        denorm_leg_masks = list()
        part_clothes_masks = list()

        denorm_upper_img = np.zeros_like(upper_img)
        denorm_lower_img = np.zeros_like(upper_img)

        for ii, bpart in enumerate(bparts):
            part_img = np.zeros((h, w, 3)).astype(np.uint8)
            part_stickman = np.zeros((h, w, 3)).astype(np.uint8)
            part_clothes_mask = np.zeros((h,w,3)).astype(np.uint8)
            M, M_inv = self.get_crop(bpart, order, wh, o_w, o_h, ar)

            if M is not None:
                if ii < 6:
                    part_img = cv2.warpPerspective(upper_img, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_clothes_mask = cv2.warpPerspective(upper_clothes_mask, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                else:
                    part_img = cv2.warpPerspective(lower_img, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                    part_clothes_mask = cv2.warpPerspective(lower_clothes_mask, M, (w,h), borderMode = cv2.BORDER_REPLICATE)
                part_stickman = cv2.warpPerspective(pose, M, (w,h), borderMode = cv2.BORDER_REPLICATE)

                denorm_patch = cv2.warpPerspective(part_img, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.warpPerspective(part_clothes_mask, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)[...,0:1]
                denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)
                
                if ii < 6:
                    denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)
                else:
                    denorm_lower_img = denorm_patch * denorm_clothes_mask_patch + denorm_lower_img * (1-denorm_clothes_mask_patch)
                
                M_invs.append(M_inv[np.newaxis,...])
            else:
                M_invs.append(np.zeros((1,3,3),dtype=np.float32))

            if ii >= 2 and ii <= 5:
                if M is not None:
                    denorm_hand_masks.append(denorm_clothes_mask_patch)
                else:
                    denorm_hand_masks.append(np.zeros_like(upper_img)[...,0:1])
            if ii == 7 or ii ==9:
                if M is not None:
                    denorm_leg_masks.append(denorm_clothes_mask_patch)
                else:
                    denorm_leg_masks.append(np.zeros_like(lower_img)[...,0:1])

            part_imgs.append(part_img)
            part_stickmen.append(part_stickman)
            part_clothes_masks.append(part_clothes_mask)

        img = np.concatenate(part_imgs, axis = 2)
        stickman = np.concatenate(part_stickmen, axis = 2)
        clothes_masks = np.concatenate(part_clothes_masks, axis=2)
        M_invs = np.concatenate(M_invs, axis=0)

        return img, stickman, denorm_upper_img, denorm_lower_img, M_invs, denorm_hand_masks, denorm_leg_masks, clothes_masks

    def denorm_clothes(self, norm_patches, M_invs, norm_clothes_mask, col, row, gnum):
        denorm_upper_img = np.zeros((256,256,3),dtype=np.uint8)
        denorm_lower_img = np.zeros((256,256,3),dtype=np.uint8)

        kernel = np.ones((5,5),np.uint8)

        gap = gnum // 3
        for ii in range(M_invs.shape[1]):
            ################ upper-body
            if ii < 6:
                ####### 0-(gap-1) visualize lower-body tryon，index of tops is row
                if row < gap:
                    norm_patch = norm_patches[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                    norm_clothes_mask_patch = norm_clothes_mask[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                ###### gap-(2*gap-1) visualize full-body tryon，gap-(gum-1) visualize upper-body tryon，index of top is col
                else:
                    norm_patch = norm_patches[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                    norm_clothes_mask_patch = norm_clothes_mask[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
            ################## lower-body
            else:
                ###### 0-(gap-1) visualize lower-body tryon，gap-(2*gap-1) visualize full-body tryon，index of pants is col 
                if row < 2 * gap:
                    norm_patch = norm_patches[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                    norm_clothes_mask_patch = norm_clothes_mask[col,ii*3:(ii+1)*3,...].transpose(1,2,0)
                ########## gap-(gum-1) visualize upper-body tryon，index of pants is row
                else:
                    norm_patch = norm_patches[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                    norm_clothes_mask_patch = norm_clothes_mask[row,ii*3:(ii+1)*3,...].transpose(1,2,0)
                    # M_inv = M_invs[row,ii]
            M_inv = M_invs[row,ii]
            if M_inv.sum() == 0:
                continue
            denorm_patch = cv2.warpPerspective(norm_patch,M_inv,(256,256),borderMode=cv2.BORDER_CONSTANT)
            denorm_clothes_mask_patch = cv2.warpPerspective(norm_clothes_mask_patch,M_inv,(256,256),borderMode=cv2.BORDER_CONSTANT)
            denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)[...,0:1]
            denorm_clothes_mask_patch = (denorm_clothes_mask_patch==255).astype(np.uint8)
            
            if ii < 6:
                denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)
            else:
                denorm_lower_img = denorm_patch * denorm_clothes_mask_patch + denorm_lower_img * (1-denorm_clothes_mask_patch)

        denorm_upper_img = denorm_upper_img.transpose(2,0,1)[np.newaxis,...]
        denorm_lower_img = denorm_lower_img.transpose(2,0,1)[np.newaxis,...]
        denorm_upper_clothes_mask = (np.sum(denorm_upper_img,axis=1,keepdims=True)>0).astype(np.uint8)
        denorm_lower_clothes_mask = (np.sum(denorm_lower_img,axis=1,keepdims=True)>0).astype(np.uint8)

        return denorm_upper_img, denorm_lower_img, denorm_upper_clothes_mask, denorm_lower_clothes_mask

    def patchwise_transformation(self, image, upper_cloth_mask):
        image = np.array(image)
        upper_cloth_mask = np.stack([upper_cloth_mask, upper_cloth_mask, upper_cloth_mask], axis=2)
        h, w = image.shape[:2]
        o_h, o_w = h, w
        wh = np.array([w, h])
        wh = np.expand_dims(wh, 0)
        kernel = np.ones((5,5),np.uint8)

        ar = 0.5
        denorm_upper_img = np.zeros_like(image)

        for ii, bpart in enumerate(bparts):
            part_src = self.get_part_corner(self.keypoints_pool['query'], bpart, wh, o_w, o_h, ar)
            part_dst = self.get_part_corner(self.keypoints_pool['reference'], bpart, wh, o_w, o_h, ar)
            
            if part_src is not None and part_dst is not None:
                dst = np.float32((wh-1) * [[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
                M_norm = cv2.getPerspectiveTransform(part_src, dst)
                M_denorm = cv2.getPerspectiveTransform(dst, part_dst)

                norm_patch = cv2.warpPerspective(image, M_norm, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_patch = cv2.warpPerspective(norm_patch, M_denorm, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                norm_clothes_mask_patch = cv2.warpPerspective(upper_cloth_mask, M_norm, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.warpPerspective(norm_clothes_mask_patch, M_denorm, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
                denorm_clothes_mask_patch = cv2.erode(denorm_clothes_mask_patch, kernel, iterations=1)
                denorm_upper_img = denorm_patch * denorm_clothes_mask_patch + denorm_upper_img * (1-denorm_clothes_mask_patch)   
                denorm_clothes_mask_patch = (denorm_clothes_mask_patch==1).astype(np.uint8)
        
        denorm_upper_clothes_mask = (np.sum(denorm_upper_img,axis=2)>0).astype(np.uint8) * 5

        return denorm_upper_img, denorm_upper_clothes_mask

    def __getitem__(self, index):
        return super().__getitem__(index)

    def get_visualize_batch(self, opt):
        vis_image_path = []
        vis_label_path = []
        vis_semantic_path = []

        with open('./data/vis_index3.txt', 'r') as f:
            vis_name = [it.strip() for it in f.readlines()]
            for name in vis_name:
                name = os.path.join('./data/vis_data', name)
                vis_image_path.append(name)
                label_path = name.replace('image', 'keypoints').replace('.jpg', '_keypoints.json')
                vis_label_path.append(os.path.join(label_path))
                semantic_path = name.replace('image', 'parsing').replace('.jpg', '_label.png')
                vis_semantic_path.append(os.path.join(semantic_path))

        batch_label_tensor = []
        batch_image_tensor = []
        batch_image_path = []
        batch_is_aug_flag = []
        batch_label_ref_tensor = []
        batch_ref_tensor = []
        batch_parsing = []
        batch_fg_parsing = []
        batch_fg_parsing_ref = []
        batch_parsing_ref = []
        batch_source_vertices_img = []
        batch_source_visibility = []
        batch_source_smpl_semantic = []
        batch_source_smpl_vertices = []
        batch_source_densepsoe = []
        batch_target_vertices_img = []
        batch_target_visibility = []
        batch_target_smpl_semantic = []
        batch_target_smpl_sym_vertices = []
        batch_target_densepose = []
        batch_symmetric_mask = []
        batch_palm_mask = []
        # for index in range(len(vis_image_path)):
        for index in range(10):
            # label Image
            label_path = vis_label_path[index]
            label_path = label_path # os.path.join(self.opt.dataroot, label_path)
            label_tensor, keypoints, params1 = self.get_label_tensor(label_path)
            # input image (real images)
            left_padding = right_padding = self.padding_size
            image_path = vis_image_path[index]
            image_path = image_path # os.path.join(self.opt.dataroot, image_path)
            image = Image.open(image_path).convert('RGB')
            image = ImageOps.expand(image, border=(left_padding,0,right_padding,0), fill=(255,255,255))       # padding, left,top,right,bottom
            transform_image = get_transform(self.opt, params1)
            image_tensor = transform_image(image)

            tar_smpl_path = image_path.replace("/image/", "/smpl/").replace(".jpg", ".pkl")
            with open(tar_smpl_path, "rb") as f:
                target_smpl = pickle.load(f)
                target_smpl['visibility'] = target_smpl['visibility'].astype(np.int64)

            target_smpl_sem, target_vertices = self.get_smpl_semantic_region(target_smpl, left_padding, right_padding)

            semantic_path = vis_semantic_path[index]
            semantic_path = semantic_path # os.path.join(self.opt.dataroot, semantic_path)
            parsing = cv2.imread(semantic_path)[..., 0:1]
            parsing = np.array(parsing[:, :, 0], dtype=np.uint8)
            parsing = np.pad(parsing,((0,0),(left_padding, right_padding)),'constant',constant_values=(0,0))

            fg_parsing_path = semantic_path.replace('512_320', '256_192').replace('parsing', 'fg_parsing/parsing').replace('_label.png', '.png')
            fg_parsing = cv2.imread(fg_parsing_path)[..., 0:1]
            fg_parsing = np.array(fg_parsing[:, :, 0], dtype=np.uint8)
            fg_parsing = np.pad(fg_parsing,((0,0),(32, 32)),'constant',constant_values=(0,0))

            palm_mask = self.get_palm(keypoints, parsing, left_padding, right_padding)
            tar_densepose_path = image_path.replace('image', 'densepose').replace('.jpg', '.pkl')
            target_uv = self.get_densepose(tar_densepose_path, left_padding, right_padding)

            # ref_tensor = 0
            # key = image_path # image_path.replace('\\', '/').split('Deepfashion_{}/'.format(self.input_size))[-1]
            # val = self.ref_dict[key]
            # if len(val)-1 == 0:
            #     ref_name = val[0]    # val[random.randint(0,len(val)-1)]  
            # else:
            #     ref_name = val[1]   
            path_ref = vis_image_path[index-1] # os.path.join(self.opt.dataroot, ref_name)
            image_ref = Image.open(path_ref).convert('RGB')
            image_ref = ImageOps.expand(image_ref, border=(left_padding,0,right_padding,0), fill=(255,255,255))       # padding, left,top,right,bottom
            
            src_smpl_path = path_ref.replace("/image/", "/smpl/").replace(".jpg", ".pkl")
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

            label_ref_path = self.imgpath_to_labelpath(path_ref)
            label_ref_tensor, ref_keypoints, params = self.get_label_tensor(label_ref_path)
            transform_image = get_transform(self.opt, params)
            ref_tensor = transform_image(image_ref) 
            
            semantic_ref_path = path_ref.replace('image', 'parsing').replace('.jpg','_label.png')
            if 'cloth' in semantic_ref_path:
                parsing_ref = np.ones(shape=ref_tensor.shape[1:], dtype=np.uint8) * 5
            else:
                parsing_ref = cv2.imread(semantic_ref_path)[..., 0:1]
                parsing_ref = np.array(parsing_ref[:, :, 0], dtype=np.uint8)
                parsing_ref = np.pad(parsing_ref,((0,0),(left_padding, right_padding)),'constant',constant_values=(0,0))

            fg_parsing_ref_path = semantic_ref_path.replace('512_320', '256_192').replace('parsing', 'fg_parsing/parsing').replace('_label.png', '.png')
            fg_parsing_ref = cv2.imread(fg_parsing_ref_path)[..., 0:1]
            fg_parsing_ref = np.array(fg_parsing_ref[:, :, 0], dtype=np.uint8)
            fg_parsing_ref = np.pad(fg_parsing_ref,((0,0),(32, 32)),'constant',constant_values=(0,0))

            is_aug_flag = 0

            source_vertices = np.pad(source_vertices,((0,0),(left_padding, right_padding),(0,0)),'constant',constant_values=(6890,6890))
            target_sym_vertices = np.stack(target_sym_vertices, axis=2)
            target_sym_vertices = np.pad(target_sym_vertices,((0,0),(left_padding, right_padding),(0,0)),'constant',constant_values=(6890,6890))

            src_densepose_path = path_ref.replace('image', 'densepose').replace('.jpg', '.pkl')
            source_uv = self.get_densepose(src_densepose_path, left_padding, right_padding)

            batch_label_tensor.append(label_tensor)
            batch_image_tensor.append(image_tensor)
            batch_image_path.append(image_path)
            batch_is_aug_flag.append(is_aug_flag)
            batch_label_ref_tensor.append(label_ref_tensor)
            batch_ref_tensor.append(ref_tensor)
            batch_parsing.append(parsing)
            batch_fg_parsing.append(fg_parsing)
            batch_parsing_ref.append(parsing_ref)
            batch_fg_parsing_ref.append(fg_parsing_ref)
            batch_palm_mask.append(palm_mask)
            batch_source_vertices_img.append(source_smpl['pred_vertices_img'])
            batch_source_visibility.append(source_smpl['visibility'])
            # batch_source_smpl_semantic.append(source_smpl_sem)
            # batch_source_smpl_vertices.append(source_vertices)
            batch_source_densepsoe.append(source_uv)
            batch_target_vertices_img.append(target_smpl['pred_vertices_img'])
            batch_target_visibility.append(target_smpl['visibility'])
            # batch_target_smpl_semantic.append(target_smpl_sem)
            # batch_target_smpl_sym_vertices.append(target_sym_vertices)
            batch_target_densepose.append(target_uv)
            batch_symmetric_mask.append(visible_sym_region)
        
        input_dict = {'label': torch.stack(batch_label_tensor),
                    'image': torch.stack(batch_image_tensor),
                    'path': np.array(batch_image_path),
                    'is_aug': np.array(batch_is_aug_flag),
                    'ref_label': torch.stack(batch_label_ref_tensor),
                    'ref': torch.stack(batch_ref_tensor),
                    'parsing_array': torch.from_numpy(np.stack(batch_parsing)),
                    'fg_parsing': torch.from_numpy(np.stack(batch_fg_parsing)),
                    'parsing_ref_array': torch.from_numpy(np.stack(batch_parsing_ref)),
                    'fg_parsing_ref': torch.from_numpy(np.stack(batch_fg_parsing_ref)),
                    'palm_mask': torch.from_numpy(np.stack(batch_palm_mask)),
                    'source_vertices_img' : torch.from_numpy(np.stack(batch_source_vertices_img)),
                    'source_visibility' : torch.from_numpy(np.stack(batch_source_visibility)),
                    'source_densepose': torch.from_numpy(np.stack(batch_source_densepsoe)),
                    'target_vertices_img' : torch.from_numpy(np.stack(batch_target_vertices_img)),
                    'target_visibility' : torch.from_numpy(np.stack(batch_target_visibility)),
                    'target_densepose': torch.from_numpy(np.stack(batch_target_densepose)),
                    'symmetric_mask': torch.from_numpy(np.stack(batch_symmetric_mask))
                    }
        return input_dict