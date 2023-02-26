import torch
import numpy as np
import cv2
import json



kptcolors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0]]

limbseq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]


def get_custom_pose(densepose_path, kpt_path, visibility, left_padding, right_padding, h=512, w=320):
    # with open(densepose_path, "rb") as f:
    #     densepose = pickle.load(f)
    # x0, y0, x1, y1 = densepose['pred_boxes_XYXY']
    # label_uv = densepose['pred_densepose_label'][np.newaxis,...].astype(np.float32) * 10
    # # uv = densepose['pred_densepose_uv'].astype(np.float32)
    # # label_uv = np.concatenate([label, uv], axis=0)
    # uv_h, uv_w = label_uv.shape[-2], label_uv.shape[-1]
    # pad_w = int(x0), w - uv_w - int(x0) 
    # pad_h = int(y0), h - uv_h - int(y0) 
    # pad_uv = np.pad(label_uv, ((0,0), (pad_h[0], pad_h[1]), (pad_w[0], pad_w[1])), 'constant', constant_values=(0,0))
    # pad_uv = np.pad(pad_uv,((0,0), (0,0), (left_padding, right_padding)),'constant',constant_values=(0,0))

    weight_map = generate_weight_map(size=512)
    denorm_pose = denorm_weight_map(kpt_path, weight_map, 0)
    denorm_pose = np.transpose(denorm_pose, (2,0,1))

    output_pose = denorm_pose.astype(np.float32) # np.concatenate([pad_uv, denorm_pose], axis=0)
    return output_pose


def denorm_weight_map(kpt_path, weight_map, box_factor):
    with open(kpt_path, 'r') as f:
        keypoints_data = json.load(f)
    if len(keypoints_data['people']) == 0:
        keypoints = np.zeros((18,3))
    else:
        keypoints = np.array(keypoints_data['people'][0]['pose_keypoints_2d']).reshape(-1,3)

    h, w = weight_map.shape[:2]
    o_h, o_w = 512, 512 # h, w
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

    denorm_patches = list()
    M_invs = list()

    # f = os.path.join(img_path)
    # image = np.array(PIL.Image.open(f))

    # denorm_img = np.zeros_like(img)
    for ii, bpart in enumerate(bparts):
        M, M_inv = get_crop(keypoints, bpart, order, wh, o_w, o_h, ar)
        if M_inv is not None:
            denorm_patch = cv2.warpPerspective(weight_map, M_inv, (o_w,o_h), borderMode=cv2.BORDER_CONSTANT)
        else:
            denorm_patch = np.zeros((h, w, 3)).astype(np.uint8)
        denorm_patches.append(denorm_patch)

        # vis_patch = (denorm_patch - np.min(denorm_patch)) / (np.max(denorm_patch) - np.min(denorm_patch))
        # vis_weight = (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))
        # plt.imshow(vis_weight)
        # plt.savefig('./debug.png')
        # plt.imshow(vis_patch)
        # plt.savefig('./debug.png')
    denorm_pose = np.concatenate(denorm_patches, axis=2)
    # M_invs = np.concatenate(M_invs, axis=0)

    return denorm_pose


def generate_weight_map(size=512):
    yy, xx = np.meshgrid(np.arange(0, size), np.arange(0, size))

    p = size // 8
    center = size // 2
    a, b, c = (center+p, center+p), (center-p, center-p), (center+p, center-p)
    i_map = (-(xx - b[0]) * (c[1] - b[1]) + (yy - b[1]) * (c[0] - b[0])) / \
            (-(a[0] - b[0]) * (c[1] - b[1]) + (a[1] - b[1]) * (c[0] - b[0]))
    j_map = (-(xx - c[0]) * (a[1] - c[1]) + (yy - c[1]) * (a[0] - c[0])) / \
            (-(b[0] - c[0]) * (a[1] - c[1]) + (b[1] - c[1]) * (a[0] - c[0]))
    k_map = 1 - i_map - j_map

    mask = np.zeros(shape=(size, size), dtype=np.uint8)
    mask[int(0.4375*size):int(0.5625*size), int(0.4375*size):int(0.5625*size)] = 1

    i_map = mask * i_map
    j_map = mask * j_map
    k_map = mask * k_map

    # plt.subplot(1,3,1)
    # plt.imshow(k_map)
    # plt.subplot(1,3,2)
    # plt.imshow(k_map)
    # plt.subplot(1,3,3)
    # plt.imshow(k_map)
    # plt.savefig('./debug.png')

    return np.stack([i_map, j_map, k_map], axis=2)
    

def draw_pose_from_cords(pose_joints, img_size, affine_matrix=None,
                            coeffs=None, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    # mask = np.zeros(shape=img_size, dtype=np.uint8)
    if draw_joints:
        for i, p in enumerate(limbseq):
            f, t = p[0]-1, p[1]-1
            from_missing = pose_joints[f][2] < 0.1
            to_missing = pose_joints[t][2] < 0.1
            if from_missing or to_missing:
                continue
            if not affine_matrix is None:
                pf = np.dot(affine_matrix, np.matrix([pose_joints[f][0], pose_joints[f][1], 1]).reshape(3, 1))
                pt = np.dot(affine_matrix, np.matrix([pose_joints[t][0], pose_joints[t][1], 1]).reshape(3, 1))
            else:
                pf = pose_joints[f][0], pose_joints[f][1]
                pt = pose_joints[t][0], pose_joints[t][1]
            fx, fy = pf[1], pf[0]# max(pf[1], 0), max(pf[0], 0)
            tx, ty = pt[1], pt[0]# max(pt[1], 0), max(pt[0], 0)
            fx, fy = int(fx), int(fy)# int(min(fx, 255)), int(min(fy, 191))
            tx, ty = int(tx), int(ty)# int(min(tx, 255)), int(min(ty, 191))
            # xx, yy, val = line_aa(fx, fy, tx, ty)
            # colors[xx, yy] = np.expand_dims(val, 1) * kptcolors[i] # 255
            cv2.line(colors, (fy, fx), (ty, tx), kptcolors[i], 2)
            # mask[xx, yy] = 255

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][2] < 0.1:
            continue
        if not affine_matrix is None:
            pj = np.dot(affine_matrix, np.matrix([joint[0], joint[1], 1]).reshape(3, 1))
        else:
            pj = joint[0], joint[1]
        x, y = int(pj[1]), int(pj[0])# int(min(pj[1], 255)), int(min(pj[0], 191))
        xx, yy = circle(x, y, radius=radius, shape=img_size)
        colors[xx, yy] = kptcolors[i]
        # mask[xx, yy] = 255

    # colors = colors * 1./255
    # mask = mask * 1./255
    
    return colors


def valid_joints(joint):
    return (joint >= 0.1).all()

def get_crop(keypoints, bpart, order, wh, o_w, o_h, ar = 1.0):
    joints = keypoints
    bpart_indices = [order.index(b) for b in bpart]
    part_src = np.float32(joints[bpart_indices][:, :2])
    # fall backs
    if not valid_joints(joints[bpart_indices][:, 2]):
        if bpart[0] == "lhip" and bpart[1] == "lknee":      # 有hip关键点但是没有knee关键点
            bpart = ["lhip"]
            bpart_indices = [order.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices][:,:2])
        elif bpart[0] == "rhip" and bpart[1] == "rknee":    #　左边同理
            bpart = ["rhip"]
            bpart_indices = [order.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices][:,:2])
        elif bpart[0] == "lshoulder" and bpart[1] == "rshoulder" and bpart[2] == "cnose": # 没有左肩右肩,鼻子这组区域
            bpart = ["lshoulder", "rshoulder", "rshoulder"]
            bpart_indices = [order.index(b) for b in bpart]
            part_src = np.float32(joints[bpart_indices][:,:2])

    if not valid_joints(joints[bpart_indices][:, 2]):
            return None, None
    part_src[:, 0] = part_src[:, 0] + 96                    # correct axis by adding pad size 

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

    # dst = np.float32([[0.0,0.0],[0.0,1.0],[1.0,1.0],[1.0,0.0]])
    dst = np.float32([[0.4375,0.4375],[0.4375,0.5625],[0.5625,0.5625],[0.5625,0.4375]])
    part_dst = np.float32(wh * dst)

    # M = cv2.getPerspectiveTransform(part_src, part_dst)
    M_inv = cv2.getPerspectiveTransform(part_dst,part_src)
    
    return None, M_inv
