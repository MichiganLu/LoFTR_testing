import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import glob
from src.loftr import LoFTR, default_cfg, Pruned_Backbone
import pandas as pd


def check_validity(mkpts0,mkpts1,H_scaled):
    total_distance = 0
    valid_distance = 0
    mask = np.zeros(len(mkpts0))
    for j in range(len(mkpts0)):
        original_coor1 = mkpts0[j]
        original_coor2 = mkpts1[j]
        homogeneous_coor = np.array([original_coor1[0], original_coor1[1], 1])
        transformed_coor = H_scaled @ homogeneous_coor
        transformed_coor[0,0] = transformed_coor[0,0] / transformed_coor[0,2]
        transformed_coor[0,1] = transformed_coor[0,1] / transformed_coor[0,2]
        result = np.array([transformed_coor[0,0], transformed_coor[0,1]])
        distance = np.linalg.norm(result - original_coor2, ord=1)  #L1 norm between two points
        total_distance = total_distance+distance
        if distance < 8:  # check if point lies close to ground truth region
            mask[j] = 1
            valid_distance = valid_distance+distance
    total_distance = total_distance/len(mkpts0)    #this is the average value
    if np.sum(mask) == 0:
        valid_distance = None
    else:
        valid_distance = valid_distance/np.sum(mask)   #this is the average value
    return mask, total_distance, valid_distance


def draw(shape,mkpts0,mkpts1,mask,img0,img1):
    # plot for LoFTR
    # initialize the output visualization image
    (hA, wA) = (shape[1],shape[0])
    (hB, wB) = (shape[1],shape[0])
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = cv2.resize(cv2.imread(img0), shape)
    vis[0:hB, wA:] = cv2.resize(cv2.imread(img1), shape)
    if len(mkpts0) != 0:
        # loop over the valid matches
        for p1, p2 in zip(mkpts0[mask.astype(bool)], mkpts1[mask.astype(bool)]):
            # draw the valid match, blue circle green line
            ptA = (int(p1[0]), int(p1[1]))
            ptB = (int(p2[0]) + wA, int(p2[1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
            cv2.circle(vis, ptA, 3, color=(255, 0, 0))
            cv2.circle(vis, ptB, 3, color=(255, 0, 0))
        # loop over invalid matches
        for p1, p2 in zip(mkpts0[~mask.astype(bool)], mkpts1[~mask.astype(bool)]):
            # draw the invalid match, yellow circle, red line
            ptA = (int(p1[0]), int(p1[1]))
            ptB = (int(p2[0]) + wA, int(p2[1]))
            cv2.line(vis, ptA, ptB, (0, 0, 255), 1)
            cv2.circle(vis, ptA, 3, color=(255, 255, 0))
            cv2.circle(vis, ptB, 3, color=(255, 255, 0))
    return vis


def main(args):
    if not os.path.isdir('./output/hpatches'):
        os.makedirs('./output/hpatches')
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    _default_cfg['match_coarse']['thr'] = args.match_threshold
    matcher_indoor = LoFTR(config=_default_cfg)
    matcher_outdoor = LoFTR(config=_default_cfg)
    shape = (640,480)

    #load model
    if os.path.isfile("weights/indoor_ds_new.ckpt"):
        model_path1 = "weights/indoor_ds_new.ckpt"
    else:
        raise NotImplementedError("cannot find indoor model")
    if os.path.isfile("weights/outdoor_ds.ckpt"):
        model_path2 = "weights/outdoor_ds.ckpt"
    else:
        raise NotImplementedError("cannot find outdoor model")
    matcher_indoor.load_state_dict(torch.load(model_path1)['state_dict'])
    matcher_outdoor.load_state_dict(torch.load(model_path2)['state_dict'])
    #new_backbone = Pruned_Backbone()
    #matcher.backbone = new_backbone
    matcher_indoor = matcher_indoor.eval().cuda()
    matcher_outdoor = matcher_outdoor.eval().cuda()

    #check if image path exist
    if not os.path.isdir(args.images):
        raise NotImplementedError("image path not exist")
    #create dictionary to record data
    dict1 = {'name':[], 'model':[], 'total_match_loftr':[], 'valid_match_loftr':[], 'avg_distance_loftr':[], 'avg_valid_distance_loftr':[],
             'total_match_SIFT':[], 'valid_match_SIFT':[], 'avg_distance_SIFT':[], 'avg_valid_distance_SIFT':[],
             'total_match_ORB':[], 'valid_match_ORB':[], 'avg_distance_ORB':[], 'avg_valid_distance_ORB':[]}

    #extracting keypoints
    img_pth = args.images
    sub_category = os.listdir(img_pth)
    sub_path = []
    for one in sub_category:
        sub_path.append(os.path.join(img_pth,one))
    sub_path = sorted(sub_path)      #sub-categories for hpathes
    #processing it directory by directory
    for directory in sub_path:
        imgs = sorted(glob.glob(directory + '/*.ppm'))
        text = sorted(glob.glob(directory + '/H_*'))
        for i in range(len(imgs)-1):
            dict1['name'].append(directory.split('/')[-1]+'1_'+str(i+2))
            img0_raw = cv2.imread(imgs[0], cv2.IMREAD_GRAYSCALE)
            img1_raw = cv2.imread(imgs[i + 1], cv2.IMREAD_GRAYSCALE)
            scale_x_A = shape[0]/(img0_raw.shape)[1]    #remember image axis is different from matrix axis. This is used to calculate the new GT homography matrix after scaling
            scale_y_A = shape[1]/(img0_raw.shape)[0]
            scale_x_B = shape[0]/(img1_raw.shape)[1]    #remember image axis is different from matrix axis. This is used to calculate the new GT homography matrix after scaling
            scale_y_B = shape[1]/(img1_raw.shape)[0]
            img0_raw = cv2.resize(img0_raw, shape)
            img1_raw = cv2.resize(img1_raw, shape)
            img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
            img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

            #for loftr
            #doing inference
            with torch.no_grad():
                batch_indoor = {'image0': img0, 'image1': img1}
                batch_outdoor = {'image0': img0, 'image1': img1}
                matcher_indoor(batch_indoor)
                matcher_outdoor(batch_outdoor)
                if len(batch_indoor['mkpts0_f']) > len(batch_outdoor['mkpts0_f']) or dict1['name'][-1].startswith('v_'):
                    mkpts0 = batch_indoor['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch_indoor['mkpts1_f'].cpu().numpy()
                    dict1['model'].append('indoor')
                else:
                    mkpts0 = batch_outdoor['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch_outdoor['mkpts1_f'].cpu().numpy()
                    dict1['model'].append('outdoor')
            if len(mkpts0) != 0:
                dict1['total_match_loftr'].append(len(mkpts0))
                #doing evaluation
                with open(text[i],'r') as f:
                    H = f.read()
                    H = H.replace('\n',';',2)
                    H = np.matrix(H)           #this is the ground truth homography matrix
                #since you have reshape the image, you need to transform the homography
                S_A, S_B = np.eye(3), np.eye(3)
                S_A[0][0] = scale_x_A
                S_A[1][1] = scale_y_A
                S_B[0][0] = scale_x_B
                S_B[1][1] = scale_y_B
                H_scaled = S_B @ H @ np.linalg.inv(S_A)
                mask_loftr, total_distance_loftr, valid_distance_loftr = check_validity(mkpts0,mkpts1,H_scaled)
                dict1['valid_match_loftr'].append(np.sum(mask_loftr))
                dict1['avg_distance_loftr'].append(total_distance_loftr)
                dict1['avg_valid_distance_loftr'].append(valid_distance_loftr)
                vis_loftr = draw(shape,mkpts0,mkpts1,mask_loftr,imgs[0],imgs[i+1])
                #plot
                cv2.imwrite('./output/hpatches/' + dict1['name'][-1] + '_loftr' + '.jpg', vis_loftr)
            else:
                dict1['total_match_loftr'].append(0)
                dict1['valid_match_loftr'].append(0)
                dict1['avg_distance_loftr'].append(None)
                dict1['avg_valid_distance_loftr'].append(None)
                vis_loftr = draw(shape,mkpts0,mkpts1,mask_loftr,imgs[0],imgs[i+1])
                #plot images without any matches
                cv2.imwrite('./output/hpatches/' + dict1['name'][-1] + '_loftr' + '.jpg', vis_loftr)


            #for ORB
            orb = cv2.ORB_create()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img0_raw, None)
            kp2, des2 = orb.detectAndCompute(img1_raw, None)
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            dict1['total_match_ORB'].append(len(matches))
            mask_ORB, total_distance_ORB, valid_distance_ORB = check_validity(src_pts, dst_pts, H_scaled)
            dict1['valid_match_ORB'].append(np.sum(mask_ORB))
            dict1['avg_distance_ORB'].append(total_distance_ORB)
            dict1['avg_valid_distance_ORB'].append(valid_distance_ORB)
            vis_ORB = draw(shape,src_pts,dst_pts,mask_ORB,imgs[0],imgs[i+1])
            #plot
            cv2.imwrite('./output/hpatches/' + dict1['name'][-1] + '_ORB' + '.jpg', vis_ORB)

            #for SIFT
            sift = cv2.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1_sift, des1_sift = sift.detectAndCompute(img0_raw, None)
            kp2_sift, des2_sift = sift.detectAndCompute(img1_raw, None)
            # Match descriptors.
            bf = cv2.BFMatcher(crossCheck=True)
            matches_sift = bf.match(des1_sift, des2_sift)
            # Sort them in the order of their distance.
            matches_sift = sorted(matches_sift, key=lambda x: x.distance)
            src_pts_sift = np.float32([kp1_sift[m.queryIdx].pt for m in matches_sift]).reshape(-1, 2)
            dst_pts_sift = np.float32([kp2_sift[m.trainIdx].pt for m in matches_sift]).reshape(-1, 2)
            dict1['total_match_SIFT'].append(len(matches_sift))
            mask_SIFT, total_distance_SIFT, valid_distance_SIFT = check_validity(src_pts_sift, dst_pts_sift, H_scaled)
            dict1['valid_match_SIFT'].append(np.sum(mask_SIFT))
            dict1['avg_distance_SIFT'].append(total_distance_SIFT)
            dict1['avg_valid_distance_SIFT'].append(valid_distance_SIFT)
            vis_SIFT = draw(shape, src_pts_sift, dst_pts_sift, mask_SIFT, imgs[0], imgs[i + 1])
            # plot
            cv2.imwrite('./output/hpatches/' + dict1['name'][-1] + '_SIFT' + '.jpg', vis_SIFT)

            print('outputting matching ' + dict1['name'][-1])

    df1 = pd.DataFrame.from_dict(dict1)
    df1.to_csv('./output/hpatches/eval_0.2_2.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--images", type=str, required=True, help='image directory')
    parser.add_argument("--match_threshold", type=float, default = 0.25, help='coarse match threshold')
    args = parser.parse_args()

    main(args)