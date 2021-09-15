import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import timeit
import pandas as pd
from scipy.spatial.transform import Rotation as R
import glob
from src.loftr import LoFTR, default_cfg
from collections import OrderedDict

mint = np.array([[458.654,0,367.215],[0,457.296,248.375],[0,0,1]]) #remember to change it for different cam

def compute_homography(matched_kp1, matched_kp2):
    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_kp1,
                                    matched_kp2,
                                    cv2.RANSAC,
                                    confidence=0.999)
    inliers = inliers.flatten()
    return H, inliers

def find_rodrigues(delta_rotation):
    # angle error between 2 rotation matrices
    cos = (np.trace(delta_rotation) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return R_err

def find_r_t(Rs, Ts, finalkp1, finalkp2, gt_rotation, gt_t, dict1, ORB):   #normally there will be four sets of R,T returned by homography decomposition, find the correct one
    if len(Rs) == 1:     #this is for the case when only one set of R and T is returned
        rotation = Rs[0]
        translation = Ts[0].squeeze()
        delta_rotation = rotation.T @ gt_rotation
        R_err = find_rodrigues(delta_rotation)
        euler_error = R.from_matrix(delta_rotation).as_euler('zyx', degrees=True)
        delta_t = translation - gt_t
        if not ORB:
            dict1['t1_error(m)'].append(delta_t[0])
            dict1['t2_error(m)'].append(delta_t[1])
            dict1['t3_error(m)'].append(delta_t[2])
            dict1['EulerZ_error(degree)'].append(euler_error[0])
            dict1['EulerY_error(degree)'].append(euler_error[1])
            dict1['EulerX_error(degree)'].append(euler_error[2])
            dict1['delta_R_LoFTR'].append(R_err)
        else:
            dict1['delta_R_ORB'].append(R_err)
        return rotation, translation

    #first find the two sets of R,t that gives positive depth
    temp_j = []
    for j in range(len(Rs)):
        if len(temp_j) == 2:
            break
        left_projection = mint @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # world-coor
        right_projection = mint @ np.concatenate((Rs[j], Ts[j]), axis=1)
        triangulation = cv2.triangulatePoints(left_projection, right_projection, finalkp1[0],
                                              finalkp2[0])  # point in world-coor
        triangulation = triangulation / triangulation[3]  # make it homogeneous (x,y,z,1)
        if triangulation[2] > 0:  # z is positive
            point_in_cam2 = np.concatenate((Rs[j], Ts[j]), axis=1) @ triangulation  # change to cam2 coordinate
            if point_in_cam2[2] > 0:  # z is positive
                temp_j.append(j)
        else:
            continue

    #choose one that gives smaller error compared to ground truth
    delta_rotation1 = Rs[temp_j[0]].T @ gt_rotation
    delta_rotation2 = Rs[temp_j[1]].T @ gt_rotation
    R_err1 = find_rodrigues(delta_rotation1)
    R_err2 = find_rodrigues(delta_rotation2)
    if abs(R_err1) < abs(R_err2):
        rotation = Rs[temp_j[0]]
        translation = Ts[temp_j[0]].squeeze()
        delta_rotation = delta_rotation1
        R_err = R_err1
    else:
        rotation = Rs[temp_j[1]]
        translation = Ts[temp_j[1]].squeeze()
        delta_rotation = delta_rotation2
        R_err = R_err2
    euler_error = R.from_matrix(delta_rotation).as_euler('zyx', degrees=True)
    delta_t = translation - gt_t
    if not ORB:
        dict1['t1_error(m)'].append(delta_t[0])
        dict1['t2_error(m)'].append(delta_t[1])
        dict1['t3_error(m)'].append(delta_t[2])
        dict1['EulerZ_error(degree)'].append(euler_error[0])
        dict1['EulerY_error(degree)'].append(euler_error[1])
        dict1['EulerX_error(degree)'].append(euler_error[2])
        dict1['delta_R_LoFTR'].append(R_err)
    else:
        dict1['delta_R_ORB'].append(R_err)
    return rotation, translation

def main(args):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    _default_cfg['match_coarse']['thr'] = args.match_threshold
    matcher = LoFTR(config=_default_cfg)

    #load model
    if args.model == 'indoor':
        model_path = "weights/indoor_ds_new.ckpt"
    elif args.model == 'outdoor':
        model_path = "weights/outdoor_ds.ckpt"
    else:
        raise NotImplementedError("model can only be either indoor or outdoor")
    matcher.load_state_dict(torch.load(model_path)['state_dict'])
    matcher = matcher.eval().cuda()

    #check if image path exist
    if not os.path.isdir(args.images):
        raise NotImplementedError("image path not exist")
    #extracting keypoints
    img_pth = args.images
    imgs = sorted(glob.glob(img_pth + '*'))
    imgs = imgs[0:750]
    #create a dict to record performance
    dict1 = {'match_id': [], 'detected_matches': [], 'valid_matches(LoFRT)': [], 'valid_matches(ORB)':[], 'inlier_rate': [],
             'detection_time_LoFTR': [], 'detection_time_ORB':[], 'EulerZ_error(degree)': [], 'EulerY_error(degree)': [],
             'EulerX_error(degree)': [], 't1_error(m)': [], 't2_error(m)': [], 't3_error(m)': [], 'delta_R_LoFTR':[], 'delta_R_ORB':[]}

    print(f"\nExtracting features for {img_pth}")
    # input ground truth dataframe path here
    if args.gtcsv != '':
        gt_dataframe = pd.read_csv(args.gtcsv, usecols=[i for i in range(1, 8)], nrows=750)

    feature_dict = OrderedDict()          #this is to save extracted feature map temporarily
    frame_interval = args.frame_interval
    for i in range(0, len(imgs)-frame_interval, frame_interval):
        # time the detection
        img0_raw = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.imread(imgs[i+frame_interval], cv2.IMREAD_GRAYSCALE)
        img0_raw = cv2.resize(img0_raw, (640, 480))
        img1_raw = cv2.resize(img1_raw, (640, 480))
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

        start = timeit.default_timer()
        with torch.no_grad():
            if i in feature_dict:                                         #the first image already been input into LoFTR before
                flag = True
                batch = {'image0': img0, 'image1': img1}
                feat1_c, feat1_f = matcher.extract_feature(batch, flag)   #only extract feature map for the second image
                feature_dict[i+frame_interval] = [feat1_c, feat1_f]       #store feature map to dictionary
                feat0_c, feat0_f = feature_dict.pop(i)                    #get the feature map for the first image from dictionary
            else:                                                         #none of the images been input before
                flag = False
                batch = {'image0': img0, 'image1': img1}
                feat0_c, feat1_c, feat0_f, feat1_f = matcher.extract_feature(batch, flag)   #extract feature map for both images
                feature_dict[i] = [feat0_c, feat0_f]                                        #store feature map to dictionary
                feature_dict[i+frame_interval] = [feat1_c, feat1_f]

            if len(feature_dict) > 10:
                feature_dict.popitem(last=False)          #preventing dictionary from building up
            matcher.transformer(feat0_c, feat1_c, feat0_f, feat1_f, batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            #mconf = batch['mconf'].cpu().numpy()
            if len(mkpts0)>1000:
                skip = max(1,int(len(mkpts0)/1000))  #if too many matches, you want to skip some
                mkpts0 = mkpts0[::skip]
                mkpts1 = mkpts1[::skip]
        stop = timeit.default_timer()
        dict1['detection_time_LoFTR'].append(stop - start)
        assert len(mkpts0) == len(mkpts1), f'mkpts0: {len(mkpts0)} v.s. mkpts1: {len(mkpts1)}'

        # get ground truth
        if args.gtcsv != '':
            t1 = gt_dataframe.iloc[i, 0:3].to_numpy()
            t2 = gt_dataframe.iloc[i + frame_interval, 0:3].to_numpy()
            gt_t = t2 - t1
            quaternion1 = gt_dataframe.iloc[i, 3:8].to_numpy()
            quaternion1_scaler_last = np.array([quaternion1[1], quaternion1[2], quaternion1[3], quaternion1[0]])
            rotation1 = R.from_quat(quaternion1_scaler_last).as_matrix()
            quaternion2 = gt_dataframe.iloc[i + frame_interval, 3:8].to_numpy()
            quaternion2_scaler_last = np.array([quaternion2[1], quaternion2[2], quaternion2[3], quaternion2[0]])
            rotation2 = R.from_quat(quaternion2_scaler_last).as_matrix()
            gt_rotation = rotation1 @ rotation2.T

        dict1['detected_matches'].append(len(mkpts0))
        H, inliers = compute_homography(mkpts0, mkpts1)
        valid_match = np.sum(inliers)
        dict1['valid_matches(LoFRT)'].append(valid_match)
        dict1['inlier_rate'].append(valid_match / len(mkpts0))
        print('**************************************')  ##1
        print(f"Total matches for match{i}.jpg is {valid_match}, outlier rate is {1 - valid_match / len(mkpts0)}")  ##1
        finalkp1 = mkpts0[inliers.astype(bool)]
        finalkp2 = mkpts1[inliers.astype(bool)]
        # print(finalkp1.shape)##1

        # record error
        if args.gtcsv != '':
            # get correct pose, you will have four sets of solution, only one with both positive z value for two cameras
            num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, mint)
            rotation, translation = find_r_t(Rs, Ts, finalkp1, finalkp2, gt_rotation, gt_t, dict1, ORB=False)

            #you may want to compare to ORB, here is the code
            # Initiate ORB detector
            orb = cv2.ORB_create()
            start = timeit.default_timer()
            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img0_raw, None)
            kp2, des2 = orb.detectAndCompute(img1_raw, None)
            # create BFMatcher object
            stop = timeit.default_timer()
            dict1['detection_time_ORB'].append(stop-start)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1, des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:1000]           #allow at most 1000 matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            H2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            valid_match_orb = np.sum(mask)
            dict1['valid_matches(ORB)'].append(valid_match_orb)
            num1, Rs1, Ts1, Ns1 = cv2.decomposeHomographyMat(H2, mint)
            mask = mask.squeeze()
            rotation_orb, translation_orb = find_r_t(Rs1, Ts1, src_pts[mask.astype(bool)], dst_pts[mask.astype(bool)], gt_rotation, gt_t, dict1, ORB=True)



        # plot, only for LoFTR, not plotting ORB
        # initialize the output visualization image
        (hA, wA) = img0_raw.shape[:2]  # cv2.imread returns (h,w,c)
        (hB, wB) = img1_raw.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = cv2.resize(cv2.imread(imgs[i]), (640,480))
        vis[0:hB, wA:] = cv2.resize(cv2.imread(imgs[i+frame_interval]),(640,480))

        # loop over the matches
        for p1, p2 in zip(finalkp1, finalkp2):
            # draw the match
            ptA = (int(p1[0]), int(p1[1]))
            ptB = (int(p2[0]) + wA, int(p2[1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
            cv2.circle(vis, ptA, 3, color=(0, 0, 255))
            cv2.circle(vis, ptB, 3, color=(0, 0, 255))
        cv2.imwrite('./output/separate/' + 'match' + str(i) + '.jpg', vis)
        print('outputting matching' + str(i))
        temp_str = 'match' + str(i)
        dict1['match_id'].append(temp_str)
        # save csv to dict
    if args.gtcsv != '':
        df1 = pd.DataFrame.from_dict(dict1)
        df1.to_csv('./output/separate/evaluation.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--model", type=str, required=True, help='model type, either indoor or outdoor')
    parser.add_argument("--gtcsv", type=str, default='', help='path to ground truth csv file')
    parser.add_argument("--images", type=str, required=True, help='image directory')
    parser.add_argument("--match_threshold", type=float, default = 0.2, help='coarse match threshold')
    parser.add_argument("--frame_interval", type=int, default = 5, help='frames to skip between matches')
    args = parser.parse_args()

    main(args)