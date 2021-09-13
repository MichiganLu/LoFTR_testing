from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid

from .geometry import warp_kpts

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):      #actually data is the input batch that you pass to the loftr network
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['LOFTR']['RESOLUTION'][0]                                                    #config['LOFTR']['RESOLUTION'] is (8,2), so scale is 8
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])                                #from image dimension to 1/8 ResNet feature map dimension

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    #[N, h0w0, 2], meshgrid of img0 in 1/8 dimension. Here the N correspond to batch number!!! The 2 cooresponds to (x,y)
                                                                                                #create_meshgrid(h0, w0, False, device) gives you dim [1, h0, w0 2], reshape gives you [1, h0w0, 2], repeat gives you [N, h0w0, 2]
    grid_pt0_i = scale0 * grid_pt0_c                                                            #scale (x,y), grid size not changing. So meshgrid corresponds to center of receptive field in original img0 from 1/8 feature map
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    #[N, h1w1, 2], meshgrid of img1 in 1/8 dimension.
    grid_pt1_i = scale1 * grid_pt1_c                                                            #scale (x,y), grid size not changing. So meshgrid corresponds to center of receptive field in original img1 from 1/8 feature map

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])    #warped keypoints still of dimension [N, h0w0, 2], warped position is wrt to original img
    _, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])    #warped keypoints still of dimension [N, h1w1, 2], warped position is wrt to original img
    w_pt0_c = w_pt0_i / scale1                                         #first warp to image1, then divide by img1 scale becomes img1 1/8 feature map coorespondence
    w_pt1_c = w_pt1_i / scale0                                         #first warp to image0, then divide by img0 scale becomes img0 1/8 feature map coorespondence

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()                        #round the warpped feature map, still of dimension [N, h0w0, 2], remember this is already feature map coorespondence
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1    #nearest_index1 is the coorespondence from img0 to img1. #It is actually from warpped grid 0
                                                                           #x + y*w1. nearest_index1 is of dimension [N, h0w0]. Actually, x + y*w1 is just turning 2D matrix to 1D.
                                                                           # for example a 2d matrix [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]], which you create from line 49, becomes [0,1,2,3,4,5]. Each index cooresponds to 1 position.
                                                                           # here the width is 3, you can do the addition x+y*3 to check for yourself.
                                                                           #I know it does not make sense the width is 3, but they seem to flip the row and column
                                                                           #this is how they arrange the grid
                                                                           #[[0,0],[1,0],[2,0]
                                                                           # [0,1],[1,1],[2,1]] The first number coorespond to column, second number coorespond to row

    w_pt1_c_round = w_pt1_c[:, :, :].round().long()                        #round the warpped feature map, still of dimension [N, h1w1, 2], remember this is already feature map coorspondence
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0    #x + y*w0. nearest_index0 is of dimension [N, h1w1]
                                                                           #nearest_index0 is from warped grid 1
    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)   #this plus sign here is logic "and", i.e. true+false=false etc. this mask is used to mask out of bound grids
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0                                #if out of bound, make index 0. note that you are not masking nearest index matrix, you are masking grid matrix
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)     #_b is the batch num, _i is the index, of dimension [h1w1,], which is x+y*w. We ignore the batch at this moment.
                                                                                                         #think about it this way, nearest_index1 map img0 to img1, nearest_index0 map img1 to img0.
                                                                                                         #so what you are doing is basically map img0 to img1, then map img0 back to 0.
                                                                                                         #ideally, if nearest neighbor, the map should be bidirectional, so 0 to 1 and back to 0 gives you [0,1,2,3,4...]
                                                                                                         #if you don't understand, just check this code
                                                                                                         #f = torch.tensor([0, 3, 1, 2, 4])
                                                                                                         #g = torch.tensor([0, 2, 3, 1, 4])
                                                                                                         #f and g are bidirectional map(nearest neighbor map) f[g] gives you [0,1,2,3,4]

    correct_0to1 = loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1)                    #the correct 0 to 1 actually means correct 1 to 0 to 1, which is ideally [0,1,2,3,4...]
                                                                                                         #so loop_back == torch.arange(h0*w0, device=device)[None].repeat(N, 1) gives you mask of the mutual nearest neighbor, a.k.a. bidirectional map
    correct_0to1[:, 0] = False                                                                           #ignore the top-left corner, again correct_0to1 is a mask!!!
                                                                                                         #this mask is used to mask nearest_index1, which is from warped img0 grid.
                                                                                                         #nearest_index1[correct_0to1] gives img1_grid pixel coorespondence to img0.
                                                                                                         #hard to understand? for example. let's say torch.where(correct_0to1 != 0) (which are True positions in mask) gives you [0,3,4].
                                                                                                         #nearest_index1[correct_0to1] gives you [0,1,4].
                                                                                                         #then [0,3,4] pixels of img0 coorespond to [0,1,4] pixels of img1.
                                                                                                         #Specifically, 0th pixel of img0 coorespond to 0th pixel of img1;; 3rd px of img0 to 1st px of img1;; 4th px of img0 to 4 px of img1.

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)                 #conf_matrix_gt is the matching matrix, if i row match to j column(h0w0 match to h1w1), then (i,j) = 1.
    b_ids, i_ids = torch.where(correct_0to1 != 0)                                #b_ids is the x position that satisfies the where condition, which is the batch num. i_ids is the y position that satisfies the condition. i_ids is from img0 to img1
    j_ids = nearest_index1[b_ids, i_ids]                                         #j_ids is from img1 to img0
                                                                                 #think of it this way, i_ids of img0 coorespond to j_ids of img1
                                                                                 #!!!!!!!!!!lets say i_ids=[0,3,4], j_ids=[0,1,4]
                                                                                 #!!!!!!!!!!it means 0th pixel of img0 coorespond to 0th pixel of img1;; 3rd px of img0 to 1st px of img1;; 4th px of img0 to 4 px of img1.

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1                                      #make ground truth matches 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })


def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

@torch.no_grad()
def spvs_fine(data, config):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. misc
    # w_pt0_i, pt1_i = data.pop('spv_w_pt0_i'), data.pop('spv_pt1_i')
    w_pt0_i, pt1_i = data['spv_w_pt0_i'], data['spv_pt1_i']    #data['w_pt0_i'] is of dim [N, h0w0, 2], data['spv_pt1_i'] is of dim [N, h1w1, 2].
                                                               #data['w_pt0_i'] is warped coorespondence of original image from img0 to img1, remember it is from the process of 1.creating a grid of 1/8 feature map size, scale to original img((x,y)*8), warp to img1.
                                                               #data['spv_pt1_i'] is not warped img1_grid*scale. Each grid pixel cooresponds to the center pixel location of receptive field in original img1.
    scale = config['LOFTR']['RESOLUTION'][1]                   #config['LOFTR']['RESOLUTION'] is (8,2), so scale is 2
    radius = config['LOFTR']['FINE_WINDOW_SIZE'] // 2          #fine window size is 5, so radius is 2

    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']     #the picked coarse matches prediction, not the ground truth!! from i_ids to j_ids meaning from i pixel of img0 to j pixel of img1.

    # 3. compute gt
    scale = scale * data['scale1'][b_ids] if 'scale0' in data else scale                    #from what I see, batch(which is a dict) does not have a key call scale, so you treat scale as 2.
    # `expec_f_gt` might exceed the window, i.e. abs(*) > 1, which would be filtered later
    expec_f_gt = (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius             #[M, 2], w_pt0_i[b_ids, i_ids] gives the img0's matching in img1 (the matchings are positions not indexes) (wrt to original image).
                                                                                            #pt1_i[b_ids, j_ids] returns position for cooresponding j_ids. Basically it is "given an index return its position" kind of situation.
                                                                                            #for example, pt1_i[0, 1] is (8,0), pt1_i[0, 2] is (16,0), pt1_i[0,3] is (24,0). pt1_i is unwarped, just the scaled original grid.
                                                                                            #you may wonder why (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids])? The point is to create a numerical mask,
                                                                                            #in which elements smaller than threshould will be True and elements larger than threshold will be false
                                                                                            #this makes sense because if the predictions are accurate, then all elements will be close to (0,0)
                                                                                            #understand it this way. (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale / radius < 1 are filtered out as false
                                                                                            #which is equivalent to (w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]) / scale < radius. It means you want to filter out points which are projected out of w*w window(wrt to 1/2 feat map)

                                                                                            #why it is the ground truth? because you directly apply the index on the warped grid(the ground truth grid), of course it will return the ground truth matching position. This is the ground truth for your picked position
                                                                                            #w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids] is possibly the distance from the nearest neighbor, divided by the scale is to scale to first feature map
    data.update({"expec_f_gt": expec_f_gt})


def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
