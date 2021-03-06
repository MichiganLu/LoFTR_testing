import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching

class Pruned_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.load('./temp_backbone/seventh_prune.pth')

    def forward(self, data):
        return self.backbone(data)

class LoFTR(nn.Module):
    #backbone = Pruned_Backbone().cuda()
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)     #original backbone
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching()

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W) remember to turn rgb to gray
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))               #feed into backbone get feature map for img0 and img1, concatenate in batch dimension. backbone returns two feature map, one for 1/2 and one for 1/8 feature map
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])      #split feature map into two for two images
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],                                            #always dimension is [batch, channel, height, width], so here you are only taking heght, weight of feature map
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]                                             #the height and width is feature map dimension
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')                               #add space encoding to coarse feature map of img0, and reshape to [N, HW, C]!!!
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')                               #add space encoding to coarse feature map of img1, and reshape to [N, HW, C]!!!

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)                              #feeding into coarse transformer

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)                        #remember feat_c0, feat_c1 is of dimension [N, HW, C]

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):                                  #this basically revise the key of the state dict and pass the right state dict to the parent's method
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)                          #super here use nn.module's method load_state_dict

    def extract_feature(self, data, flag):
        #self.backbone = self.__class__.backbone
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if flag:                                                                                                   #if the first image has been input before, ignore the first image
            (feat_c1, feat_f1) = self.backbone(data['image1'])
            ratio = round(data['hw0_i'][0]/data['hw1_i'][0])
            data.update({
                'hw0_c': feat_c1.shape[2:]*ratio, 'hw1_c': feat_c1.shape[2:],              #always dimension is [batch, channel, height, width], so here you are only taking heght, weight of feature map
                'hw0_f': feat_f1.shape[2:]*ratio, 'hw1_f': feat_f1.shape[2:]               #the height and width is feature map dimension
            })
            feat_c1 = rearrange(self.pos_encoding(feat_c1),
                                'n c h w -> n (h w) c')                                                            #add space encoding to coarse feature map of img1, and reshape to [N, HW, C]!!!
            return feat_c1, feat_f1
        else:
            if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
                feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))               #feed into backbone get feature map for img0 and img1, concatenate in batch dimension. backbone returns two feature map, one for 1/2 and one for 1/8 feature map
                (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])      #split feature map into two for two images
            else:  # handle different input shapes
                (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

            data.update({
                'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],                                            #always dimension is [batch, channel, height, width], so here you are only taking heght, weight of feature map
                'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]                                             #the height and width is feature map dimension
            })

            # 2. coarse-level loftr module
            # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
            feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')                               #add space encoding to coarse feature map of img0, and reshape to [N, HW, C]!!!
            feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')                               #add space encoding to coarse feature map of img1, and reshape to [N, HW, C]!!!

            return feat_c0, feat_c1, feat_f0, feat_f1

    def transformer(self, feat_c0, feat_c1, feat_f0, feat_f1, data):
        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)                              #feeding into coarse transformer

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)                        #remember feat_c0, feat_c1 is of dimension [N, HW, C]

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)