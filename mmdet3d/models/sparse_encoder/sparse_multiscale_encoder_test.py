import math
import numpy as np

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from ..builder import SPARSE_ENCODERS
from mmdet3d.ops import flat2window, window2flat
import random
import pickle as pkl
import os
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule

@SPARSE_ENCODERS.register_module()
class SpasreMultiscaleEncoderTest(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]). R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching.
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing.
    """

    def __init__(self,
                 in_channel=None,
                 spatial_shape=None,
                 stage_channels=[],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 debug=False,
                 ):
        super().__init__()
        # self.fp16_enabled = False
        self.debug = debug

        self.spatial_shape = spatial_shape

        # self.submconv = make_sparse_convmodule(
        #         in_channel,
        #         in_channel,
        #         3,
        #         norm_cfg=norm_cfg,
        #         padding=1,
        #         indice_key='subm1',
        #         conv_type='SubMConv3d')
        #
        # self.spconv_downsample = make_sparse_convmodule(
        #         stage_channels[0],
        #         stage_channels[1],
        #         kernel_size=[3, 3, 3],
        #         stride=[2, 2, 2],
        #         norm_cfg=norm_cfg,
        #         padding=[0, 1, 1],
        #         indice_key='spconv1',
        #         conv_type='SparseConv3d')
        # self.linear=nn.Linear(in_channel,stage_channels[0])
        self.spconv = spconv.SparseSequential(
            make_sparse_convmodule(
                in_channel,
                stage_channels[0],
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d'),
            make_sparse_convmodule(
                stage_channels[0],
                stage_channels[1],
                kernel_size=[3, 3, 3],
                stride=[2, 2, 2],
                norm_cfg=norm_cfg,
                padding=[0, 1, 1],
                indice_key='spconv1',
                conv_type='SparseConv3d'),
        )

    @force_fp32(apply_to=('voxel_feat'), out_fp16=True)
    def forward(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
        # print('voxel_feat',voxel_feat.dtype,type(coors))
        sp_tensor = spconv.SparseConvTensor(voxel_feat, coors.int(),
                                            self.spatial_shape,
                                            batch_size)

        # sp_tensor = self.submconv(sp_tensor)
        # sp_tensor.features = self.linear(sp_tensor.features)
        # sp_tensor = self.spconv_downsample(sp_tensor)
        sp_tensor = self.spconv(sp_tensor)

        return sp_tensor.features, sp_tensor.indices,batch_size
