# Do not use this file. Please wait for future release.
from mmdet.models import BACKBONES

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import auto_fp16
from mmdet3d.models.sst.sra_block import SRABlock
from mmdet3d.ops import SRATensor, DebugSRATensor, spconv

from ipdb import set_trace


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@BACKBONES.register_module()
class SSTOutConv(nn.Module):

    def __init__(
        self,

        num_attached_conv=0,
        conv_in_channel=64,
        conv_out_channel=64,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),

        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],

        fp16=True,
        ):
        super().__init__()
        
        # assert isinstance(batching_info, tuple)
        # self.batching_info = batching_info
        # self.no_pos_embed = no_pos_embed
        # self.pos_temperature = pos_temperature
        # self.d_model = d_model
        # self.window_shape = window_shape
        # self.key = key
        # self.normalize_pos = normalize_pos
        # self.nhead = nhead
        # self.checkpoint_blocks = checkpoint_blocks
        # self.init_sparse_shape = init_sparse_shape
        self.fp16 = fp16


        self._reset_parameters()


        self.num_attached_conv = num_attached_conv

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                    )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)
            
            self.conv_layer = nn.ModuleList(conv_list)

    def forward(self, x):

        # voxel_feats, voxel_coors, batch_size = input_tuple
        # voxel_coors = voxel_coors.long()
        # if self.fp16:
        #     x = x.to(torch.half)
        # if self.training:
        #     batching_info = self.batching_info[0]
        # else:
        #     batching_info = self.batching_info[1]



        if self.num_attached_conv > 0:
            for conv in self.conv_layer:
                x = conv(x)

        output_list = []
        output_list.append(x)

        return output_list
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

