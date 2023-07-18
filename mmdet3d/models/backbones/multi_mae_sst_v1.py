from mmdet.models import BACKBONES

import torch
import torch.nn as nn
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import auto_fp16
from mmdet3d.models.sst.sst_basic_block import BasicShiftBlock

from mmdet3d.ops import flat2window, window2flat
from ipdb import set_trace
import numpy as np
import random


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@BACKBONES.register_module()
class MultiMAESST(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        window_shape,
        shifts_list,
        point_cloud_range,
        voxel_size,
        shuffle_voxels=False,
        d_model=[],
        nhead=[],
        sub_voxel_ratio_low=[],
        sub_voxel_ratio_med=[],
        cls_sub_voxel=False,
        encoder_num_blocks=6,
        decoder_num_blocks=2,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        # num_attached_conv=2,
        # conv_in_channel=64,
        # conv_out_channel=64,
        # norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        # conv_cfg=dict(type='Conv2d', bias=False),
        debug=True,
        drop_info=None,
        normalize_pos=False,
        pos_temperature=10000,
        in_channel=None,
        conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
        checkpoint_blocks=[],
        ):
        super().__init__()
        
        assert drop_info is not None

        self.shifts_list = shifts_list
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.shuffle_voxels = shuffle_voxels

        self.meta_drop_info = drop_info
        self.pos_temperature = pos_temperature
        self.d_model = d_model
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks
        self.cls_sub_voxel=cls_sub_voxel

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        encoder_block_list=[]
        decoder_block_list = []
        for i in range(encoder_num_blocks):
            encoder_block_list.append(
                BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i)
            )
        for i in range(decoder_num_blocks):
            decoder_block_list.append(
                BasicShiftBlock(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i)
            )
        self.encoder_blocks = nn.ModuleList(encoder_block_list)
        self.decoder_blocks = nn.ModuleList(decoder_block_list)
        self.mask_token = nn.Parameter(torch.zeros(1, d_model[-1]))
        self.per_sub_voxel_num_low = sub_voxel_ratio_low[0] * sub_voxel_ratio_low[1] * sub_voxel_ratio_low[2]
        self.per_sub_voxel_num_med = sub_voxel_ratio_med[0] * sub_voxel_ratio_med[1] * sub_voxel_ratio_med[2]
        self.decoder_pred_low = nn.Linear(d_model[-1], self.per_sub_voxel_num_low * 3 , bias=True)
        self.decoder_pred_med = nn.Linear(d_model[-1], self.per_sub_voxel_num_med * 3 , bias=True)
        self.decoder_pred_top = nn.Linear(d_model[-1], 3, bias=True)

        if cls_sub_voxel:
            self.cls_pred_low = nn.Linear(d_model[-1], self.per_sub_voxel_num_low * 2 , bias=True)
            self.cls_pred_med = nn.Linear(d_model[-1], self.per_sub_voxel_num_med * 2 , bias=True)
        self._reset_parameters()

        self.output_shape = output_shape
        self.debug = debug

    @auto_fp16(apply_to=('voxel_feat',))
    def forward(self, voxel_feat, coors,coors_mask,batch_size):
        visible_tokens_tuple = self.get_voxel_info(voxel_feat, coors, batch_size)
        visible_output = self.forward_encoder(visible_tokens_tuple)
        x = self.forward_decoder(visible_output, coors, coors_mask, batch_size)
        return x

    @auto_fp16(apply_to=('voxel_feat',))
    def get_voxel_info(self, voxel_feat, coors, batch_size):
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
        self.set_drop_info()
        voxel_info = {}
        coors = coors.long()
        # print('voxel_feat',voxel_feat.dtype)
        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            num_voxel = len(voxel_feat)
            shuffle_inds = torch.randperm(num_voxel)
            voxel_feat = voxel_feat[shuffle_inds]
            coors = coors[shuffle_inds]
            for k, tensor in voxel_info.items():
                if isinstance(tensor, torch.Tensor) and len(tensor) == num_voxel:
                    voxel_info[k] = tensor[shuffle_inds]

        voxel_info = self.window_partition(coors, voxel_info)
        voxel_info = self.get_voxel_keep_inds(voxel_info,
                                                   len(self.shifts_list))  # voxel_info is updated in this function
        voxel_keep_inds = voxel_info['voxel_keep_inds']


        voxel_num_before_drop = len(voxel_feat)
        voxel_feat = voxel_feat[voxel_keep_inds]
        coors = coors[voxel_keep_inds]
        voxel_info['coors'] = coors

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        flat2win_inds_list = [
            self.get_flat2win_inds(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'])
            for i in range(len(self.shifts_list))
        ]

        if self.debug:
            coors_3d_dict_shift0 = flat2window(coors, voxel_info['voxel_drop_level_shift0'], flat2win_inds_list[0],
                                               self.drop_info)
            coors_2d = window2flat(coors_3d_dict_shift0, flat2win_inds_list[0])
            assert (coors_2d == coors).all()

        return voxel_feat, flat2win_inds_list, voxel_info


    def forward_encoder(self, input_tuple):
        '''
        '''
        voxel_feat, ind_dict_list, voxel_info = input_tuple # 3 outputs of SSTInputLayer, containing pre-computed information for Sparse Regional Attention.
        # print(voxel_feat.dtype)
        assert voxel_info['coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        self.set_drop_info()
        device = voxel_info['coors'].device
        batch_size = voxel_info['coors'][:, 0].max().item() + 1

        num_shifts = len(ind_dict_list) # Usually num_shifts == 2, one for non-shifted layout, one for shifted layout
        
        padding_mask_list = [
            self.get_key_padding_mask(ind_dict_list[i], voxel_info[f'voxel_drop_level_shift{i}'], device) 
            for i in range(num_shifts)
        ]
        pos_embed_list = [
            self.get_pos_embed(
                _t,
                voxel_info[f'coors_in_win_shift{i}'],
                voxel_info[f'voxel_drop_level_shift{i}'],
                voxel_feat.dtype,
                voxel_info.get(f'voxel_win_level_shift{i}', None)
            ) 
            for i, _t in enumerate(ind_dict_list) # 2-times for-loop, one for non-shifted layout, one for shifted layout
        ]
        voxel_drop_level_list = [voxel_info[f'voxel_drop_level_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)
        for i, block in enumerate(self.encoder_blocks):
            output = block(output, pos_embed_list, ind_dict_list, voxel_drop_level_list,
                padding_mask_list, self.drop_info)
        return output


    def forward_decoder(self, visible_voxel_feat,coors,coors_mask,batch_size):
        '''
        '''
        masked_start_id=coors.shape[0]
        mask_tokens = self.mask_token.repeat(coors_mask.shape[0], 1)
        # print(voxel_feat.shape,mask_tokens.shape)
        voxel_feat_ = torch.cat([visible_voxel_feat,mask_tokens],dim=0)
        coors_ = torch.cat([coors,coors_mask],dim=0)
        voxel_feat, ind_dict_list, voxel_info = self.get_voxel_info(voxel_feat_,coors_,batch_size)

        assert voxel_info['coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'
        self.set_drop_info()
        device = voxel_info['coors'].device

        num_shifts = len(ind_dict_list)  # Usually num_shifts == 2, one for non-shifted layout, one for shifted layout

        padding_mask_list = [
            self.get_key_padding_mask(ind_dict_list[i], voxel_info[f'voxel_drop_level_shift{i}'], device)
            for i in range(num_shifts)
        ]
        pos_embed_list = [
            self.get_pos_embed(
                _t,
                voxel_info[f'coors_in_win_shift{i}'],
                voxel_info[f'voxel_drop_level_shift{i}'],
                voxel_feat.dtype,
                voxel_info.get(f'voxel_win_level_shift{i}', None)
            )
            for i, _t in enumerate(ind_dict_list)
            # 2-times for-loop, one for non-shifted layout, one for shifted layout
        ]
        voxel_drop_level_list = [voxel_info[f'voxel_drop_level_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        for i, block in enumerate(self.decoder_blocks):
            output = block(output, pos_embed_list, ind_dict_list, voxel_drop_level_list,
                           padding_mask_list, self.drop_info)

        masked_output = output[masked_start_id:]
        reg_pred_low = self.decoder_pred_low(masked_output).view(-1, self.per_sub_voxel_num_low, 3)
        reg_pred_med = self.decoder_pred_med(masked_output).view(-1, self.per_sub_voxel_num_med, 3)
        reg_pred_top = self.decoder_pred_top(masked_output)
        if self.cls_sub_voxel:
            cls_pred_low = self.cls_pred_low(masked_output).view(-1, self.per_sub_voxel_num_low, 2)
            cls_pred_med = self.cls_pred_med(masked_output).view(-1, self.per_sub_voxel_num_med, 2)
            return reg_pred_low,reg_pred_med,reg_pred_top,cls_pred_low,cls_pred_med,
        else:
            return reg_pred_low,reg_pred_med,reg_pred_top


    def get_key_padding_mask(self, ind_dict, voxel_drop_lvl, device):
        num_all_voxel = len(voxel_drop_lvl)
        key_padding = torch.ones((num_all_voxel, 1)).to(device).bool()

        window_key_padding_dict = flat2window(key_padding, voxel_drop_lvl, ind_dict, self.drop_info)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict
        
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas

    def get_pos_embed(self, ind_dict, coors_in_win, voxel_drop_level, dtype, voxel_window_level):
        '''
        Args:
        '''

        # [N,]
        win_x, win_y = self.window_shape

        x, y = coors_in_win[:, 0] - win_x/2, coors_in_win[:, 1] - win_y/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]

        pos_length = self.d_model[0] // 2
        assert self.d_model[0] == self.d_model[1] == self.d_model[-1], 'If you want to use different d_model, Please implement corresponding pos embendding.'
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()],
                              dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()],
                              dim=-1).flatten(1)
        # [num_tokens, pos_length * 2]
        pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)

        window_pos_emb_dict = flat2window(
            pos_embed_2d, voxel_drop_level, ind_dict, self.drop_info)
        
        return window_pos_emb_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta

    @torch.no_grad()
    def get_flat2win_inds(self, batch_win_inds, voxel_drop_lvl):
        '''
        Args:
            batch_win_inds: shape=[N, ]. Indicates which window a voxel belongs to. Window inds is unique is the whole batch.
            voxel_drop_lvl: shape=[N, ]. Indicates batching_level of the window the voxel belongs to.
        Returns:
            flat2window_inds_dict: contains flat2window_inds of each voxel, shape=[N,]
                Determine the voxel position in range [0, num_windows * max_tokens) of each voxel.
        '''
        device = batch_win_inds.device

        flat2window_inds_dict = {}

        drop_info = self.drop_info

        for dl in drop_info:

            dl_mask = voxel_drop_lvl == dl
            if not dl_mask.any():
                continue

            conti_win_inds = self.make_continuous_inds(batch_win_inds[dl_mask])

            num_windows = len(torch.unique(conti_win_inds))
            max_tokens = drop_info[dl]['max_tokens']

            inner_win_inds = self.get_inner_win_inds(conti_win_inds)

            flat2window_inds = conti_win_inds * max_tokens + inner_win_inds

            flat2window_inds_dict[dl] = (flat2window_inds, torch.where(dl_mask))

            if self.debug:
                assert inner_win_inds.max() < max_tokens, f'Max inner inds({inner_win_inds.max()}) larger(equal) than {max_tokens}'
                assert (flat2window_inds >= 0).all()
                max_ind = flat2window_inds.max().item()
                assert max_ind < num_windows * max_tokens, f'max_ind({max_ind}) larger than upper bound({num_windows * max_tokens})'
                assert max_ind >= (
                            num_windows - 1) * max_tokens, f'max_ind({max_ind}) less than lower bound({(num_windows - 1) * max_tokens})'

        return flat2window_inds_dict

    @torch.no_grad()
    def get_inner_win_inds(self, win_inds):
        '''
        Fast version of get_innner_win_inds_slow

        Args:
            win_inds indicates which windows a voxel belongs to. Voxels share a window have same inds.
            shape = [N,]

        Return:
            inner_inds: shape=[N,]. Indicates voxel's id in a window. if M voxels share a window, their inner_inds would be torch.arange(M, dtype=torch.long)

        Note that this function might output different results from get_inner_win_inds_slow due to the unstable pytorch sort.
        '''

        sort_inds, order = win_inds.sort()  # sort_inds is like [0,0,0, 1, 2,2] -> [0,1, 2, 0, 0, 1]
        roll_inds_left = torch.roll(sort_inds, -1)  # [0,0, 1, 2,2,0]

        diff = sort_inds - roll_inds_left  # [0, 0, -1, -1, 0, 2]
        end_pos_mask = diff != 0

        bincount = torch.bincount(win_inds)
        # assert bincount.max() <= max_tokens
        unique_sort_inds, _ = torch.sort(torch.unique(win_inds))
        num_tokens_each_win = bincount[unique_sort_inds]  # [3, 1, 2]

        template = torch.ones_like(win_inds)  # [1,1,1, 1, 1,1]
        template[end_pos_mask] = (num_tokens_each_win - 1) * -1  # [1,1,-2, 0, 1,-1]

        inner_inds = torch.cumsum(template, 0)  # [1,2,0, 0, 1,0]
        inner_inds[end_pos_mask] = num_tokens_each_win  # [1,2,3, 1, 1,2]
        inner_inds -= 1  # [0,1,2, 0, 0,1]

        # recover the order
        inner_inds_reorder = -torch.ones_like(win_inds)
        inner_inds_reorder[order] = inner_inds

        ##sanity check
        if self.debug:
            assert (inner_inds >= 0).all()
            assert (inner_inds == 0).sum() == len(unique_sort_inds)
            assert (num_tokens_each_win > 0).all()
            random_win = unique_sort_inds[random.randint(0, len(unique_sort_inds) - 1)]
            random_mask = win_inds == random_win
            num_voxel_this_win = bincount[random_win].item()
            random_inner_inds = inner_inds_reorder[random_mask]

            assert len(torch.unique(random_inner_inds)) == num_voxel_this_win
            assert random_inner_inds.max() == num_voxel_this_win - 1
            assert random_inner_inds.min() == 0

        return inner_inds_reorder

    def get_inner_win_inds_slow(self, win_inds):
        unique_win_inds = torch.unique(win_inds)
        inner_inds = -torch.ones_like(win_inds)
        for ind in unique_win_inds:
            mask = win_inds == ind
            num = mask.sum().item()
            inner_inds[mask] = torch.arange(num, dtype=win_inds.dtype, device=win_inds.device)
        assert (inner_inds >= 0).all()
        return inner_inds

    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = self.get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds]  #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl

        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel

        # print(keep_mask.shape,keep_mask.sum())
        return keep_mask, drop_lvl_per_voxel

    @torch.no_grad()
    def get_voxel_keep_inds_loop(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        for i in range(num_shifts):
            batch_win_inds_si = voxel_info[f'batch_win_inds_shift{i}']
            keep_mask_si, drop_lvl_si = self.drop_single_shift(batch_win_inds_si)
            if self.debug:
                assert (drop_lvl_si >= 0).all()

            drop_lvl_si = drop_lvl_si[keep_mask_si]
            batch_win_inds_si = batch_win_inds_si[keep_mask_si]

            voxel_info[f'voxel_drop_level_shift{i}'] = drop_lvl_si
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds_si
        return voxel_info

    @torch.no_grad()
    def get_voxel_keep_inds(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        ### sanity check
        if self.debug:
            for dl in self.drop_info:
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    #print(f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    #print(f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors, voxel_info):

        shifts_list = self.shifts_list
        win_shape_x, win_shape_y = self.window_shape
        pc_range = self.point_cloud_range
        voxel_size = self.voxel_size  # using the min voxel size
        assert isinstance(voxel_size, tuple)

        bev_shape_x = int(np.ceil((pc_range[3] - pc_range[0]) / voxel_size[0]))
        bev_shape_y = int(np.ceil((pc_range[4] - pc_range[1]) / voxel_size[1]))

        max_num_win_x = int(np.ceil((bev_shape_x / win_shape_x)) + 1)  # plus one here to meet the needs of shift.
        max_num_win_y = int(np.ceil((bev_shape_y / win_shape_y)) + 1)  # plus one here to meet the needs of shift.
        max_num_win_per_sample = max_num_win_x * max_num_win_y

        for i in range(len(shifts_list)):
            shift_x, shift_y = shifts_list[i]
            assert shift_x == 0 or shift_x == win_shape_x // 2, 'Usually ...'
            shifted_coors_x = coors[:, 3] + (win_shape_x - shift_x if shift_x > 0 else 0)
            shifted_coors_y = coors[:, 2] + (win_shape_y - shift_y if shift_y > 0 else 0)

            win_coors_x = shifted_coors_x // win_shape_x
            win_coors_y = shifted_coors_y // win_shape_y
            batch_win_inds = coors[:, 0] * max_num_win_per_sample + win_coors_x * max_num_win_y + win_coors_y
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds

            coors_in_win_x = shifted_coors_x % win_shape_x
            coors_in_win_y = shifted_coors_y % win_shape_y
            voxel_info[f'coors_in_win_shift{i}'] = torch.stack([coors_in_win_x, coors_in_win_y], dim=-1)

        return voxel_info

    @torch.no_grad()
    def make_continuous_inds(self, inds):
        '''
        Make batch_win_inds continuous, e.g., [1, 3, 4, 6, 10] -> [0, 1, 2, 3, 4].
        '''

        dtype = inds.dtype
        device = inds.device

        unique_inds, _ = torch.sort(torch.unique(inds))
        num_valid_inds = len(unique_inds)
        max_origin_inds = unique_inds.max().item()
        canvas = -torch.ones((max_origin_inds + 1,), dtype=dtype, device=device)
        canvas[unique_inds] = torch.arange(num_valid_inds, dtype=dtype, device=device)

        conti_inds = canvas[inds]

        if self.debug:
            assert conti_inds.max() == len(torch.unique(conti_inds)) - 1, 'Continuity check failed.'
            assert conti_inds.min() == 0, '-1 in canvas should not be indexed.'
        return conti_inds




