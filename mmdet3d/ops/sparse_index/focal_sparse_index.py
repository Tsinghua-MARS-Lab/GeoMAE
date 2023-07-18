import torch
from torch import nn
from torch.autograd import Function
from .sparse_index_ext import sparse_index_test,sparse_index_test_,sparse_index_backward_test
import time
#torch.set_printoptions(profile="full")


class FocalSparseIndexFunctionLongTest_(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,layer=None):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        relative_pos_list = []
        points_index_list = []
        dilated=False
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=coors.device,dtype=torch.int32)
            #window_offset= torch.tensor(window_offset, device=coors.device)
            dilated_size=torch.tensor([1,1,2,2],device=voxel_features.device)
            partition_mode = torch.tensor(0, device=voxel_features.device)
            if inter_flag:
                if dilated:
                    windows_partition = coors // dilated_size[None, :] % window_sizes[None, :]
                    partition_mode = torch.tensor(1, device=voxel_features.device)
                    r_pos_coor = coors[:, 1:] //( window_sizes[None, 1:] * dilated_size[None,1:])

                else:
                    windows_partition = coors % window_sizes[None, :]
                    partition_mode = torch.tensor(1, device=voxel_features.device)
                    r_pos_coor = coors[:, 1:] // window_sizes[None, 1:]

            else:
                if window_offset is not None:
                    window_offset = torch.tensor(window_offset, device=coors.device)
                    windows_partition = (coors+window_offset) // window_sizes[None, :]
                    r_pos_coor = (coors+window_offset)[:, 1:] % window_sizes[None, 1:]
                else:
                    windows_partition = coors  // window_sizes[None, :]
                    r_pos_coor = coors [:, 1:] % window_sizes[None, 1:]
            #print('partition mode', partition_mode)
            #voxel_features=torch.cat(voxel_features,max_featuers)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)



            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            # tensor_lengthes = torch.tensor((1,2,4,8,16,32,64,128,256,384,512), dtype=torch.int32,
            #                                  device=window_to_tensor_idx.device)

            count = 0
            #print('windows_partition', windows_partition.shape)
            #if partition_mode == 1:
            #    print(_[:10], window_lengthes.max())

            # print('windows num', window_lengthes.shape, 'voxel num', voxel_features.shape[0])
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)


        valid_length=[]
        #w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
        for i in range(tensor_lengthes.shape[0]):
            length=tensor_lengthes[i]
            if i==0:
                region_in_tensor_index =  (window_lengthes <= length)
            else:
                region_in_tensor_index = (window_lengthes > tensor_lengthes[i-1]) & (window_lengthes <= length)
            if region_in_tensor_index.sum()==0:
                continue
            valid_length.append(i)
            window_regions[region_in_tensor_index] = count
            region_tensor = window_lengthes[region_in_tensor_index]
            region_length = region_tensor.shape[0]
            region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
            window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
            sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
            sparse_tensor_index_list.append(coors.new_zeros(region_length, length, dtype=torch.bool))
            points_index_list.append(coors.new_zeros(region_length, length, dtype=torch.int32))
            relative_pos_list.append(coors.new_zeros(region_length, length, 3, dtype=torch.int32,requires_grad=False))
            count=count+1
            # if i==0:
            #     print(layer, inter_flag, 'range {} - {} :'.format(0, length), region_tensor.shape)
            # else:
            #     print(layer,inter_flag,'range {} - {} :'.format(tensor_lengthes[i-1], length), region_tensor.shape)
        #print()
        tensor_lengthes=tensor_lengthes[valid_length]
        sparse_index_test(sparse_tensor_list, sparse_tensor_index_list, points_index_list, relative_pos_list,
                          point_to_window_idx,
                          r_pos_coor.contiguous().to(dtype=torch.int32), voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape,partition_mode)
        with torch.no_grad():

            if partition_mode==0:
                w1=window_sizes[1]
                w2=window_sizes[2]
                w3=window_sizes[3]
            else:
                w1 = ((spatial_shape[0]-1) // window_sizes[1]) + 1
                w2 = ((spatial_shape[1]-1) // window_sizes[2]) + 1
                w3 = ((spatial_shape[2]-1) // window_sizes[3]) + 1
            for i in range(len(relative_pos_list)):
                pos = relative_pos_list[i]
                pos = pos.permute(2, 0, 1)
                pos = pos[:, :, :, None] - pos[:, :, None, :]
                pos = pos.permute(1, 2, 3, 0).contiguous()
                pos[:, :, :, 0] += (w1 - 1)
                pos[:, :, :, 1] += (w2 - 1)
                pos[:, :, :, 2] += (w3 - 1)
                pos[:, :, :, 0] *= ((2 * w3 - 1)*(2 * w2 - 1)) # TODO: How to avoid different pos but same sum ? & require_grad=False ?
                pos[:, :, :, 1] *= (2 * w3 - 1)
                relative_pos_list[i] = pos.sum(-1)
        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in relative_pos_list:
            return_tuple.append(sparse_tensor_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//4]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape,partition_mode=ctx.saved_tensors
        grad_voxel=torch.zeros((voxel_shape[0],voxel_shape[1]),dtype=torch.float32,device=voxel_shape.device)
        # print('grad type & len &shape', grad_tensor_list[0].dtype, len(a) // 4,grad_voxel.shape)
        # for i in range(len(a)//4):
        #      print(voxel_shape[1],i,torch.isnan(a[i]).sum()/voxel_shape[1]/a[i].shape[1],a[i].max())
        sparse_index_backward_test(
            grad_voxel,
            grad_tensor_list,
            point_to_window_idx,
            window_indexes.to(dtype=torch.int32),
            window_regions.to(dtype=torch.int32),
            window_to_tensor_idx.to(dtype=torch.int32),
            tensor_lengthes,
            len(a) // 4
        )
        return None,grad_voxel,None,None,None,None,None,None,None,None

focal_sparse_index_sample=FocalSparseIndexFunctionLongTest_.apply