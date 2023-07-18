import torch
from torch import nn
from torch.autograd import Function
from .sparse_index_ext import sparse_index_test,sparse_index_test_,sparse_index_backward_test,\
    sparse_index_wo_pos,sparse_index_with_pos,sparse_index_with_pos_half,sparse_index_backward_half
import time
#torch.set_printoptions(profile="full")

def test_sparse_index(sparse_tensor_index_list,
                          voxel_features,
                          window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes):
    window_count=window_to_tensor_idx.new_zeros(window_to_tensor_idx.shape)
    for idx,window_i in enumerate(window_indexes):
        sparse_tensor_i=sparse_tensor_index_list[window_regions[window_i]]
        tensor_pos=window_to_tensor_idx[window_i]
        window_pos=window_count[window_i]

        #print(sparse_tensor_i.shape,tensor_pos,window_pos)
        sparse_tensor_i[tensor_pos][window_pos]=voxel_features[idx]
        window_count[window_i] = window_count[window_i] + 1


class SparseIndex(Function):
    @staticmethod

    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,window_sizes:int):
        #coors=coors/window_sizes
        #print('coors',coors.shape)
        #torch.save(coors,'./test_coor_{}.pth'.format(int(time.time())))

        sparse_tensor_list=[]
        sparse_tensor_index_list=[]
        points_index_list=[]
        feature_channel=voxel_features.shape[1]
        window_sizes = torch.tensor([1, 4, 4, 8], device=coors.device)
        # print()
        # print()
        # window_sizes=torch.tensor([1,4,4,8],device=coors.device)
        # print('test divde',coors[:5],(coors//8)[:5],(coors//window_sizes[None,:])[:5])
        # print()
        # print()

        _, window_indexes, window_lengthes = torch.unique(coors// window_sizes[None,:], sorted=True, return_inverse=True,
                                                          return_counts=True,dim=0)
        #window_indexes=window_indexes.to(dtype=torch.int32)
        #window_lengthes=window_lengthes.to(dtype=torch.int32)
        window_to_tensor_idx = torch.zeros_like(window_lengthes,device=window_lengthes.device)
        window_regions = torch.zeros_like(window_lengthes,device=window_lengthes.device)
        point_to_window_idx = -torch.ones(coors.shape[0],device=window_lengthes.device,dtype=torch.int32)
        tensor_lengthes=torch.logspace(start=0, end=8, steps=9, base=2,dtype=torch.int32,device=window_to_tensor_idx.device)
        # print('tensor_lengthes',tensor_lengthes)
        # print('window_indexes',window_indexes.shape)
        # print('window_lengthes',_.shape,window_lengthes.shape)
        # print('window_regions',window_regions.dtype)
        count=0

        for i in range(9):
            region_in_tensor_index=(window_lengthes<=2**i)
            window_regions[region_in_tensor_index]=i
            region_tensor=window_lengthes[region_in_tensor_index]
            region_length=region_tensor.shape[0]
            region_to_tensor_index=torch.arange(region_length,device=region_tensor.device)
            window_to_tensor_idx[region_in_tensor_index]=region_to_tensor_index
            sparse_tensor_list.append(voxel_features.new_zeros(region_length, 2 ** i, feature_channel))
            sparse_tensor_index_list.append(voxel_features.new_zeros(region_length, 2 ** i,dtype=torch.bool))
            points_index_list.append(voxel_features.new_zeros(region_length, 2 ** i, dtype=torch.int32))

        #     print('range {} - {} :'.format(pow(2, i-1), pow(2, i)), region_tensor.shape)
        # #print(window_to_tensor_idx[:20])
        # print('data_ptr',type(window_lengthes.data_ptr()),window_lengthes.device,torch.zeros(1).data_ptr())
        #
        # print((coors//window_sizes).dtype)
        # print('sparse_tensor_list',len(sparse_tensor_list))
        # print('window_to_tensor_idx',window_to_tensor_idx.dtype)
        # print('window_to_tensor_idx', window_to_tensor_idx.dtype)
        # print('sparse tensor before', sparse_tensor_list[0][0][0])
        # print()
        # print('window_index',window_indexes[0:4],window_regions[window_indexes[0:4]],tensor_lengthes[window_regions[window_indexes[0:4]]])


        sparse_index_test(sparse_tensor_list,sparse_tensor_index_list,points_index_list,
                          point_to_window_idx,
                          (coors//window_sizes)[:,1:].contiguous(),voxel_features,
                          window_indexes.to(dtype=torch.int32),window_regions.to(dtype=torch.int32),window_to_tensor_idx.to(dtype=torch.int32),tensor_lengthes,3,9)
        # test_sparse_index(sparse_tensor_index_list, voxel_features, window_indexes, window_regions,window_to_tensor_idx, tensor_lengthes)
        # #print('point_to_window_idx',(point_to_window_idx>256).sum())

        # print('test sparse tensor',sparse_tensor_index_list[0][0][0])
        # for i in range(10):
        #     print('test equal {}'.format(i),(sparse_tensor_list[i]==sparse_tensor_index_list[i]).all())
        #print('time',time1-time0,time0-time_1,time2-time1,time3-time2)
        #print('sparse tensor', sparse_tensor_list[4][0][0], sparse_tensor_list[4].shape,
        #      sparse_tensor_index_list[4].shape, sparse_tensor_index_list[4][0],'   ',voxel_features[points_index_list[4][0][0]][:10])
        count=0
        test_pos_tensor=torch.tensor([0,13,27,8],device=coors.device)
        for i in range(len(sparse_tensor_list)):
            count+=sparse_tensor_list[i][sparse_tensor_index_list[i]].shape[0]
        #print("test pos",coors[test_pos_tensor])
        return (sparse_tensor_list,sparse_tensor_index_list,points_index_list)
sparse_index=SparseIndex.apply

# def sparse_index_sample(coors:torch.Tensor,voxel_features:torch.Tensor,window_sizes,inter_flag=False):
#
#         sparse_tensor_list=[]
#         sparse_tensor_index_list=[]
#         relative_pos_list=[]
#         points_index_list=[]
#         feature_channel=voxel_features.shape[1]
#         window_sizes = torch.tensor(window_sizes, device=coors.device)
#         # print()
#         # print()
#         # window_sizes=torch.tensor([1,4,4,8],device=coors.device)
#         # print('test divde',coors[:5],(coors//8)[:5],(coors//window_sizes[None,:])[:5])
#         # print()
#         # print()
#         #print('coors',coors[:,1].max())
#         if inter_flag:
#             windows_partition = coors% window_sizes[None,:]
#         else:
#             windows_partition = coors // window_sizes[None, :]
#         _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
#                                                           return_counts=True,dim=0)
#         #window_indexes=window_indexes.to(dtype=torch.int32)
#         #window_lengthes=window_lengthes.to(dtype=torch.int32)
#         window_to_tensor_idx = torch.zeros_like(window_lengthes,device=window_lengthes.device)
#         window_regions = torch.zeros_like(window_lengthes,device=window_lengthes.device)
#         point_to_window_idx = -torch.ones(coors.shape[0],device=window_lengthes.device,dtype=torch.int32)
#         tensor_lengthes=torch.logspace(start=0, end=8, steps=9, base=2,dtype=torch.int32,device=window_to_tensor_idx.device)
#
#         count=0
#         time_1=time.time()
#         print('windows num',window_lengthes.shape,'voxel num',voxel_features.shape[0])
#         for i in range(9):
#             region_in_tensor_index=(window_lengthes>2**(i-1))&(window_lengthes<=2**i)
#             window_regions[region_in_tensor_index]=i
#             region_tensor=window_lengthes[region_in_tensor_index]
#             region_length=region_tensor.shape[0]
#             region_to_tensor_index=torch.arange(region_length,device=region_tensor.device)
#             window_to_tensor_idx[region_in_tensor_index]=region_to_tensor_index
#             sparse_tensor_list.append(voxel_features.new_zeros(region_length, 2 ** i, feature_channel))
#             sparse_tensor_index_list.append(coors.new_zeros(region_length, 2 ** i,dtype=torch.bool))
#             points_index_list.append(coors.new_zeros(region_length, 2 ** i, dtype=torch.int32))
#             relative_pos_list.append(coors.new_zeros(region_length, 2 ** i,3, dtype=torch.int32))
#
#             print('range {} - {} :'.format(pow(2, i-1), pow(2, i)), region_tensor.shape)
#         # #print(window_to_tensor_idx[:20])
#         # print('data_ptr',type(window_lengthes.data_ptr()),window_lengthes.device,torch.zeros(1).data_ptr())
#         #
#         # print((coors//window_sizes).dtype)
#         # print('sparse_tensor_list',len(sparse_tensor_list))
#         # print('window_to_tensor_idx',window_to_tensor_idx.dtype)
#         # print('window_to_tensor_idx', window_to_tensor_idx.dtype)
#         # print('sparse tensor before', sparse_tensor_list[0][0][0])
#         # print()
#         # print('window_index',window_indexes[0:4],window_regions[window_indexes[0:4]],tensor_lengthes[window_regions[window_indexes[0:4]]])
#
#         time0=time.time()
#         sparse_index_test(sparse_tensor_list,sparse_tensor_index_list,points_index_list,relative_pos_list,
#                           point_to_window_idx,
#                           windows_partition[:,1:].contiguous().to(dtype=torch.int32),voxel_features,
#                           window_indexes.to(dtype=torch.int32),window_regions.to(dtype=torch.int32),window_to_tensor_idx.to(dtype=torch.int32),tensor_lengthes,3,9)
#         # test_sparse_index(sparse_tensor_index_list, voxel_features, window_indexes, window_regions,window_to_tensor_idx, tensor_lengthes)
#         # #print('point_to_window_idx',(point_to_window_idx>256).sum())
#
#         # print('test sparse tensor',sparse_tensor_index_list[0][0][0])
#         # for i in range(10):
#         #     print('test equal {}'.format(i),(sparse_tensor_list[i]==sparse_tensor_index_list[i]).all())
#
#         # Time Test!
#         # time1=time.time()
#         # for i in range(9):
#         #     a=torch.nonzero(sparse_tensor_index_list[i])
#         # time2=time.time()
#         # for i in range(9):
#         #     b=sparse_tensor_index_list[i]!=0
#         # time3=time.time()
#         # print('time',time1-time0,time0-time_1,time2-time1,time3-time2)
#         # print('sparse tensor', sparse_tensor_list[4][0][0], sparse_tensor_list[4].shape,
#         #       sparse_tensor_index_list[4].shape, sparse_tensor_index_list[4][0],'   ',voxel_features[points_index_list[4][0][0]][:10])
#         # count=0
#         # test_pos_tensor=torch.tensor([0,13,27,8],device=coors.device)
#         # for i in range(len(sparse_tensor_list)):
#         #     count+=sparse_tensor_list[i][sparse_tensor_index_list[i]].shape[0]
#         # print("test pos",coors[test_pos_tensor])
#         for i in range(len(relative_pos_list)):
#             pos=relative_pos_list[i]
#             pos = pos.permute(2, 0, 1)
#             pos = pos[:, :, :, None] - pos[:, :, None, :]
#             pos = pos.permute(1, 2, 3, 0).contiguous()
#             pos[:, :, :, 0] += window_sizes[1] - 1
#             pos[:, :, :, 1] += window_sizes[2] - 1
#             pos[:, :, :, 2] += window_sizes[3] - 1
#             pos[:, :, :, 0] *= 2 * (window_sizes[2]+window_sizes[3]) - 1  #TODO: How to avoid different pos but same sum ? & require_grad=False ?
#             pos[:, :, :, 1] *= 2 * window_sizes[2]  - 1
#             relative_pos_list[i]=pos.sum(-1)
#
#         return sparse_tensor_list,sparse_tensor_index_list,points_index_list,relative_pos_list#window_lengthes.shape[0]



class SparseIndexFunction(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,window_sizes,inter_flag):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        relative_pos_list = []
        points_index_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)

            partition_mode = torch.tensor(0, device=voxel_features.device)
            if inter_flag:
                windows_partition = coors % window_sizes[None, :]
                partition_mode = torch.tensor(1, device=voxel_features.device)
                r_pos_coor=coors[:,1:]//window_sizes[None, 1:]
            else:
                windows_partition = coors // window_sizes[None, :]
                r_pos_coor = coors[:, 1:] % window_sizes[None, 1:]
            #print('partition mode', partition_mode)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)
            # window_indexes=window_indexes.to(dtype=torch.int32)
            # window_lengthes=window_lengthes.to(dtype=torch.int32)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            tensor_lengthes = torch.tensor((1,2,4,8,16,32,64,128,256,384,512), dtype=torch.int32,
                                             device=window_to_tensor_idx.device)

            count = 0
            #print('windows_partition', windows_partition.shape)
            #if partition_mode == 1:
            #    print(_[:10], window_lengthes.max())

            # print('windows num', window_lengthes.shape, 'voxel num', voxel_features.shape[0])
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)


        valid_length=[]
        #w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
        for i in range(11):
            length=tensor_lengthes[i]
            if i==0:
                region_in_tensor_index = (window_lengthes > 2 ** (i - 1)) & (window_lengthes <= length)
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

            #print('range {} - {} :'.format(pow(2, i - 1), pow(2, i)), region_tensor.shape)

        #print()
        tensor_lengthes=tensor_lengthes[valid_length]
        #print('tensor_lengthes',tensor_lengthes)
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
                w1 = (spatial_shape[0] // window_sizes[1]) + 1
                w2 = (spatial_shape[1]//window_sizes[2])+1
                w3 = (spatial_shape[2]//window_sizes[3])+1
            for i in range(len(relative_pos_list)):
                pos = relative_pos_list[i]
                pos = pos.permute(2, 0, 1)
                pos = pos[:, :, :, None] - pos[:, :, None, :]
                pos = pos.permute(1, 2, 3, 0).contiguous()
                pos[:, :, :, 0] += w1 - 1
                pos[:, :, :, 1] += w2 - 1
                pos[:, :, :, 2] += w3 - 1
                pos[:, :, :, 0] *= (2 * w3 - 1)*(2 * w2 - 1) # TODO: How to avoid different pos but same sum ? & require_grad=False ?
                pos[:, :, :, 1] *= 2 * w3 - 1
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
        #return sparse_tensor_list, sparse_tensor_index_list, points_index_list, relative_pos_list  # window_lengthes.shape[0]
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
        return None,grad_voxel,None,None,None


class SparseIndexFunctionLongTest(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,window_sizes,inter_flag):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        relative_pos_list = []
        points_index_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)

            partition_mode = torch.tensor(0, device=voxel_features.device)
            if inter_flag:
                windows_partition = coors % window_sizes[None, :]
                partition_mode = torch.tensor(1, device=voxel_features.device)
                r_pos_coor=coors[:,1:]//window_sizes[None, 1:]
            else:
                windows_partition = coors // window_sizes[None, :]
                r_pos_coor = coors[:, 1:] % window_sizes[None, 1:]
            #print('partition mode', partition_mode)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)


            # print('test window_lengthes',window_lengthes.dtype)
            # window_indexes=window_indexes.to(dtype=torch.int32)
            # window_lengthes=window_lengthes.to(dtype=torch.int32)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            tensor_lengthes = torch.tensor((1,2,4,8,16,32,64,128,256,384,512), dtype=torch.int32,
                                             device=window_to_tensor_idx.device)

            count = 0
            #print('windows_partition', windows_partition.shape)
            #if partition_mode == 1:
            #    print(_[:10], window_lengthes.max())

            # print('windows num', window_lengthes.shape, 'voxel num', voxel_features.shape[0])
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)


        valid_length=[]
        #w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
        for i in range(11):
            length=tensor_lengthes[i]
            if i==0:
                region_in_tensor_index = (window_lengthes <= length)
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

            #print('range {} - {} :'.format(pow(2, i - 1), pow(2, i)), region_tensor.shape)

        #print()
        tensor_lengthes=tensor_lengthes[valid_length]
        #print('tensor_lengthes',tensor_lengthes)
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
                w1 = (spatial_shape[0] // window_sizes[1]) + 1
                w2 = (spatial_shape[1] // window_sizes[2]) + 1
                w3 = (spatial_shape[2] // window_sizes[3]) + 1
            for i in range(len(relative_pos_list)):
                pos = relative_pos_list[i]
                pos = pos.permute(2, 0, 1)
                pos = pos[:, :, :, None] - pos[:, :, None, :]
                pos = pos.permute(1, 2, 3, 0).contiguous()
                pos[:, :, :, 0] += w1 - 1
                pos[:, :, :, 1] += w2 - 1
                pos[:, :, :, 2] += w3 - 1
                pos[:, :, :, 0] *= (2 * w3 - 1)*(2 * w2 - 1) # TODO: How to avoid different pos but same sum ? & require_grad=False ?
                pos[:, :, :, 1] *= 2 * w3 - 1
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
        #return sparse_tensor_list, sparse_tensor_index_list, points_index_list, relative_pos_list  # window_lengthes.shape[0]
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
        return None,grad_voxel,None,None,None

class SparseIndexFunctionLongTestWoPos(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes,window_offset=None,inter_flag=False,layer=None):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
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

                else:
                    windows_partition = coors % window_sizes[None, :]
                    partition_mode = torch.tensor(1, device=voxel_features.device)


            else:
                if window_offset is not None:
                    window_offset = torch.tensor(window_offset, device=coors.device)
                    windows_partition = (coors+window_offset) // window_sizes[None, :]

                else:
                    windows_partition = coors  // window_sizes[None, :]
            #print('partition mode', partition_mode)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)



            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
        valid_length=[]
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
            count=count+1

        tensor_lengthes=tensor_lengthes[valid_length]
        sparse_index_wo_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,
                          point_to_window_idx,
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape,partition_mode)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//3]

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
            len(a) // 3
        )
        return None,grad_voxel,None,None,None,None,None,None

class SparseFocalAwareWoPos(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        points_index_list = []
        window_indice_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=coors.device,dtype=torch.int32)

            windows_partition = coors  // window_sizes[None, :]

            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)

            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            valid_length=[]
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                    # print('0 - {}'.format(length),region_in_tensor_index.sum())
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                        # print('{} - {}'.format(tensor_lengthes[i - 1],length),region_in_tensor_index.sum())
                if region_in_tensor_index.sum() == 0:
                    continue
                window_indice_list.append(region_in_tensor_index)
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(coors.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(coors.new_zeros(region_length, length, dtype=torch.int32))
                count = count + 1

            tensor_lengthes = tensor_lengthes[valid_length]

        sparse_index_wo_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,
                          point_to_window_idx,
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for window_indice_i in window_indice_list:
            return_tuple.append(window_indice_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//4]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
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
        return None,grad_voxel,None,None,None,None,None,None


class SparseSwinWoPos(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes,swin_num):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        points_index_list = []
        window_indice_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=coors.device,dtype=torch.int32)

            windows_partition = (coors+torch.tensor([0,0,2,2],dtype=coors.dtype,device=coors.device)*swin_num)  // window_sizes[None, :]

            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)

            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            valid_length=[]
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                if region_in_tensor_index.sum() == 0:
                    continue
                window_indice_list.append(region_in_tensor_index)
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(coors.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(coors.new_zeros(region_length, length, dtype=torch.int32))
                count = count + 1

            tensor_lengthes = tensor_lengthes[valid_length]

        sparse_index_wo_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,
                          point_to_window_idx,
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for window_indice_i in window_indice_list:
            return_tuple.append(window_indice_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//4]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
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
        return None,grad_voxel,None,None,None,None,None,None




class SparseSwinWithAbsPos(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,tensor_lengthes,window_sizes,window_offsets=None,swin_num=None):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        points_index_list = []
        window_indice_list = []
        abs_pos_list = []
        window_coor_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=coors.device,dtype=torch.int32)
            if swin_num is not None:
                windows_partition = (coors + torch.tensor(window_offsets,dtype=coors.dtype,device=coors.device)* swin_num)  // window_sizes[None, :]
            else:
                windows_partition = (coors + torch.tensor(window_offsets, dtype=coors.dtype,
                                                          device=coors.device)) // window_sizes[None, :]

            window_coors, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)


            #print(window_coors.shape,window_indexes.shape,window_lengthes.shape,windows_partition.shape)
            #print(window_indexes.shape)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            valid_length=[]
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                if region_in_tensor_index.sum() == 0:
                    continue
                window_indice_list.append(region_in_tensor_index)
                window_coor_list.append(window_coors[region_in_tensor_index])
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(coors.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(coors.new_zeros(region_length, length, dtype=torch.int32))
                abs_pos_list.append(coors.new_zeros(region_length,length, 3, dtype=torch.int32))
                count = count + 1

            tensor_lengthes = tensor_lengthes[valid_length]
        if voxel_features.dtype == torch.float16:
            sparse_index_with_pos_half(sparse_tensor_list, sparse_tensor_index_list, points_index_list,abs_pos_list,
                          point_to_window_idx,
                          coors[:,1:],
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )
        else:
            sparse_index_with_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,abs_pos_list,
                          point_to_window_idx,
                          coors[:,1:],
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for pos_i in abs_pos_list:
            return_tuple.append(pos_i)
        for window_coor in window_coor_list:
            return_tuple.append(window_coor)
        for window_indice_i in window_indice_list:
            return_tuple.append(window_indice_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//6]
        dtype=grad_tensor_list[0].dtype
        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
        grad_voxel=torch.zeros((voxel_shape[0],voxel_shape[1]),dtype=dtype,device=voxel_shape.device)
        # print('grad type & len &shape', grad_tensor_list[0].dtype, len(a) // 4,grad_voxel.shape)
        # for i in range(len(a)//4):
        #      print(voxel_shape[1],i,torch.isnan(a[i]).sum()/voxel_shape[1]/a[i].shape[1],a[i].max())
        if dtype==torch.float32:
            sparse_index_backward_test(
            grad_voxel,
            grad_tensor_list,
            point_to_window_idx,
            window_indexes.to(dtype=torch.int32),
            window_regions.to(dtype=torch.int32),
            window_to_tensor_idx.to(dtype=torch.int32),
            tensor_lengthes,
            len(a) // 6
            )
        elif dtype==torch.float16:
            sparse_index_backward_half(
            grad_voxel,
            grad_tensor_list,
            point_to_window_idx,
            window_indexes.to(dtype=torch.int32),
            window_regions.to(dtype=torch.int32),
            window_to_tensor_idx.to(dtype=torch.int32),
            tensor_lengthes,
            len(a) // 6
            )
        return None,grad_voxel,None,None,None,None,None,None




class SparseSwinWoPosByOffset(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes,window_offset):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        points_index_list = []
        window_indice_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=coors.device,dtype=torch.int32)

            windows_partition = (coors+torch.tensor(window_offset,dtype=coors.dtype,device=coors.device))  // window_sizes[None, :]

            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)

            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            valid_length=[]
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                if region_in_tensor_index.sum() == 0:
                    continue
                window_indice_list.append(region_in_tensor_index)
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(coors.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(coors.new_zeros(region_length, length, dtype=torch.int32))
                count = count + 1

            tensor_lengthes = tensor_lengthes[valid_length]

        sparse_index_wo_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,
                          point_to_window_idx,
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for window_indice_i in window_indice_list:
            return_tuple.append(window_indice_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//4]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
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
        return None,grad_voxel,None,None,None,None,None,None


class FocalSparseIndexFunctionLongTestWoPos(Function):
    @staticmethod
    def forward(ctx,windows_partition:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,layer=None):
        with torch.no_grad():
            sparse_tensor_list = []
            sparse_tensor_index_list = []
            points_index_list = []

            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(voxel_features.shape[0], device=window_lengthes.device, dtype=torch.int32)

            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=voxel_features.device, dtype=torch.int32)
            feature_channel = voxel_features.shape[1]
            valid_length = []

            # w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                if region_in_tensor_index.sum() == 0:
                    continue
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(voxel_features.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(voxel_features.new_zeros(region_length, length, dtype=torch.int32))
                # print(sparse_tensor_list[-1].shape)
                # print(i,tensor_lengthes.shape[0])
                count = count + 1
                # if i==0:
                #     print(layer, inter_flag, 'range {} - {} :'.format(0, length), region_tensor.shape)
                # else:
                #     print(layer,inter_flag,'range {} - {} :'.format(tensor_lengthes[i-1], length), region_tensor.shape)
            # print()
            tensor_lengthes = tensor_lengthes[valid_length]
        sparse_index_wo_pos(sparse_tensor_list, sparse_tensor_index_list, points_index_list,
                          point_to_window_idx,
                          voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//3]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
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
            len(a) // 3
        )
        return None,grad_voxel,None,None,None


class FocalSparseIndexFunctionLongTestWithPos(Function):
    @staticmethod
    def forward(ctx,pos:torch.Tensor,windows_partition:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,layer=None):
        with torch.no_grad():
            sparse_tensor_list = []
            sparse_tensor_index_list = []
            pos_list = []
            points_index_list = []

            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(voxel_features.shape[0], device=window_lengthes.device, dtype=torch.int32)

            count = 0
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)
            tensor_lengthes = torch.tensor(tensor_lengthes, device=voxel_features.device, dtype=torch.int32)
            feature_channel = voxel_features.shape[1]
            valid_length = []

            # w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
            for i in range(tensor_lengthes.shape[0]):
                length = tensor_lengthes[i]
                if i == 0:
                    region_in_tensor_index = (window_lengthes <= length)
                else:
                    region_in_tensor_index = (window_lengthes > tensor_lengthes[i - 1]) & (window_lengthes <= length)
                if region_in_tensor_index.sum() == 0:
                    continue
                valid_length.append(i)
                window_regions[region_in_tensor_index] = count
                region_tensor = window_lengthes[region_in_tensor_index]
                region_length = region_tensor.shape[0]
                region_to_tensor_index = torch.arange(region_length, device=region_tensor.device)
                window_to_tensor_idx[region_in_tensor_index] = region_to_tensor_index
                sparse_tensor_list.append(voxel_features.new_zeros(region_length, length, feature_channel))
                sparse_tensor_index_list.append(voxel_features.new_zeros(region_length, length, dtype=torch.bool))
                points_index_list.append(voxel_features.new_zeros(region_length, length, dtype=torch.int32))
                pos_list.append(
                    voxel_features.new_zeros(region_length, length, 3, dtype=torch.int32))
                # print(sparse_tensor_list[-1].shape)
                # print(i,tensor_lengthes.shape[0])
                count = count + 1
                # if i==0:
                #     print(layer, inter_flag, 'range {} - {} :'.format(0, length), region_tensor.shape)
                # else:
                #     print(layer,inter_flag,'range {} - {} :'.format(tensor_lengthes[i-1], length), region_tensor.shape)
            # print()
            tensor_lengthes = tensor_lengthes[valid_length]
        sparse_index_test(sparse_tensor_list, sparse_tensor_index_list, points_index_list, pos_list,
                          point_to_window_idx,
                          pos.to(dtype=torch.int32), voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in pos_list:
            return_tuple.append(sparse_tensor_i)

        return_tuple=tuple(return_tuple)
        return return_tuple

    @staticmethod
    def backward(ctx,*args):
        a=args
        grad_tensor_list=a[:len(a)//4]

        point_to_window_idx, window_indexes, window_regions, window_to_tensor_idx, tensor_lengthes,voxel_shape=ctx.saved_tensors
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
        return None,None,grad_voxel,None,None,None
class SparseIndexFunctionLongTest_(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes,window_offset=None,inter_flag=False,layer=None):
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
            # print(sparse_tensor_list[-1].shape)
            #print(i,tensor_lengthes.shape[0])
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
        return None,grad_voxel,None,None,None,None,None,None

class SparseIndexFunctionLongTest(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,window_sizes,inter_flag):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        relative_pos_list = []
        points_index_list = []
        with torch.no_grad():
            feature_channel = voxel_features.shape[1]
            window_sizes = torch.tensor(window_sizes, device=coors.device)

            partition_mode = torch.tensor(0, device=voxel_features.device)
            if inter_flag:
                windows_partition = coors % window_sizes[None, :]
                partition_mode = torch.tensor(1, device=voxel_features.device)
                r_pos_coor=coors[:,1:]//window_sizes[None, 1:]
            else:
                windows_partition = coors // window_sizes[None, :]
                r_pos_coor = coors[:, 1:] % window_sizes[None, 1:]
            #print('partition mode', partition_mode)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)


            # print('test window_lengthes',window_lengthes.dtype)
            # window_indexes=window_indexes.to(dtype=torch.int32)
            # window_lengthes=window_lengthes.to(dtype=torch.int32)
            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            tensor_lengthes = torch.tensor((1,2,4,8,16,32,64,128,256,384,512), dtype=torch.int32,
                                             device=window_to_tensor_idx.device)

            count = 0
            #print('windows_partition', windows_partition.shape)
            #if partition_mode == 1:
            #    print(_[:10], window_lengthes.max())

            # print('windows num', window_lengthes.shape, 'voxel num', voxel_features.shape[0])
            voxel_shape = torch.tensor(voxel_features.shape, device=voxel_features.device)


        valid_length=[]
        #w_interval=torch.tensor((0,1,2,4,8,16,32,64,128,256,384,512),device=tensor_lengthes.device)
        for i in range(11):
            length=tensor_lengthes[i]
            if i==0:
                region_in_tensor_index = (window_lengthes <= length)
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

            #print('range {} - {} :'.format(pow(2, i - 1), pow(2, i)), region_tensor.shape)

        #print()
        tensor_lengthes=tensor_lengthes[valid_length]
        #print('tensor_lengthes',tensor_lengthes)
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
                w1 = (spatial_shape[0] // window_sizes[1]) + 1
                w2 = (spatial_shape[1] // window_sizes[2]) + 1
                w3 = (spatial_shape[2] // window_sizes[3]) + 1
            for i in range(len(relative_pos_list)):
                pos = relative_pos_list[i]
                pos = pos.permute(2, 0, 1)
                pos = pos[:, :, :, None] - pos[:, :, None, :]
                pos = pos.permute(1, 2, 3, 0).contiguous()
                pos[:, :, :, 0] += w1 - 1
                pos[:, :, :, 1] += w2 - 1
                pos[:, :, :, 2] += w3 - 1
                pos[:, :, :, 0] *= (2 * w3 - 1)*(2 * w2 - 1) # TODO: How to avoid different pos but same sum ? & require_grad=False ?
                pos[:, :, :, 1] *= 2 * w3 - 1
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
        #return sparse_tensor_list, sparse_tensor_index_list, points_index_list, relative_pos_list  # window_lengthes.shape[0]
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
        return None,grad_voxel,None,None,None




class SparseIndexFunctionLongTestWithAbsPos(Function):
    @staticmethod
    def forward(ctx,coors:torch.Tensor,voxel_features:torch.Tensor,spatial_shape,tensor_lengthes,window_sizes,window_offset=None,inter_flag=False,layer=None):
        sparse_tensor_list = []
        sparse_tensor_index_list = []
        abs_pos_list = []
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
                    # r_pos_coor = coors[:, 1:] //( window_sizes[None, 1:] * dilated_size[None,1:])

                else:
                    windows_partition = coors % window_sizes[None, :]
                    partition_mode = torch.tensor(1, device=voxel_features.device)
                    # r_pos_coor = coors[:, 1:] // window_sizes[None, 1:]

            else:
                if window_offset is not None:
                    window_offset = torch.tensor(window_offset, device=coors.device)
                    windows_partition = (coors+window_offset) // window_sizes[None, :]
                    # r_pos_coor = (coors+window_offset)[:, 1:] % window_sizes[None, 1:]
                else:
                    windows_partition = coors  // window_sizes[None, :]
                    # r_pos_coor = coors [:, 1:] % window_sizes[None, 1:]
            #print('partition mode', partition_mode)
            _, window_indexes, window_lengthes = torch.unique(windows_partition, sorted=True, return_inverse=True,
                                                              return_counts=True, dim=0)



            window_to_tensor_idx = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            window_regions = torch.zeros_like(window_lengthes, device=window_lengthes.device)
            point_to_window_idx = -torch.ones(coors.shape[0], device=window_lengthes.device, dtype=torch.int32)
            # tensor_lengthes = torch.tensor((1,2,4,8,16,32,64,128,256,384,512), dtype=torch.int32,
            #                                  device=window_to_tensor_idx.device)

            count = 0
            # print('windows_partition', windows_partition.shape)
            # if partition_mode == 1:
            # print(_[:10], window_lengthes.max())

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
            abs_pos_list.append(coors.new_zeros(region_length, length, 3, dtype=torch.int32,requires_grad=False))
            # print(sparse_tensor_list[-1].shape)
            # print(i,tensor_lengthes.shape[0])
            count=count+1
            # if i==0:
            #     print(layer, inter_flag, 'range {} - {} :'.format(0, length), region_tensor.shape)
            # else:
            #     print(layer,inter_flag,'range {} - {} :'.format(tensor_lengthes[i-1], length), region_tensor.shape)
        #print()
        tensor_lengthes=tensor_lengthes[valid_length]
        sparse_index_test(sparse_tensor_list, sparse_tensor_index_list, points_index_list, abs_pos_list,
                          point_to_window_idx,
                          coors[:,1:].contiguous().to(dtype=torch.int32), voxel_features,
                          window_indexes.to(dtype=torch.int32), window_regions.to(dtype=torch.int32),
                          window_to_tensor_idx.to(dtype=torch.int32), tensor_lengthes, 3,count )

        ctx.save_for_backward(point_to_window_idx,window_indexes,window_regions,window_to_tensor_idx,tensor_lengthes,voxel_shape,partition_mode)

        return_tuple=[]
        for sparse_tensor_i in sparse_tensor_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in sparse_tensor_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in points_index_list:
            return_tuple.append(sparse_tensor_i)
        for sparse_tensor_i in abs_pos_list:
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
        return None,grad_voxel,None,None,None,None,None,None


sparse_index_sample__=SparseIndexFunction.apply  # Ori

sparse_index_sample_=SparseIndexFunctionLongTest.apply  # with relative pos

sparse_index_sample=SparseIndexFunctionLongTest_.apply

sparse_index_sample_wo_pos=SparseIndexFunctionLongTestWoPos.apply

sparse_focal_index_sample_wo_ps=FocalSparseIndexFunctionLongTestWoPos.apply

sparse_index_sample_with_abs_pos=SparseIndexFunctionLongTestWithAbsPos.apply

sparse_focal_index_sample_with_pos=FocalSparseIndexFunctionLongTestWithPos.apply

sparse_focal_aware_index_sample_wo_pos=SparseFocalAwareWoPos.apply

sparse_swin_index_sample_wo_pos=SparseSwinWoPos.apply

sparse_swin_index_sample_wo_pos_by_offset=SparseSwinWoPosByOffset.apply


sparse_swin_index_with_abs_pos=SparseSwinWithAbsPos.apply