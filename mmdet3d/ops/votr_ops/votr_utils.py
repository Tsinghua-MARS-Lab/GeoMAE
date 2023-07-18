import torch
from torch.autograd import Function, Variable

from . import votr_ops_cuda as votr

class BuildTensorTable(Function):

    @staticmethod
    def forward(ctx, batch_size, spatial_shape, voxel_indices, v_bs_cnt):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        x_max, y_max, z_max = spatial_shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, x_max, y_max, z_max)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        votr.build_mapping_with_tensor_wrapper(x_max, y_max, z_max, num_voxels, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_tensor_table = BuildTensorTable.apply

class BuildHashTable(Function):

    @staticmethod
    def forward(ctx, batch_size, hash_size, spatial_shape, voxel_indices, v_bs_cnt):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        z_max, y_max,x_max  = spatial_shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        votr.build_mapping_with_hash_wrapper(x_max, y_max, z_max, num_voxels, hash_size, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_hash_table = BuildHashTable.apply

class BuildHashTableTest(Function):

    @staticmethod
    def forward(ctx, batch_size, hash_size, spatial_shape, voxel_indices, v_bs_cnt):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        z_max, y_max,x_max  = spatial_shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        dense_map = torch.zeros((batch_size, hash_size, 2)).int().fill_(-1)
        dense_map = dense_map.to(voxel_indices.device)

        votr.build_mapping_with_hash_test_wrapper(x_max, y_max, z_max, num_voxels, hash_size, voxel_indices, v_bs_cnt, dense_map)
        return dense_map

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

build_hash_table_test = BuildHashTableTest.apply


class SparseStridedAttentionTensorIndices(Function):

    @staticmethod
    def forward(ctx, attend_size, range_spec, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, x_max, y_max, z_max) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        batch_size, x_max, y_max, z_max = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_strided_attention_with_tensor_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                            num_voxels, attend_size, num_range,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

window_attention_tensor_indices = SparseStridedAttentionTensorIndices.apply

class SparseStridedAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, range_spec, strides, dense_map, voxel_indices):
        """
        Args:
            ctx:
            dense_map: (bs_idx, x_max, y_max, z_max) -> old map table
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x) -> new downsampled indices
        Returns:
        """
        x_stride, y_stride, z_stride = strides
        x_max, y_max, z_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()
        range_spec = torch.tensor(range_spec).int().to(voxel_indices.device)
        num_range = range_spec.shape[0]
        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.sparse_strided_attention_with_hash_wrapper(x_max, y_max, z_max, x_stride, y_stride, z_stride,
                                                            num_voxels, attend_size, num_range, hash_size,
                                                            attend_indices, voxel_indices,
                                                            dense_map, range_spec)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

#window_attention_hash_indices = SparseStridedAttentionHashIndices.apply


class FocalAttentionHashIndices(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, attend_range,choose_indices, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        z_max, y_max, x_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = torch.zeros((num_voxels, attend_size)).int().fill_(-1).to(voxel_indices.device)

        votr.window_focal_attention_with_hash_wrapper(x_max, y_max, z_max,
                                                        num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices,choose_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

window_focal_attention_hash_indices = FocalAttentionHashIndices.apply


class FocalAttentionHashIndicesTest(Function):

    @staticmethod
    def forward(ctx, spatial_shape, attend_size, attend_range,choose_indices, dense_map, voxel_indices):
        """
        Args:
            ctx:
            voxel_indices: (num_voxels, 4) (bs_idx, z, y, x)
        Returns:
        """
        z_max, y_max, x_max = spatial_shape
        batch_size, hash_size, _ = dense_map.shape
        num_voxels = voxel_indices.shape[0]
        assert voxel_indices.is_contiguous()

        attend_indices = -torch.ones((num_voxels, attend_size)).int().to(voxel_indices.device)

        votr.window_focal_attention_with_hash_test_wrapper(x_max, y_max, z_max,
                                                        num_voxels, attend_size, attend_range, hash_size,
                                                        attend_indices,choose_indices, voxel_indices, dense_map)
        return attend_indices

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

window_focal_attention_hash_indices_test = FocalAttentionHashIndicesTest.apply
class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, features_batch_cnt: torch.Tensor,
                idx: torch.Tensor, idx_batch_cnt: torch.Tensor):
        """
        Args:
            ctx:
            features: (N1 + N2 ..., C) tensor of features to group
            features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
            idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
            idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with

        Returns:
            output: (M1 + M2, C, nsample) tensor
        """
        assert features.is_contiguous()
        assert features_batch_cnt.is_contiguous()
        assert idx.is_contiguous()
        assert idx_batch_cnt.is_contiguous()

        assert features.shape[0] == features_batch_cnt.sum(), \
            'features: %s, features_batch_cnt: %s' % (str(features.shape), str(features_batch_cnt))
        assert idx.shape[0] == idx_batch_cnt.sum(), \
            'idx: %s, idx_batch_cnt: %s' % (str(idx.shape), str(idx_batch_cnt))

        M, nsample = idx.size()
        N, C = features.size()
        B = idx_batch_cnt.shape[0]
        output = torch.cuda.FloatTensor(M, C, nsample).zero_()

        votr.group_features_wrapper(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, output)

        ctx.for_backwards = (B, N, idx, features_batch_cnt, idx_batch_cnt)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            ctx:
            grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward

        Returns:
            grad_features: (N1 + N2 ..., C) gradient of the features
        """
        B, N, idx, features_batch_cnt, idx_batch_cnt = ctx.for_backwards

        M, C, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(N, C).zero_())

        grad_out_data = grad_out.data.contiguous()
        votr.group_features_grad_wrapper(B, M, C, N, nsample, grad_out_data, idx,
                                            idx_batch_cnt, features_batch_cnt, grad_features.data)
        return grad_features, None, None, None

grouping_operation = GroupingOperation.apply