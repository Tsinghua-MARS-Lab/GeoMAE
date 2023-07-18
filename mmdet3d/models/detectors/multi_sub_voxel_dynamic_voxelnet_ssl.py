import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .voxelnet import VoxelNet
from ..builder import build_loss

from .. import builder
from mmdet3d.ops import Voxelization,Voxelization_with_flag
from .single_stage import SingleStage3DDetector
from spconv.pytorch.ops import get_indice_pairs,get_indice_pairs_implicit_gemm
from spconv.core import ConvAlgo
import copy

from mmdet3d.ops import points_in_boxes_cpu,points_in_boxes_gpu
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
# torch.set_printoptions(profile="full")
eps=1e-9
@DETECTORS.register_module()
class MultiSubVoxelDynamicVoxelNetSSL(SingleStage3DDetector):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """
    def __init__(self,
                 loss,
                 loss_ratio_low,
                 loss_ratio_med,
                 loss_ratio_top,
                 loss_ratio_low_nor,
                 loss_ratio_med_nor,
                 loss_ratio_top_nor,
                 hard_sub_voxel_layer_low,
                 hard_sub_voxel_layer_med,
                 hard_sub_voxel_layer_top,
                 random_mask_ratio,
                 grid_size,
                 sub_voxel_ratio_low,
                 sub_voxel_ratio_med,
                 voxel_layer,
                 sub_voxel_layer_low,
                 sub_voxel_layer_med,
                 voxel_encoder,
                 backbone,
                 spatial_shape=[1, 468, 468],
                 nor_usr_sml1=None,
                 cls_loss_ratio_low=None,
                 cls_loss_ratio_med=None,
                 vis=False,
                 cls_sub_voxel=False,
                 normalize_sub_voxel=None,
                 use_focal_mask=None,
                 norm_curv=True,
                 mse_loss=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultiSubVoxelDynamicVoxelNetSSL, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.spatial_shape=spatial_shape

        self.nor_usr_sml1=nor_usr_sml1
        self.norm_curv=norm_curv

        self.loss_ratio_med = loss_ratio_med
        self.loss_ratio_low = loss_ratio_low
        self.loss_ratio_top = loss_ratio_top

        self.loss_ratio_low_nor = loss_ratio_low_nor
        self.loss_ratio_med_nor=loss_ratio_med_nor
        self.loss_ratio_top_nor = loss_ratio_top_nor

        self.cls_loss_ratio_low = cls_loss_ratio_low
        self.cls_loss_ratio_med = cls_loss_ratio_med

        self.cls_sub_voxel = cls_sub_voxel
        self.vis=vis
        self.random_mask_ratio=random_mask_ratio
        self.point_cloud_range=voxel_layer['point_cloud_range']
        self.voxel_size=voxel_layer['voxel_size']
        self.grid_size = grid_size

        self.sub_voxel_size_low = sub_voxel_layer_low['voxel_size']
        self.sub_voxel_size_med = sub_voxel_layer_med['voxel_size']
        self.sub_voxel_ratio_low = sub_voxel_ratio_low
        self.sub_voxel_ratio_med = sub_voxel_ratio_med
        self.sub_voxel_layer_low = Voxelization(**sub_voxel_layer_low)
        self.sub_voxel_layer_med = Voxelization(**sub_voxel_layer_med)


        self.hard_sub_voxel_layer_low = Voxelization_with_flag(**hard_sub_voxel_layer_low)
        self.hard_sub_voxel_layer_med = Voxelization_with_flag(**hard_sub_voxel_layer_med)
        self.hard_sub_voxel_layer_top = Voxelization(**hard_sub_voxel_layer_top)

        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.reg_loss = build_loss(loss)

        self.mse_loss = mse_loss
        self.use_focal_mask = use_focal_mask
        self.normalize_sub_voxel=normalize_sub_voxel

        if cls_sub_voxel:
            self.cls_loss = build_loss(
                dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
                # dict(
                # type='FocalLoss',
                # use_sigmoid=True,
                # gamma=2.0,
                # alpha=0.25,
                # loss_weight=1.0))
        if self.nor_usr_sml1 is not None:
            self.nor_loss=build_loss(dict(type='SmoothL1Loss', reduction='mean', loss_weight=1.0))


    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """

        if self.cls_sub_voxel:
            x,centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high, \
            centroid_normal_low, centroid_normal_med, centroid_normal_high, \
                = self.extract_feat(points,gt_bboxes_3d,gt_labels_3d, img_metas)
            reg_pred_low, reg_pred_med,reg_pred_high,nor_pred_low,nor_pred_med,nor_pred_high,cls_pred_low, cls_pred_med = x

            loss = self.forward_loss(centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,centroid_normal_low,centroid_normal_med,centroid_normal_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high, nor_pred_low,nor_pred_med,nor_pred_high,cls_pred_low, cls_pred_med,)
        else:
            x,centroid_low,centroid_low_mask,centroid_med,centroid_med_mask,centroid_high,\
            centroid_normal_low,centroid_normal_med,centroid_normal_high, \
                = self.extract_feat(points, gt_bboxes_3d,gt_labels_3d, img_metas)

            reg_pred_low, reg_pred_med, reg_pred_high,nor_pred_low,nor_pred_med,nor_pred_high, = x

            loss = self.forward_loss(centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,centroid_normal_low,centroid_normal_med,centroid_normal_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high,nor_pred_low,nor_pred_med,nor_pred_high,)

        return loss


    def extract_feat(self, points: list,gt_bboxes_3d,gt_labels_3d, img_metas,vis=False):
        """Extract features from points."""

        batch_size=len(points)
        voxels, coors = self.voxelize(points)
        sub_coors_low = self.sub_voxelize_low(points)
        sub_coors_med = self.sub_voxelize_med(points)

        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)


        if self.use_focal_mask is not None:
            ids_keep, ids_mask = self.get_focal_mask_index(feature_coors, gt_bboxes_3d, gt_labels_3d)
        else:
            ids_keep,ids_mask=self.get_vanilla_mask_index(feature_coors,batch_size)

        centroids_low, centroid_voxel_coors_low, labels_count_low = self.get_centroid_per_voxel(voxels[:,[2,1,0]],sub_coors_low)
        centroids_med, centroid_voxel_coors_med, labels_count_med = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]],sub_coors_med)
        centroids_high, centroid_voxel_coors_high, labels_count_high = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]],coors)

        centroids_med_for_curv,centroid_mask_med_for_curv=self.get_multi_voxel_id_to_tensor_id_for_curv \
            (feature_coors.long(),centroid_voxel_coors_med.long(),centroids_med,batch_size)

        out_inds, indice_num_per_loc, pair, pair_bwd,           \
             pair_mask, pair_mask_bwd_splits,            \
             mask_argsort_fwd_splits, mask_argsort_bwd_splits, masks \
            = get_indice_pairs_implicit_gemm(
            indices=feature_coors,
            batch_size=batch_size,
            spatial_shape=self.spatial_shape,
            algo=ConvAlgo.MaskImplicitGemm,
            ksize=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            dilation=[1, 1, 1],
            out_padding=[0, 0, 0],
            subm=True,
            transpose=False,
            is_train=False)


        centroids_normal,centroids_curv=self.cal_regular_voxel_nor_and_curv(centroids_med_for_curv,centroid_mask_med_for_curv,centroids_high,pair.long())

        if self.normalize_sub_voxel is not None:
            centroids_low = self.normalize_centroid_sub_voxel(centroid_voxel_coors_low[:,1:],centroids_low,layer='low')
            centroids_med = self.normalize_centroid_sub_voxel(centroid_voxel_coors_med[:, 1:], centroids_med,layer='med')
            centroids_high = self.normalize_centroid_sub_voxel(centroid_voxel_coors_high[:, 1:], centroids_high,layer='top')

        centroids_low,centroid_mask_low,centroids_med,centroid_mask_med,\
            =self.get_multi_voxel_id_to_tensor_id_ori(feature_coors.long(),centroid_voxel_coors_low.long(),centroid_voxel_coors_med.long(),
                                                  centroids_low,centroids_med,ids_mask,batch_size)


        with torch.no_grad():
            centroids_high = centroids_high[ids_mask]
            centroids_normal = centroids_normal[ids_mask]
            centroids_curv = centroids_curv[ids_mask]


        centroids_normal_med=None
        centroids_normal_high=None

        mask_coors=feature_coors[ids_mask]
        if self.normalize_sub_voxel is None:
            centroids_low=self.normalize_centroid(mask_coors[:,1:],centroids_low)
            centroids_med = self.normalize_centroid(mask_coors[:, 1:], centroids_med)
            centroids_high = self.normalize_centroid(mask_coors[:, 1:], centroids_high)

        x = self.backbone(voxel_features[ids_keep],feature_coors[ids_keep],mask_coors,batch_size)

        if self.vis:
            return x, centroids_low, centroid_mask_low,centroids_med,centroid_mask_med,centroids_high,mask_coors[:,1:]
        else:
            return x, centroids_low, centroid_mask_low,centroids_med,centroid_mask_med,centroids_high,centroids_normal,centroids_normal_med,centroids_normal_high

    @torch.no_grad()
    def get_focal_mask_index(self,coors,gt_bboxes,gt_labels_3d):
        #TODO: this version is only a tricky implmentation of judging pillar in bboxes. Also having some error.
        batch_size=len(gt_bboxes)
        device=coors.device
        voxel_size=torch.tensor(self.voxel_size[:2],device=device)
        start_coors=torch.tensor(self.point_cloud_range[:2],device=device)

        ids_mask_list=[]
        ids_keep_list=[]
        previous_length=0
        for i in range(batch_size):
            inds = torch.where(coors[:, 0] == i)
            #print('inds',inds.shape,inds.dtype)
            coors_per_batch=coors[inds][:,[3,2]]*voxel_size+start_coors
            z_coors = torch.ones((coors_per_batch.shape[0],1),device=device)
            coors_per_batch=torch.cat([coors_per_batch,z_coors],dim=1)

            valid_index=gt_labels_3d[i]!=-1
            valid_gt_bboxes=gt_bboxes[i][valid_index]
            valid_gt_bboxes.tensor[:, 2] = 1
            valid_gt_bboxes.tensor[:, 5] = 2
            voxel_in_gt_bboxes=valid_gt_bboxes.points_in_boxes(coors_per_batch)

            fg_index = voxel_in_gt_bboxes!=-1
            fg_index = torch.nonzero(fg_index)
            bg_index = voxel_in_gt_bboxes==-1
            bg_index = torch.nonzero(bg_index)

            L = fg_index.shape[0]
            len_keep = int(L * (1 - self.random_mask_ratio))
            ids_shuffle = torch.randperm(L, device=device)
            ids_mask_list.append(fg_index[ids_shuffle[len_keep:]]+previous_length)
            ids_keep_list.append(fg_index[ids_shuffle[:len_keep]]+previous_length)
            ids_keep_list.append(bg_index+previous_length)
            previous_length+=coors_per_batch.shape[0]

        ids_keep_list = torch.cat(ids_keep_list).squeeze()
        ids_mask_list = torch.cat(ids_mask_list).squeeze()

        return ids_keep_list, ids_mask_list


    @torch.no_grad()
    def get_vanilla_mask_index(self,coors,batch_size):
        #TODO: this version is only a tricky implmentation of judging pillar in bboxes. Also having some error.
        device=coors.device

        ids_keep_list=[]
        ids_mask_list=[]
        for i in range(batch_size):
            inds = torch.where(coors[:, 0] == i)
            L=inds[0].shape[0]
            len_keep = int(L * (1 - self.random_mask_ratio))
            ids_shuffle=torch.randperm(L,device=device)
            ids_keep_list.append(inds[0][ids_shuffle[:len_keep]])
            ids_mask_list.append(inds[0][ids_shuffle[len_keep:]])

        ids_keep_list = torch.cat(ids_keep_list)
        ids_mask_list = torch.cat(ids_mask_list)
        return ids_keep_list,ids_mask_list


    @torch.no_grad()
    @force_fp32()
    def sub_voxelize_low(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i,res in enumerate(points):
            res_coors = self.sub_voxel_layer_low(res)
            coors.append(res_coors)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return coors_batch

    @torch.no_grad()
    @force_fp32()
    def sub_voxelize_med(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i,res in enumerate(points):
            res_coors = self.sub_voxel_layer_med(res)
            coors.append(res_coors)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return coors_batch

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    def map_voxel_centroid_id(self,voxel_coor,centriod_coor,voxel_size,point_cloud_range,batch_size):
        x_max = (point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]
        y_max = (point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]
        z_max = (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
        all_len=x_max*y_max*z_max
        y_len=y_max*x_max
        voxel_id=voxel_coor[:,0]*all_len + voxel_coor[:,1]*y_len + voxel_coor[:,2]*x_max + voxel_coor[:,3]
        centroid_id = centriod_coor[:, 0] * all_len + centriod_coor[:, 1] * y_len + centriod_coor[:, 2] * x_max + centriod_coor[:, 3]
        voxel_id = torch.sort(voxel_id)[1]
        centroid_id = torch.sort(centroid_id)[1]
        centroid_to_voxel=voxel_id.new_zeros(voxel_id.shape)
        voxel_to_centroid=voxel_id.new_zeros(voxel_id.shape)
        centroid_to_voxel[voxel_id]=centroid_id
        voxel_to_centroid[centroid_id]=voxel_id
        return centroid_to_voxel,voxel_to_centroid

    @torch.no_grad()
    def map_voxel_centroids_to_sub_voxel(self, voxel_coors,voxel_centroids, voxel_coors_low, voxel_coors_med,batch_size):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        sub_voxel_num_low=voxel_coors_low.shape[0]
        sub_voxel_num_med = voxel_coors_med.shape[0]
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        # per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        # per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        #
        # centroid_target_low = voxel_centroids.new_zeros((sub_voxel_num_low , 3))
        # centroid_target_med = voxel_centroids.new_zeros((sub_voxel_num_med , 3))


        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id

        tensor_id_low = voxel_coors_low[:, 0] * grid_shape + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * \
                        self.grid_size[1] + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        tensor_id_low = hash_table[tensor_id_low]
        centroid_target_low=voxel_centroids[tensor_id_low]



        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * \
                        self.grid_size[1] + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        tensor_id_med = hash_table[tensor_id_med]
        centroid_target_med = voxel_centroids[tensor_id_med]

        return centroid_target_low, centroid_target_med

    @torch.no_grad()
    def map_voxel_to_sub_voxel(self, voxel_coors,voxel_centroids, voxel_coors_low, voxel_coors_med,voxel_centroids_low,voxel_centroids_med,batch_size):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        sub_voxel_num_low=voxel_coors_low.shape[0]
        sub_voxel_num_med = voxel_coors_med.shape[0]
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        centroid_target_low = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_low , 3))
        centroid_target_med = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_med , 3))

        centroid_to_low = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_low , 3))
        centroid_to_med = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_med , 3))

        sub_voxel_low_grid_xy = self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        sub_voxel_med_grid_xy = self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id

        tensor_id_low = voxel_coors_low[:, 0] * grid_shape + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * \
                        self.grid_size[1] + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        tensor_id_low = hash_table[tensor_id_low]
        centroid_for_low=voxel_centroids[tensor_id_low]

        target_id_low = tensor_id_low * per_sub_voxel_num_low + \
                        (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0]) * sub_voxel_low_grid_xy  + \
                        (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1]) * self.sub_voxel_ratio_low[2] + \
                        voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_low = centroid_target_low.view(voxel_num , per_sub_voxel_num_low , 3)

        centroid_to_low[target_id_low] = centroid_for_low
        centroid_to_low = centroid_to_low.view(voxel_num , per_sub_voxel_num_low , 3)

        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * \
                        self.grid_size[1] + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        tensor_id_med = hash_table[tensor_id_med]
        centroid_for_med = voxel_centroids[tensor_id_med]

        tensor_id_med = tensor_id_med * per_sub_voxel_num_med + \
                        (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0]) * sub_voxel_med_grid_xy + \
                        voxel_coors_med[:,2] % self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2] + \
                        voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_med = centroid_target_med.view(voxel_num, per_sub_voxel_num_med, 3)

        centroid_to_med[tensor_id_med] = centroid_for_med
        centroid_to_med = centroid_to_med.view(voxel_num , per_sub_voxel_num_med , 3)

        return centroid_to_low,centroid_to_med,centroid_target_low, centroid_target_med


    @torch.no_grad()
    def map_voxel_and_center_to_sub_voxel(self, voxel_coors,voxel_centroids, voxel_coors_low, voxel_coors_med,voxel_centroids_low,voxel_centroids_med,batch_size):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar


        sub_voxel_low_grid_xy = self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        sub_voxel_med_grid_xy = self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        centroid_target_low = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_low , 3))
        centroid_target_med = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_med , 3))

        centroid_to_low = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_low , 3))
        centroid_to_med = voxel_centroids.new_zeros((voxel_num * per_sub_voxel_num_med , 3))


        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id

        tensor_id_low = voxel_coors_low[:, 0] * grid_shape + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * \
                        self.grid_size[1] + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        tensor_id_low = hash_table[tensor_id_low]
        centroid_for_low = voxel_centroids[tensor_id_low]

        target_id_low = tensor_id_low * per_sub_voxel_num_low + \
                        (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0]) * sub_voxel_low_grid_xy  + \
                        (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1]) * self.sub_voxel_ratio_low[2] + \
                        voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_low = centroid_target_low.view(voxel_num , per_sub_voxel_num_low , 3)

        centroid_to_low[target_id_low] = centroid_for_low
        centroid_to_low = centroid_to_low.view(voxel_num , per_sub_voxel_num_low , 3)

        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * \
                        self.grid_size[1] + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        tensor_id_med = hash_table[tensor_id_med]
        centroid_for_med = voxel_centroids[tensor_id_med]

        tensor_id_med = tensor_id_med * per_sub_voxel_num_med + \
                        (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0]) * sub_voxel_med_grid_xy + \
                        voxel_coors_med[:,2] % self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2] + \
                        voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_med = centroid_target_med.view(voxel_num, per_sub_voxel_num_med, 3)

        centroid_to_med[tensor_id_med] = centroid_for_med
        centroid_to_med = centroid_to_med.view(voxel_num , per_sub_voxel_num_med , 3)

        return centroid_to_low,centroid_to_med,centroid_target_low, centroid_target_med

    @torch.no_grad()
    @force_fp32()
    def cal_voxel_curv(self,voxel_points,voxel_flag,centroids,centroid_to_voxel_id,voxel_to_centroid_id):
        N,max_points,_=voxel_points.shape
        voxel_cetroids=centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1,max_points,1)
        voxel_cetroids[~voxel_flag]=0
        voxel_points=voxel_points-voxel_cetroids

        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal = torch.svd(cov)[2][..., -1]
        est_normal = est_normal/torch.norm(est_normal,p=2, dim=-1, keepdim=True)
        return est_normal[voxel_to_centroid_id]

    @torch.no_grad()
    @force_fp32()
    def cal_regular_voxel_curv(self,centroid_for_low,centroid_for_med,centroid_target_low, centroid_target_med):
        # N,max_points,_=voxel_points.shape
        # voxel_cetroids=centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1,max_points,1)
        # voxel_cetroids[~voxel_flag]=0
        voxel_points=centroid_target_low-centroid_for_low
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal_low = torch.svd(cov)[2][..., -1]
        est_normal_low = est_normal_low/torch.norm(est_normal_low,p=2, dim=-1, keepdim=True)

        voxel_points=centroid_target_med-centroid_for_med
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        est_normal_med = torch.svd(cov)[2][..., -1]
        est_normal_med = est_normal_med/torch.norm(est_normal_med,p=2, dim=-1, keepdim=True)

        return est_normal_low,est_normal_med


    @torch.no_grad()
    @force_fp32()
    def cal_regular_voxel_nor_and_curv(self,centroid_low,centroid_low_mask,voxel_centroid,indice_pairs):
        # N,max_points,_=voxel_points.shape
        # voxel_cetroids=centroids[centroid_to_voxel_id].unsqueeze(dim=1).repeat(1,max_points,1)
        # voxel_cetroids[~voxel_flag]=0
        sub_voxel_num=centroid_low.shape[1]
        around_num,voxel_num=indice_pairs.shape
        around_mask=(indice_pairs==-1)
        centroid_low_around = centroid_low[indice_pairs]
        centroid_low_mask_around = centroid_low_mask[indice_pairs]

        centroid_low_around[around_mask]=0
        centroid_low_mask_around[around_mask]=False

        centroid_low_around=centroid_low_around.transpose(0,1).contiguous().view(voxel_num,-1,3)
        centroid_low_mask_around=centroid_low_mask_around.transpose(0,1).contiguous().view(voxel_num,-1)

        voxel_centroid_around = voxel_centroid.unsqueeze(dim=1).repeat(1, sub_voxel_num * around_num, 1)

        voxel_centroid_around[~centroid_low_mask_around]=0
        voxel_points=centroid_low_around - voxel_centroid_around
        cov = voxel_points.transpose(-2, -1) @ voxel_points
        svd = torch.svd(cov)
        est_normal = svd[2][..., -1]

        if self.norm_curv:
            est_normal = est_normal/torch.norm(est_normal,p=2, dim=-1, keepdim=True)

        est_curv = svd[1].to(dtype=torch.float64)
        est_curv = est_curv+eps

        est_curv = est_curv / est_curv.sum(dim=-1,keepdim=True)


        return est_normal,est_curv




    @torch.no_grad()
    def normalize_centroid(self,coors,centroids):
        device=coors.device
        voxel_size=torch.tensor(self.voxel_size[::-1],device=device)
        start_coors=torch.tensor(self.point_cloud_range[:3][::-1],device=device)
        #print('test shape',coors.shape,voxel_size.shape,start_coors.shape)
        coors_ = coors * voxel_size + start_coors
        centroids=(centroids-coors_.unsqueeze(dim=1))/voxel_size
        return centroids


    @torch.no_grad()
    def normalize_centroid_sub_voxel(self, coors, centroids,layer=None):

        device = coors.device
        if layer=='low':
            voxel_size = torch.tensor(self.sub_voxel_size_low[::-1], device=device)
        elif layer=='med':
            voxel_size = torch.tensor(self.sub_voxel_size_med[::-1], device=device)
        else:
            voxel_size = torch.tensor(self.voxel_size[::-1], device=device)

        start_coors = torch.tensor(self.point_cloud_range[:3][::-1], device=device)
        # print('test shape',coors.shape,voxel_size.shape,start_coors.shape)
        coors = coors * voxel_size + start_coors
        centroids = (centroids - coors) / voxel_size
        return centroids

    @torch.no_grad()
    def get_multi_voxel_id_to_tensor_id_for_curv(self, voxel_coors,voxel_coors_med,voxel_centroids_med, batch_size, ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        centroid_target_med = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num_med, 3), dtype=torch.float32)
        centroid_target_mask_med = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num_med, dtype=torch.bool)

        sub_voxel_med_grid_xy = self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)

        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id
        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * \
                    self.grid_size[1] + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        tensor_id_med = hash_table[tensor_id_med]
        tensor_id_med = tensor_id_med * per_sub_voxel_num_med + \
                        (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0]) * sub_voxel_med_grid_xy + \
                        voxel_coors_med[:,2] % self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2] + \
                        voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_mask_med[tensor_id_med] = True
        centroid_target_med = centroid_target_med.view(voxel_num, per_sub_voxel_num_med, 3)
        centroid_target_mask_med= centroid_target_mask_med.view(voxel_num, per_sub_voxel_num_med)

        return centroid_target_med,centroid_target_mask_med

    @torch.no_grad()
    def get_multi_voxel_id_to_tensor_id_ori(self, voxel_coors,voxel_coors_low,voxel_coors_med,voxel_centroids_low,voxel_centroids_med,ids_masked, batch_size, ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        centroid_target_low = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num_low, 3), dtype=torch.float32)
        centroid_target_mask_low = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num_low, dtype=torch.bool)

        centroid_target_med = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num_med, 3), dtype=torch.float32)
        centroid_target_mask_med = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num_med, dtype=torch.bool)

        sub_voxel_low_grid_xy = self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        sub_voxel_med_grid_xy = self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]


        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id

        tensor_id_low = voxel_coors_low[:, 0] * grid_shape + voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * \
                    self.grid_size[1] + voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        tensor_id_low = hash_table[tensor_id_low]
        target_id_low = tensor_id_low * per_sub_voxel_num_low + \
                        (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0]) * sub_voxel_low_grid_xy  + \
                        (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1]) * self.sub_voxel_ratio_low[2] + \
                        voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_mask_low[target_id_low] = True
        centroid_target_low = centroid_target_low.view(voxel_num, per_sub_voxel_num_low, 3)[ids_masked]
        centroid_target_mask_low = centroid_target_mask_low.view(voxel_num, per_sub_voxel_num_low)[ids_masked]



        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * \
                    self.grid_size[1] + voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]
        tensor_id_med = hash_table[tensor_id_med]
        tensor_id_med = tensor_id_med * per_sub_voxel_num_med + \
                        (voxel_coors_med[:, 1] % self.sub_voxel_ratio_med[0]) * sub_voxel_med_grid_xy + \
                        voxel_coors_med[:,2] % self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2] + \
                        voxel_coors_med[:, 3] % self.sub_voxel_ratio_med[2]
        centroid_target_med[tensor_id_med] = voxel_centroids_med
        centroid_target_mask_med[tensor_id_med] = True
        centroid_target_med = centroid_target_med.view(voxel_num, per_sub_voxel_num_med, 3)[ids_masked]
        centroid_target_mask_med= centroid_target_mask_med.view(voxel_num, per_sub_voxel_num_med)[ids_masked]

        return centroid_target_low, centroid_target_mask_low,centroid_target_med,centroid_target_mask_med



    @torch.no_grad()
    @force_fp32()
    def get_centroid_per_voxel(self,points:torch.Tensor, voxel_idxs:torch.Tensor, num_points_in_voxel=None):
        """
        Args:
            points: (N, 3 + (f)) [bxyz + (f)]
            voxel_idxs: (N, 4) [bxyz]
            num_points_in_voxel: (N)
        Returns:
            centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel
            centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids
            labels_count: (N') Number of points in each voxel
        """
        assert points.shape[0] == voxel_idxs.shape[0]
        voxel_idxs_valid_mask = (voxel_idxs>=0).all(-1)

        # print('non zero test',voxel_idxs.shape,voxel_idxs_valid_mask.sum())

        voxel_idxs = voxel_idxs[voxel_idxs_valid_mask]


        points=points[voxel_idxs_valid_mask]

        centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(dim=0,sorted=False, return_inverse=True,
                                                                           return_counts=True)

        unique_idxs = unique_idxs.view(unique_idxs.size(0), 1).expand(-1, points.size(-1))

        # Scatter add points based on unique voxel idxs
        if num_points_in_voxel is not None:
            centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs,
                                                                    points * num_points_in_voxel.unsqueeze(-1))
            num_points_in_centroids = torch.zeros((centroid_voxel_idxs.shape[0]), device=points.device,
                                                  dtype=torch.int64).scatter_add_(0, unique_idxs[:, 0],
                                                                                  num_points_in_voxel)
            centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
        else:
            centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs, points)
            centroids = centroids / labels_count.float().unsqueeze(-1)

        return centroids, centroid_voxel_idxs, labels_count

    @torch.no_grad()
    @force_fp32()
    def get_centroid_and_normal_per_voxel(self,points:torch.Tensor, voxel_idxs:torch.Tensor, num_points_in_voxel=None):
        """
        Args:
            points: (N, 3 + (f)) [bxyz + (f)]
            voxel_idxs: (N, 4) [bxyz]
            num_points_in_voxel: (N)
        Returns:
            centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel
            centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids
            labels_count: (N') Number of points in each voxel
        """
        assert points.shape[0] == voxel_idxs.shape[0]
        voxel_idxs_valid_mask = (voxel_idxs>=0).all(-1)

        # print('non zero test',voxel_idxs.shape,voxel_idxs_valid_mask.sum())

        voxel_idxs = voxel_idxs[voxel_idxs_valid_mask]


        points=points[voxel_idxs_valid_mask]

        centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(dim=0,sorted=True, return_inverse=True,
                                                                           return_counts=True)
        N,C=points.shape
        ori_id = torch.arange(N)
        ori_id_reverse = torch.flip(ori_id, dims=[0])
        inverse_unique_idxs= torch.flip(unique_idxs, dims=[0])


        unique_idxs_ = unique_idxs.view(unique_idxs.size(0), 1).expand(-1, C).clone()

        # Scatter add points based on unique voxel idxs
        if num_points_in_voxel is not None:
            centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs_,
                                                                    points * num_points_in_voxel.unsqueeze(-1))
            num_points_in_centroids = torch.zeros((centroid_voxel_idxs.shape[0]), device=points.device,
                                                  dtype=torch.int64).scatter_add_(0, unique_idxs_[:, 0],
                                                                                  num_points_in_voxel)
            centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
        else:
            centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs_, points)
            centroids = centroids / labels_count.float().unsqueeze(-1)
        points_centroids = centroids[unique_idxs]
        sort_idx=torch.sort(unique_idxs)[1]
        sort_idx_inverse=ori_id_reverse[torch.sort(inverse_unique_idxs)[1]]
        edge_vec1=points[sort_idx] - points_centroids[sort_idx]
        edge_vec2=points[sort_idx_inverse] - points_centroids[sort_idx_inverse]
        nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
        nor_len=torch.norm(nor, dim=-1, keepdim=True)
        nor_len[nor_len==0]=1
        nor = nor / nor_len

        unique_idxs = unique_idxs[sort_idx].view(unique_idxs.size(0), 1).expand(-1, C)
        sur_nor = torch.zeros((centroid_voxel_idxs.shape[0], nor.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs, nor)
        # nor_len=torch.norm(sur_nor, dim=-1, keepdim=True)
        # nor_len[nor_len == 0] = 1
        # sur_nor = sur_nor / nor_len
        sur_nor = sur_nor / labels_count.float().unsqueeze(-1)
        return centroids,sur_nor, centroid_voxel_idxs, labels_count



    @force_fp32(apply_to=('reg_pred', 'centroid_target','cls_pred'))
    def forward_loss(self,centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,centroid_normal_low,centroid_normal_med,centroid_normal_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high,nor_pred_low,nor_pred_med,nor_pred_high,cls_pred_low=None, cls_pred_med=None):
        centroid_low_mask=centroid_low_mask.view(-1)
        centroid_low=centroid_low.view(-1,3)[centroid_low_mask]

        reg_pred_low = reg_pred_low.view(-1,3)[centroid_low_mask]


        centroid_med_mask=centroid_med_mask.view(-1)
        centroid_med=centroid_med.view(-1,3)[centroid_med_mask]

        reg_pred_med = reg_pred_med.view(-1,3)[centroid_med_mask]


        if self.mse_loss:
            loss_reg_low = (reg_pred_low - centroid_low) ** 2
            loss_reg_low = loss_reg_low.mean(dim=-1)
            loss_reg_low = loss_reg_low.sum() / loss_reg_low.shape[0] * self.loss_ratio_low

            loss_reg_med = (reg_pred_med - centroid_med) ** 2
            loss_reg_med = loss_reg_med.mean(dim=-1)
            loss_reg_med = loss_reg_med.sum() / loss_reg_med.shape[0] * self.loss_ratio_med

            loss_reg_top = (reg_pred_high - centroid_high) ** 2
            loss_reg_top = loss_reg_top.mean(dim=-1)
            loss_reg_top = loss_reg_top.sum() / loss_reg_top.shape[0] * self.loss_ratio_top


            if self.nor_usr_sml1 is None:
                if nor_pred_low is None and nor_pred_med is None:
                    loss_nor_low = (nor_pred_high - centroid_normal_low) ** 2
                    loss_nor_low = loss_nor_low.mean(dim=-1)
                    loss_nor_low = loss_nor_low.sum() / loss_nor_low.shape[0] * self.loss_ratio_low_nor
                else:
                    loss_nor_low = (nor_pred_low - centroid_normal_low) ** 2
                    loss_nor_low = loss_nor_low.mean(dim=-1)
                    loss_nor_low = loss_nor_low.sum() / loss_nor_low.shape[0] * self.loss_ratio_low_nor


            else:

                if nor_pred_low is None and nor_pred_med is None:
                    loss_nor_low = self.nor_loss(nor_pred_high,centroid_normal_low) * self.loss_ratio_low_nor
                else:
                    loss_nor_low = self.nor_loss(nor_pred_low, centroid_normal_low) * self.loss_ratio_low_nor

        else:
            loss_reg_low = self.reg_loss(reg_pred_low, centroid_low) * self.loss_ratio_low
            loss_reg_med = self.reg_loss(reg_pred_med, centroid_med) * self.loss_ratio_med
            loss_reg_top = self.reg_loss(reg_pred_high, centroid_high) * self.loss_ratio_top

        if self.cls_sub_voxel:
            assert cls_pred_low is not None
            cls_pred_low = cls_pred_low.view(-1, 2)
            cls_pred_med = cls_pred_med.view(-1, 2)

            loss_cls_low = self.cls_loss(cls_pred_low, centroid_low_mask.long())*self.cls_loss_ratio_low
            loss_cls_med = self.cls_loss(cls_pred_med, centroid_med_mask.long())*self.cls_loss_ratio_med
            loss = dict(loss_curv_around=loss_nor_low,loss_centroid_low=loss_reg_low,loss_centroid_med=loss_reg_med,loss_centroid_top=loss_reg_top,
                      loss_cls_low=loss_cls_low,loss_cls_med=loss_cls_med)

        else:
            loss = dict(loss_centroid_low=loss_reg_low,loss_centroid_med=loss_reg_med,loss_centroid_top=loss_reg_top,
                        loss_nor_low=loss_nor_low,)
        return loss




    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)
        return feats



    @torch.no_grad()
    def cal_center_loss(self,centroid_target,centroid_target_mask,x):
        centroid_target_mask=centroid_target_mask.view(-1)
        centroid_target=centroid_target.view(-1,3)[centroid_target_mask]
        x=x.view(-1,3)[centroid_target_mask]
        center = x.new_ones(x.shape)*0.5
        centroid_loss = self.loss(x,centroid_target) * self.loss_ratio
        center_loss = self.loss(x,center) * self.loss_ratio
        center_centroid_loss = self.loss(center,centroid_target) * self.loss_ratio
        print('centroid_loss:',centroid_loss,'center_loss:',center_loss,'center_centroid_loss:',center_centroid_loss,)


    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        #print(type(img_metas),type(imgs),type(rescale),type(outs))
        if self.centerpoint_head:
            bbox_list = self.bbox_head.get_bboxes(
                outs, img_metas=img_metas, rescale=rescale)
        else:

            bbox_list = self.bbox_head.get_bboxes(
                    *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        return bbox_results
