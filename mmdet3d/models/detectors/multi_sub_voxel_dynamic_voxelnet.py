import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .voxelnet import VoxelNet
from ..builder import build_loss

from .. import builder
from mmdet3d.ops import Voxelization
from .single_stage import SingleStage3DDetector
import copy

from mmdet3d.ops import points_in_boxes_cpu,points_in_boxes_gpu

@DETECTORS.register_module()
class MultiSubVoxelDynamicVoxelNet(SingleStage3DDetector):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """
    def __init__(self,
                 loss,
                 loss_ratio_low,
                 loss_ratio_med,
                 loss_ratio_top,
                 random_mask_ratio,
                 grid_size,
                 sub_voxel_ratio_low,
                 sub_voxel_ratio_med,
                 voxel_layer,
                 sub_voxel_layer_low,
                 sub_voxel_layer_med,
                 voxel_encoder,
                 backbone,
                 cls_loss_ratio_low=None,
                 cls_loss_ratio_med=None,
                 vis=False,
                 cls_sub_voxel=False,
                 use_focal_cls=False,
                 normalize_sub_voxel=None,
                 use_focal_mask=None,
                 mse_loss=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MultiSubVoxelDynamicVoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.loss_ratio_med = loss_ratio_med
        self.loss_ratio_low = loss_ratio_low
        self.loss_ratio_top = loss_ratio_top
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

        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.reg_loss = build_loss(loss)

        self.mse_loss = mse_loss
        self.use_focal_mask = use_focal_mask
        self.normalize_sub_voxel=normalize_sub_voxel
        if cls_sub_voxel:
            if use_focal_cls:
                self.cls_loss = build_loss(
                    dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0))
            else:
                self.cls_loss = build_loss(
                    dict(
                        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
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
            x,centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high = \
                self.extract_feat(points,gt_bboxes_3d,gt_labels_3d, img_metas)
            reg_pred_low, reg_pred_med,reg_pred_high,cls_pred_low, cls_pred_med = x

            # print(centroid_target_mask.shape[0]*centroid_target_mask.shape[1]/centroid_target_mask.sum())
            loss = self.forward_loss(centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high,cls_pred_low, cls_pred_med)
        else:
            x,centroid_low,centroid_low_mask,centroid_med,centroid_med_mask,centroid_high = self.extract_feat(points, gt_bboxes_3d,
                                                                                          gt_labels_3d, img_metas)

            reg_pred_low, reg_pred_med, reg_pred_high = x

            loss = self.forward_loss(centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high)
        #self.cal_center_loss(centroid_target,centroid_target_mask,x)



        return loss


    def extract_feat(self, points: list,gt_bboxes_3d,gt_labels_3d, img_metas,vis=False):
        """Extract features from points."""
        #print('points',len(points),points[0].shape)
        batch_size=len(points)
        voxels, coors = self.voxelize(points)
        sub_coors_low = self.sub_voxelize_low(points)
        sub_coors_med = self.sub_voxelize_med(points)


        # print('voxels',voxels.shape,coors[:,1].max(),coors[:,2].max(),coors[:,3].max(),
        #       ' sub_voxels',sub_coors[:,1].max(),sub_coors[:,2].max(),sub_coors[:,3].max())


        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)


        if self.use_focal_mask is not None:
            ids_keep, ids_mask = self.get_focal_mask_index(feature_coors, gt_bboxes_3d, gt_labels_3d)
        else:
            ids_keep,ids_mask=self.get_vanilla_mask_index(feature_coors,batch_size)

        #print(feature_coors.shape,ids_keep.shape)

        #print('sub_coors',sub_coors[:,1].max(),sub_coors[:,2].max(),sub_coors[:,3].max())
        centroids_low, centroid_voxel_coors_low, labels_count_low=self.get_centroid_per_voxel(voxels[:,[2,1,0]],sub_coors_low)
        #print('labels_count_low',labels_count_low.max())
        centroids_med, centroid_voxel_coors_med, labels_count_med = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]],sub_coors_med)
        centroids_high, centroid_voxel_coors_high, labels_count_high = self.get_centroid_per_voxel(voxels[:, [2, 1, 0]],coors)

        # if vis:
        #     self.vis_centroid(voxels,centroids[:,[2,1,0]],ids_keep,ids_mask)

        if self.normalize_sub_voxel is not None:
            centroids_low = self.normalize_centroid_sub_voxel(centroid_voxel_coors_low[:,1:],centroids_low,layer='low')
            centroids_med = self.normalize_centroid_sub_voxel(centroid_voxel_coors_med[:, 1:], centroids_med,layer='med')
            centroids_high = self.normalize_centroid_sub_voxel(centroid_voxel_coors_high[:, 1:], centroids_high,layer='top')


        # print('test',feature_coors.shape,centroid_voxel_coors.shape,feature_coors[0],centroid_voxel_coors[0])
        #
        centroids_low,centroid_mask_low,centroids_med,centroid_mask_med,\
            =self.get_multi_voxel_id_to_tensor_id(feature_coors.long(),centroid_voxel_coors_low.long(),centroid_voxel_coors_med.long(),
                                                  centroids_low,centroids_med,ids_mask,batch_size)

        centroids_high=centroids_high[ids_mask]


        mask_coors=feature_coors[ids_mask]
        if self.normalize_sub_voxel is None:
            centroids_low=self.normalize_centroid(mask_coors[:,1:],centroids_low)
            centroids_med = self.normalize_centroid(mask_coors[:, 1:], centroids_med)
            centroids_high = self.normalize_centroid(mask_coors[:, 1:], centroids_high)





        x = self.backbone(voxel_features[ids_keep],feature_coors[ids_keep],mask_coors,batch_size)

        if self.vis:
            return x, centroids_low, centroid_mask_low,centroids_med,centroid_mask_med,centroids_high,mask_coors[:,1:]
        else:
            return x, centroids_low, centroid_mask_low,centroids_med,centroid_mask_med,centroids_high

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
            # print('gt_bboxes[i]',gt_bboxes[i])
            # voxel_in_gt_bboxes=points_in_boxes_gpu(coors_per_batch.unsqueeze(0),gt_bboxes[i].tensor.unsqueeze(0).to(device=device))
            #voxel_in_gt_bboxes+=1
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

            #print('inds', inds[0][:10],previous_length)
            #print(coors_per_batch.shape,voxel_in_gt_bboxes.shape,test_index.sum(),voxel_in_gt_bboxes[test_index][:10])
            #print(coors.shape,voxel_in_gt_bboxes.shape,voxel_in_gt_bboxes.dtype,voxel_in_gt_bboxes[0,:10])

        ids_keep_list = torch.cat(ids_keep_list).squeeze()
        ids_mask_list = torch.cat(ids_mask_list).squeeze()
        #print('ids_mask_list',ids_mask_list.shape)

        #print('test focal',len(ids_keep_list),len(ids_mask_list),coors.shape)
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


    # def voxelize(self, points):
    #     """Apply dynamic voxelization to points.
    #
    #     Args:
    #         points (list[torch.Tensor]): Points of each sample.
    #
    #     Returns:
    #         tuple[torch.Tensor]: Concatenated points and coordinates.
    #     """
    #
    #     coors_batch = []
    #
    #     # dynamic voxelization only provide a coors mapping
    #     for i,res in enumerate(points):
    #         res_coors = self.sub_voxel_layer(res)
    #         res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
    #         coors_batch.append(res_coors)
    #
    #     coors_batch = torch.cat(coors_batch, dim=0)
    #     points = torch.cat(points, dim=0)
    #
    #     return points,coors_batch

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
    def get_multi_voxel_id_to_tensor_id(self, voxel_coors,voxel_coors_low,voxel_coors_med,voxel_centroids_low,voxel_centroids_med,ids_masked, batch_size, ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num_low = self.sub_voxel_ratio_low[0] * self.sub_voxel_ratio_low[1] * self.sub_voxel_ratio_low[2]
        per_sub_voxel_num_med = self.sub_voxel_ratio_med[0] * self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]

        centroid_target_low = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num_low, 3), dtype=torch.float32)
        centroid_target_mask_low = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num_low, dtype=torch.bool)

        centroid_target_med = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num_med, 3), dtype=torch.float32)
        centroid_target_mask_med = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num_med, dtype=torch.bool)

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id


        sub_voxel_low_grid_xy=self.sub_voxel_ratio_low[1]*self.sub_voxel_ratio_low[2]
        tensor_id_low = voxel_coors_low[:, 0] * grid_shape + \
                        voxel_coors_low[:, 2] // self.sub_voxel_ratio_low[1] * self.grid_size[1] + \
                        voxel_coors_low[:, 3] // self.sub_voxel_ratio_low[2]
        tensor_id_low = hash_table[tensor_id_low]
        target_id_low = tensor_id_low * per_sub_voxel_num_low + \
                        (voxel_coors_low[:, 1] % self.sub_voxel_ratio_low[0]) * sub_voxel_low_grid_xy  + \
                        (voxel_coors_low[:, 2] % self.sub_voxel_ratio_low[1]) * self.sub_voxel_ratio_low[2] + \
                        voxel_coors_low[:, 3] % self.sub_voxel_ratio_low[2]
        centroid_target_low[target_id_low] = voxel_centroids_low
        centroid_target_mask_low[target_id_low] = True
        centroid_target_low = centroid_target_low.view(voxel_num, per_sub_voxel_num_low, 3)[ids_masked]
        centroid_target_mask_low = centroid_target_mask_low.view(voxel_num, per_sub_voxel_num_low)[ids_masked]

        sub_voxel_med_grid_xy = self.sub_voxel_ratio_med[1] * self.sub_voxel_ratio_med[2]
        tensor_id_med = voxel_coors_med[:, 0] * grid_shape + \
                        voxel_coors_med[:, 2] // self.sub_voxel_ratio_med[1] * self.grid_size[1] + \
                        voxel_coors_med[:, 3] // self.sub_voxel_ratio_med[2]

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
    def get_voxel_id_to_tensor_id(self, voxel_coors, sub_voxel_coors, sub_voxel_centroids, ids_masked, batch_size, ):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num = voxel_coors.shape[0]
        grid_shape = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        per_sub_voxel_num = self.sub_voxel_ratio[0] * self.sub_voxel_ratio[1] * self.sub_voxel_ratio[2]
        centroid_target = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num, 3), dtype=torch.float32)
        centroid_target_mask = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num, dtype=torch.bool)

        hash_table = voxel_coors.new_zeros(batch_size * grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id = voxel_coors[:, 0] * grid_shape + voxel_coors[:, 2] * self.grid_size[1] + voxel_coors[:, 3]
        hash_table[tensor_id] = voxel_id

        tensor_id = sub_voxel_coors[:, 0] * grid_shape + sub_voxel_coors[:, 2] // self.sub_voxel_ratio[1] * \
                    self.grid_size[1] + sub_voxel_coors[:, 3] // self.sub_voxel_ratio[1]
        tensor_id = hash_table[tensor_id]
        target_id = tensor_id * per_sub_voxel_num + sub_voxel_coors[:, 1] % self.sub_voxel_ratio[0] + sub_voxel_coors[:,2] % \
                    self.sub_voxel_ratio[1] + sub_voxel_coors[:, 3] % self.sub_voxel_ratio[2]

        centroid_target[target_id] = sub_voxel_centroids
        centroid_target_mask[target_id] = True
        centroid_target = centroid_target.view(voxel_num, per_sub_voxel_num, 3)[ids_masked]
        centroid_target_mask = centroid_target_mask.view(voxel_num, per_sub_voxel_num)[ids_masked]
        return centroid_target, centroid_target_mask



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

        centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(dim=0, return_inverse=True,
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
        edge_vec1=points[sort_idx] - points_centroids
        edge_vec2=points[ori_id_reverse[torch.sort(inverse_unique_idxs)[1]]] - points_centroids
        nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
        nor = nor / torch.norm(nor, dim=-1, keepdim=True)
        unique_idxs = unique_idxs[sort_idx].view(unique_idxs.size(0), 1).expand(-1, C)
        sur_nor = torch.zeros((centroid_voxel_idxs.shape[0], nor.shape[-1]), device=points.device,
                                    dtype=torch.float).scatter_add_(0, unique_idxs, nor)
        sur_nor = sur_nor / labels_count.unsqueeze(-1)
        return centroids,sur_nor, centroid_voxel_idxs, labels_count



    @force_fp32(apply_to=('reg_pred', 'centroid_target','cls_pred'))
    def forward_loss(self,centroid_low,centroid_low_mask ,centroid_med,centroid_med_mask ,centroid_high,
                                     reg_pred_low,reg_pred_med,reg_pred_high,cls_pred_low=None, cls_pred_med=None):
        centroid_low_mask=centroid_low_mask.view(-1)
        centroid_low=centroid_low.view(-1,3)[centroid_low_mask]
        reg_pred_low=reg_pred_low.view(-1,3)[centroid_low_mask]

        centroid_med_mask=centroid_med_mask.view(-1)
        centroid_med=centroid_med.view(-1,3)[centroid_med_mask]
        reg_pred_med=reg_pred_med.view(-1,3)[centroid_med_mask]

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

        else:
            loss_reg_low = self.reg_loss(reg_pred_low, centroid_low) * self.loss_ratio_low
            loss_reg_med = self.reg_loss(reg_pred_med, centroid_med) * self.loss_ratio_med
            loss_reg_top = self.reg_loss(reg_pred_high, centroid_high) * self.loss_ratio_top

        if self.cls_sub_voxel:
            assert cls_pred_low is not None
            cls_pred_low = cls_pred_low.view(-1, 2)
            cls_pred_med = cls_pred_med.view(-1, 2)
            # with torch.no_grad():
            #     test=torch.max(cls_pred.sigmoid(),dim=1)[1]
            #     print((test==centroid_target_mask).sum()/centroid_target_mask.shape[0])
            loss_cls_low = self.cls_loss(cls_pred_low, centroid_low_mask.long())*self.cls_loss_ratio_low
            loss_cls_med = self.cls_loss(cls_pred_med, centroid_med_mask.long())*self.cls_loss_ratio_med
            loss = dict(loss_centroid_low=loss_reg_low,loss_centroid_med=loss_reg_med,loss_centroid_top=loss_reg_top,
                      loss_cls_low=loss_cls_low,loss_cls_med=loss_cls_med,)
        else:
            loss = dict(loss_centroid_low=loss_reg_low,loss_centroid_med=loss_reg_med,loss_centroid_top=loss_reg_top)
        return loss

    def simple_test(self, points,gt_bboxes_3d,gt_labels_3d, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x,centroid_target,centroid_target_mask,mask_coors = self.extract_feat(points,gt_bboxes_3d,gt_labels_3d, img_metas,vis=self.vis)

        if self.vis:
            device = mask_coors.device
            voxel_size = torch.tensor(self.voxel_size[::-1], device=device)
            start_coors = torch.tensor(self.point_cloud_range[:3][::-1], device=device)
            # print('test shape',coors.shape,voxel_size.shape,start_coors.shape)
            mask_coors = mask_coors * voxel_size + start_coors
            x = x * voxel_size + mask_coors.unsqueeze(dim=1)
            centroid_target_mask = centroid_target_mask.view(-1)
            x = x.view(-1, 3)[centroid_target_mask]
            from mmdet3d.models.utils.pc_util import write_ply
            import os

            dump_dir = '/data/yangming/txy/mae_centroid/'
            write_ply(x[:,[2,1,0]], os.path.join(dump_dir, '{}_pred.ply'.format(time_save)))

        return  [dict(
        boxes_3d=[],
        scores_3d=[],
        labels_3d=[])]


    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)
        return feats

    @torch.no_grad()
    def vis_centroid(self,points,centroid,ids_keep,ids_mask):
        from mmdet3d.models.utils.pc_util import write_ply
        import os
        import time
        dump_dir='/data/yangming/txy/mae_centroid/'
        global time_save
        time_save=time.time()
        write_ply(points, os.path.join(dump_dir, '{}_pc.ply'.format(time_save)))
        write_ply(centroid, os.path.join(dump_dir, '{}_centroid.ply'.format(time_save)))
        # write_ply(centroid[ids_keep], os.path.join(dump_dir, '{}_centroid_keep.ply'.format(time_save)))
        # write_ply(centroid[ids_mask], os.path.join(dump_dir, '{}_centroid_mask.ply'.format(time_save)))
        print("!!!!!!!!!!!!!!!!!!!!!!")
        return

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



