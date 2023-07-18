import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .voxelnet import VoxelNet
from ..builder import build_loss

from .. import builder
from mmdet3d.ops import Voxelization
from .single_stage import SingleStage3DDetector

from mmdet3d.ops import points_in_boxes_cpu,points_in_boxes_gpu


@DETECTORS.register_module()
class VoxelDynamicVoxelNet(SingleStage3DDetector):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """
    def __init__(self,
                 loss,
                 loss_ratio,
                 random_mask_ratio,
                 grid_size,
                 sub_voxel_ratio,
                 voxel_layer,
                 sub_voxel_layer,
                 voxel_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(VoxelDynamicVoxelNet, self).__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.random_mask_ratio=random_mask_ratio
        self.point_cloud_range=voxel_layer['point_cloud_range']
        self.voxel_size=voxel_layer['voxel_size']
        self.grid_size=grid_size
        # self.sub_voxel_ratio=sub_voxel_ratio
        # self.sub_voxel_layer = Voxelization(**sub_voxel_layer)
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.loss = build_loss(loss)
        self.loss_ratio = loss_ratio

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
        x,centroid_target = self.extract_feat(points,gt_bboxes_3d,gt_labels_3d, img_metas)
        loss = self.forward_loss(x,centroid_target)

        return loss


    def extract_feat(self, points: list,gt_bboxes_3d,gt_labels_3d, img_metas):
        """Extract features from points."""

        batch_size=len(points)
        voxels, coors = self.voxelize(points)
        # sub_coors = self.sub_voxelize(points)



        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)

        ids_keep,ids_mask=self.get_vanilla_mask_index(feature_coors,batch_size)

        centroids, centroid_voxel_coors, labels_count=self.get_centroid_per_voxel(voxels[:,[2,1,0]],coors)

        #
        # print('test',feature_coors.shape,centroid_voxel_coors.shape,feature_coors[0],centroid_voxel_coors[0])
        #
        # centroid_target,centroid_target_mask=self.get_voxel_id_to_tensor_id(feature_coors.long(),centroid_voxel_coors.long(),centroids,ids_mask,batch_size)
        centroids=centroids[ids_mask]

        mask_coors=feature_coors[ids_mask]
        centroids=self.normalize_centroid_per_voxel(mask_coors[:,1:],centroids)

        x = self.backbone(voxel_features[ids_keep],feature_coors[ids_keep],mask_coors,batch_size)

        return x,centroids

    @torch.no_grad()
    def get_focal_mask_index(self,coors,gt_bboxes,gt_labels_3d):
        #TODO: this version is only a tricky implmentation of judging pillar in bboxes. Also having some error.
        batch_size=len(gt_bboxes)
        device=coors.device
        voxel_size=torch.tensor(self.voxel_size[:2],device=device)
        start_coors=torch.tensor(self.point_cloud_range[:2],device=device)


        for i in range(batch_size):
            inds = torch.where(coors[:, 0] == i)
            #print('inds',inds.shape,inds.dtype)

            coors_per_batch=coors[inds][:,[3,2]]*voxel_size+start_coors
            z_coors = torch.ones((coors_per_batch.shape[0],1),device=device)*(-0.5)
            coors_per_batch=torch.cat([coors_per_batch,z_coors],dim=1)
            #print(coors_per_batch.shape,coors_per_batch[:10])
            #print(gt_bboxes[i])
            valid_index=gt_labels_3d[i]!=-1
            #print(gt_bboxes[i][valid_index])
            voxel_in_gt_bboxes=gt_bboxes[i][valid_index].points_in_boxes(coors_per_batch)
            # print('gt_bboxes[i]',gt_bboxes[i])
            # voxel_in_gt_bboxes=points_in_boxes_gpu(coors_per_batch.unsqueeze(0),gt_bboxes[i].tensor.unsqueeze(0).to(device=device))
            test_index=voxel_in_gt_bboxes!=-1

            print(voxel_in_gt_bboxes.shape,test_index.sum(),voxel_in_gt_bboxes[test_index][:10])
            #print(coors.shape,voxel_in_gt_bboxes.shape,voxel_in_gt_bboxes.dtype,voxel_in_gt_bboxes[0,:10])
        return None

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

            # noise = torch.rand(L, device=coors.device)
            # ids_shuffle = torch.argsort(noise)
            # ids_restore = torch.argsort(ids_shuffle)
            # ids_keep = ids_shuffle[:, :len_keep]

            #print('inds',inds.shape,inds.dtype)
            #print(type(inds),len(inds),inds[0].shape,coors.shape,inds[0][:10])
        ids_keep_list = torch.cat(ids_keep_list)
        ids_mask_list = torch.cat(ids_mask_list)
        return ids_keep_list,ids_mask_list


    @torch.no_grad()
    @force_fp32()
    def sub_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            torch.Tensor: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for i,res in enumerate(points):
            res_coors = self.sub_voxel_layer(res)
            coors.append(res_coors)

        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return coors_batch

    # @torch.no_grad()
    # @force_fp32()
    # def sub_voxelize(self, points):
    #     """Apply dynamic voxelization to points.
    #
    #     Args:
    #         points (list[torch.Tensor]): Points of each sample.
    #
    #     Returns:
    #         torch.Tensor: Concatenated points and coordinates.
    #     """
    #     coors = []
    #     points_list = []
    #     # dynamic voxelization only provide a coors mapping
    #     for i,res in enumerate(points):
    #         res_coors = self.sub_voxel_layer(res)
    #         coors.append(res_coors)
    #         points_list.append(F.pad(res, (1, 0), mode='constant', value=i))
    #     points_list=torch.cat(points_list)
    #     coors_batch = []
    #     for i, coor in enumerate(coors):
    #         coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
    #         coors_batch.append(coor_pad)
    #     coors_batch = torch.cat(coors_batch, dim=0)
    #     return points_list,coors_batch

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

        coors_ = coors * voxel_size + start_coors
        centroids=(centroids-coors_.unsqueeze(dim=1))/voxel_size
        return centroids

    @torch.no_grad()
    def normalize_centroid_per_voxel(self,coors,centroids):
        device=coors.device
        voxel_size=torch.tensor(self.voxel_size[::-1],device=device)
        start_coors=torch.tensor(self.point_cloud_range[:3][::-1],device=device)

        coors_ = coors * voxel_size + start_coors
        centroids=(centroids-coors_)/voxel_size
        return centroids

    @torch.no_grad()
    def get_voxel_id_to_tensor_id(self,voxel_coors,sub_voxel_coors,sub_voxel_centroids,ids_masked,batch_size,):
        # TODO: this version doesn't support the ori voxel's height isn't the whole pillar
        voxel_num=voxel_coors.shape[0]
        grid_shape=self.grid_size[0]*self.grid_size[1]*self.grid_size[2]
        per_sub_voxel_num=self.sub_voxel_ratio[0]*self.sub_voxel_ratio[1]*self.sub_voxel_ratio[2]
        centroid_target = voxel_coors.new_zeros((voxel_num * per_sub_voxel_num,3),dtype=torch.float32)
        centroid_target_mask = voxel_coors.new_zeros(voxel_num * per_sub_voxel_num, dtype=torch.bool)

        hash_table = voxel_coors.new_zeros(batch_size*grid_shape, dtype=torch.int64)
        voxel_id = torch.arange(voxel_coors.shape[0], device=voxel_coors.device)
        tensor_id=voxel_coors[:,0]*grid_shape+voxel_coors[:,2]*self.grid_size[1]+voxel_coors[:,3]
        hash_table[tensor_id] = voxel_id

        tensor_id=sub_voxel_coors[:,0]*grid_shape+sub_voxel_coors[:,2]//self.sub_voxel_ratio[1]*self.grid_size[1]+sub_voxel_coors[:,3]//self.sub_voxel_ratio[1]
        tensor_id=hash_table[tensor_id]
        target_id=tensor_id*per_sub_voxel_num+sub_voxel_coors[:,1]%self.sub_voxel_ratio[0]+sub_voxel_coors[:,2]%self.sub_voxel_ratio[1]+sub_voxel_coors[:,3]%self.sub_voxel_ratio[2]

        centroid_target[target_id]= sub_voxel_centroids
        centroid_target_mask[target_id]= True
        centroid_target = centroid_target.view(voxel_num,per_sub_voxel_num,3)[ids_masked]
        centroid_target_mask = centroid_target_mask.view(voxel_num, per_sub_voxel_num)[ids_masked]

        return centroid_target,centroid_target_mask



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



    @force_fp32(apply_to=('x', 'centroid_target'))
    def forward_loss(self,x,centroid_target):
        x=x.view(-1,3)
        loss = self.loss(x,centroid_target) * self.loss_ratio
        return dict(loss_centroid=loss,)



    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas)

        return x

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)



        return feats
