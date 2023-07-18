import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .voxelnet import VoxelNet
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d


@DETECTORS.register_module()
class DynamicVoxelNetPretrained(VoxelNet):
    r"""VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """
    def __init__(self,

                 model_path,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 centerpoint_head=False,
                 eval_flag=False,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):

        super(DynamicVoxelNetPretrained, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.centerpoint_head = centerpoint_head
        if not eval_flag:

            model=torch.load(model_path,map_location='cpu')
            voxel_encoder_dict=self.voxel_encoder.state_dict()
            voxel_pretrained_dict={k[14:]:v for k,v in model['state_dict'].items() if (k[14:] in voxel_encoder_dict and 'voxel_encoder' in k)}

            print('#########' * 50)
            print(voxel_encoder_dict.keys())
            print()
            print(voxel_pretrained_dict.keys())
            print('#########' * 50)

            voxel_encoder_dict.update(voxel_pretrained_dict)
            self.voxel_encoder.load_state_dict(voxel_encoder_dict)


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

        x = self.extract_feat(points, img_metas)
        outs = self.bbox_head(x)
        if self.centerpoint_head:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        else:
            loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


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

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        #print('points',len(points),points[0].shape)
        voxels, coors = self.voxelize(points)
        #print('voxels',type(voxels),voxels.shape,coors[:10])
        voxel_features, feature_coors = self.voxel_encoder(voxels, coors)
        #print('voxel_features',type(voxel_features),voxel_features.shape)
        batch_size = coors[-1, 0].item() + 1
        #print('test',batch_size,coors[:,0].max())
        #print(self.middle_encoder)
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

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
