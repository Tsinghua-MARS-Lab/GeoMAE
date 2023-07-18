_base_ = [
    '../_base_/models/sat_base.py',
    '../_base_/datasets/waymo-3d-car.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]
voxel_size = (0.32, 0.32, 6)
window_shape=(10, 10) # 12 * 0.32m
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]


model = dict(
    type='DynamicVoxelNet',

    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicVFE',
        in_channels=3,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),
    middle_encoder=dict(
        type='NoneSSTInputLayer',
        shuffle_voxels=True,
    ),

    backbone=dict(
        type='SwinBlockSparseFormerDebug',
        window_sizes=[[1, 1, 10, 10]],
        window_offsets=[[0,0,0,0],[0,0,5,5]],
        maxk=2,
        num_heads=[8],
        tensor_lengthes=[16,32,48,64,80,100],
        spatial_shapes=[[0, 0, 0, 1, 468, 468], [0, 0, 0, 1, 400, 352], [0, 0, 0, 1, 200, 176]],
        input_channels=128,
        batch_size=4,
        hash_sizes=[20000, 7500, 7500],
        attend_sizes=[9],
        choose_indices=[[[0, 0, 0], [0, -1, 0], [0, -1, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, -1], [0, 0, -1],
                         [0, -1, -1], ]],
        depths=[6],
        mlp_ratio=[2],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        mode='max',
        pos_type='fourier',
        output_shape=[468, 468],
        num_attached_conv=3,
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=1,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[[-74.88, -74.88, -0.0345, 74.88, 74.88, -0.0345], ],
            sizes=[
                [2.08, 4.73, 1.77],  # car
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=7),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.5),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.55,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1), ],
        allowed_border=0,
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pos_weight=-1,
        debug=False
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=12)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            load_interval=5)
    ),
)