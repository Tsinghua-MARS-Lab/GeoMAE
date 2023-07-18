_base_ = [
    '../_base_/datasets/nus-3d-ssl.py',
    '../_base_/schedules/cosine_2x.py',
    '../_base_/default_runtime.py',
]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

dataset_type = 'NuScenesDatasetSSL'
data_root = 'data/nuscenes/'

voxel_size = (0.256, 0.256, 8)
window_shape=(12, 12) # 12 * 0.32m
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

sub_voxel_size_low = (0.064, 0.064, 1)
sub_voxel_size_med = (0.128, 0.128, 2)
sub_voxel_size_top = (0.256, 0.256, 8)
sub_voxel_ratio_low = (8,4,4) # z,y,x
sub_voxel_ratio_med = (4,2,2) # z,y,x
input_voxel_ratio = (1,2,2)
grid_size=(1,400,400)    # z,y,x
random_mask_ratio=0.7
loss_ratio_low=10.0
loss_ratio_med=8.0
loss_ratio_top=10.0
cls_loss_ratio_low=5.0
cls_loss_ratio_med=2.0
cls_sub_voxel=True

loss_ratio_low_nor=4.0
loss_ratio_med_nor=1.0
loss_ratio_top_nor=1.0


drop_info_training ={
    0:{'max_tokens':56, 'drop_range':(0, 56)},
    1:{'max_tokens':144, 'drop_range':(56, 100000)},
}
drop_info_test ={
    0:{'max_tokens':32, 'drop_range':(0, 32)},
    1:{'max_tokens':72, 'drop_range':(32, 72)},
    2:{'max_tokens':144, 'drop_range':(72, 100000)},
}

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
# drop_info_training ={
#     0:{'max_tokens':30, 'drop_range':(0, 30)},
#     1:{'max_tokens':60, 'drop_range':(30, 60)},
#     2:{'max_tokens':80, 'drop_range':(60, 100)},
#     3:{'max_tokens':144, 'drop_range':(100, 100000)},
# }
# drop_info_test ={
#     0:{'max_tokens':30, 'drop_range':(0, 30)},
#     1:{'max_tokens':60, 'drop_range':(30, 60)},
#     2:{'max_tokens':100, 'drop_range':(60, 100)},
#     3:{'max_tokens':144, 'drop_range':(100, 100000)},
# }

# drop_info_training ={
#     0:{'max_tokens':16, 'drop_range':(0, 16)},
#     1:{'max_tokens':32, 'drop_range':(16, 32)},
#     2:{'max_tokens':64, 'drop_range':(32, 100)},
#
# }
# drop_info_test ={
#     0:{'max_tokens':16, 'drop_range':(0, 16)},
#     1:{'max_tokens':32, 'drop_range':(16, 32)},
#     2:{'max_tokens':64, 'drop_range':(32, 100)},
# }


drop_info = (drop_info_training, drop_info_test)
shifts_list=[(0, 0), (window_shape[0]//2, window_shape[1]//2) ]


model = dict(
    type='MultiSubVoxelDynamicVoxelNetSSL',
    normalize_sub_voxel=True,
    mse_loss=True,
    loss=dict(type='SmoothL1Loss', reduction='mean', loss_weight=1.0),
    spatial_shape=[1,400,400],
    loss_ratio_low_nor=loss_ratio_low_nor,
    loss_ratio_med_nor=loss_ratio_med_nor,
    loss_ratio_top_nor=loss_ratio_top_nor,
    loss_ratio_low=loss_ratio_low,
    loss_ratio_med=loss_ratio_med,
    loss_ratio_top=loss_ratio_top,
    cls_sub_voxel=cls_sub_voxel,
    cls_loss_ratio_low=cls_loss_ratio_low,
    cls_loss_ratio_med=cls_loss_ratio_med,
    random_mask_ratio=random_mask_ratio,
    grid_size=grid_size,
    sub_voxel_ratio_low=sub_voxel_ratio_low,
    sub_voxel_ratio_med=sub_voxel_ratio_med,
    voxel_layer=dict(
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    sub_voxel_layer_low=dict(
        voxel_size=sub_voxel_size_low,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    sub_voxel_layer_med=dict(
        voxel_size=sub_voxel_size_med,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    hard_sub_voxel_layer_low=dict(
        voxel_size=sub_voxel_size_low,
        max_num_points=30,
        point_cloud_range=point_cloud_range,
        max_voxels=(140000, 140000)
    ),
    hard_sub_voxel_layer_med=dict(
        voxel_size=sub_voxel_size_med,
        max_num_points=50,
        point_cloud_range=point_cloud_range,
        max_voxels=(80000, 80000)
    ),
    hard_sub_voxel_layer_top=dict(
        voxel_size=voxel_size,
        max_num_points=100,
        point_cloud_range=point_cloud_range,
        max_voxels=(40000, 40000)
    ),
    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),


    backbone=dict(
        type='MultiMAESSTSPChoose',
        cls_sub_voxel=cls_sub_voxel,
        window_shape=window_shape,
        shifts_list=shifts_list,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        shuffle_voxels=False,
        low=False,
        med=False,
        top=True,
        d_model=[128, ] * 6,
        nhead=[8, ] * 6,
        sub_voxel_ratio_low=sub_voxel_ratio_low,
        sub_voxel_ratio_med=sub_voxel_ratio_med,
        encoder_num_blocks=6,
        decoder_num_blocks=2,
        dim_feedforward=[256, ] * 6,
        output_shape=[400, 400],
        # num_attached_conv=2,
        # conv_kwargs=[
        #     # dict(kernel_size=3, dilation=1, padding=1, stride=1),
        #     dict(kernel_size=3, dilation=2, padding=2, stride=1),
        #     dict(kernel_size=3, dilation=2, padding=2, stride=1),
        # ],
        # conv_in_channel=128,
        # conv_out_channel=128,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        # checkpoint_blocks=[0,1]
        ),
)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# evaluation = dict(interval=12)
file_client_args = dict(backend='disk')
# db_sampler = dict(
#     data_root=data_root,
#     info_path=data_root + 'nuscenes_dbinfos_train.pkl',
#     rate=1.0,
#     prepare=dict(
#         filter_by_difficulty=[-1],
#         filter_by_min_points=dict(
#             car=5,
#             truck=5,
#             bus=5,
#             trailer=5,
#             construction_vehicle=5,
#             traffic_cone=5,
#             barrier=5,
#             motorcycle=5,
#             bicycle=5,
#             pedestrian=5)),
#     classes=class_names,
#     sample_groups=dict(
#         car=2,
#         truck=2,
#         construction_vehicle=4,
#         bus=2,
#         trailer=4,
#         barrier=2,
#         motorcycle=4,
#         bicycle=4,
#         pedestrian=2,
#         traffic_cone=2),
#     points_loader=dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=[0, 1, 2, 3, 4],
#         file_client_args=file_client_args))


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='ObjectSampleSSL', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

# data = dict(
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file=data_root + 'nuscenes_ssl_infos_train.pkl',
#             pipeline=train_pipeline,
#             classes=class_names,
#             test_mode=False,
#             use_valid_flag=True,
#             # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
#             # and box_type_3d='Depth' in sunrgbd and scannet dataset.
#             box_type_3d='LiDAR')),
#     val=dict(pipeline=test_pipeline, classes=class_names),
#     test=dict(pipeline=test_pipeline, classes=class_names))

# fp16 = dict(loss_scale=512.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_ssl_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_ssl_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_ssl_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

lr = 1e-5
runner = dict(type='EpochBasedRunner', max_epochs=96)
evaluation = dict(interval=100, pipeline=eval_pipeline)


#     backbone=dict(
#         type='MultiMAESST',
#         cls_sub_voxel=cls_sub_voxel,
#         window_shape=window_shape,
#         shifts_list=shifts_list,
#         point_cloud_range=point_cloud_range,
#         voxel_size=voxel_size,
#         shuffle_voxels=False,
#
#         d_model=[128,] * 6,
#         nhead=[8, ] * 6,
#         sub_voxel_ratio_low=sub_voxel_ratio_low,
#         sub_voxel_ratio_med=sub_voxel_ratio_med,
#         encoder_num_blocks=6,
#         decoder_num_blocks=2,
#         dim_feedforward=[256, ] * 6,
#         output_shape=[468, 468],
#         # num_attached_conv=2,
#         # conv_kwargs=[
#         #     # dict(kernel_size=3, dilation=1, padding=1, stride=1),
#         #     dict(kernel_size=3, dilation=2, padding=2, stride=1),
#         #     dict(kernel_size=3, dilation=2, padding=2, stride=1),
#         # ],
#         # conv_in_channel=128,
#         # conv_out_channel=128,
#         debug=True,
#         drop_info=drop_info,
#         pos_temperature=10000,
#         normalize_pos=False,
#         # checkpoint_blocks=[0,1]
#     ),
# )