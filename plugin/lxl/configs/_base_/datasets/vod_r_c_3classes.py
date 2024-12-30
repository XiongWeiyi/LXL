# dataset settings
data_root = '/mnt/e/view_of_delft_PUBLIC/'      # TODO: set path
radar_use_dim = 7
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]

class_names = ['Car', 'Pedestrian', 'Cyclist']

img_norm_cfg = dict(mean=[147.666, 135.721, 106.998], std=[71.947, 71.020, 66.431], to_rgb=False)
radar_pts_norm_cfg = dict(mean=[0, 0, 0, -12.339, -2.818, 0.037, 0], std=[1, 1, 1, 13.240, 1.926, 1.929, 1])

input_modality = dict(use_lidar=False, use_radar=True, use_camera=True)

train_keys = ('gt_bboxes_3d', 'gt_labels_3d')
keys = ()
if input_modality['use_camera']:
    keys += ('img',)
if input_modality['use_lidar']:
    keys += ('lidar_points',)
if input_modality['use_radar']:
    keys += ('radar_points',)
train_keys += keys

train_pipeline = [
    dict(
        type='LoadPointsFromFileV2',
        coord_type='LIDAR',
        sensor='radar',
        load_dim=7,
        use_dim=radar_use_dim,
        **radar_pts_norm_cfg
    ),
    dict(type='LoadImageFromFileMono3D', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilterV2', point_cloud_range=point_cloud_range, sensor='radar'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffleV2'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='RandomFlip3DV2',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0
    ),
    dict(type='DefaultFormatBundle3DV2'),
    dict(type='Collect3D', keys=train_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'radar2img', 'cam2img', 'pad_shape',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx',
                    'lidar_filename', 'radar_filename', 'flip', 'pcd_horizontal_flip'))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFileV2',
        coord_type='LIDAR',
        sensor='radar',
        load_dim=7,
        use_dim=radar_use_dim,
        **radar_pts_norm_cfg),
    dict(type='LoadImageFromFileMono3D', to_float32=True),
    dict(type='PointsRangeFilterV2', point_cloud_range=point_cloud_range, sensor='radar'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3DV2'),
    dict(type='Collect3D', keys=keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 'radar2img', 'cam2img', 'pad_shape',
                    'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx',
                    'lidar_filename', 'radar_filename', 'flip', 'pcd_horizontal_flip'))
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=0,
    train=dict(
        type='VoDDataset',
        data_root=data_root,
        ann_file=data_root + 'VoD_infos_train.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range),
    val=dict(
        type='VoDDataset',
        data_root=data_root,
        ann_file=data_root + 'VoD_infos_val.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range),
    test=dict(
        type='VoDDataset',
        data_root=data_root,
        ann_file=data_root + 'VoD_infos_val.pkl',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        pcd_limit_range=point_cloud_range))

evaluation = dict(interval=1, save_best='pts_bbox/entire_area/mAP_3d_all')