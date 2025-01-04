_base_ = [
    '../_base_/datasets/vod_r_c_3classes.py',
    '../_base_/schedules/vod_step_80e.py',
    '../../../../configs/_base_/default_runtime.py'
]

plugin = True
plugin_dir = 'plugin/lxl/'

model = dict(
    type='LXLDetector',
    bev_augmentation_rate=0.5,
    img_backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    img_neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    pretrained_img='/mnt/e/LXL/work_dir/result/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',   # TODO: set path
    load_img=['backbone', 'neck'],
    depth_head=dict(
        type='DepthHead',
        max_depth=60,
        depth_dim=60,
        num_lvl=3,
        in_channels=384),
    view_cfg=dict(
        type='SamplingBasedViewTrans',
        embed_dims=384,
        max_depth=60,
        voxel_shape=[160, 160, 5],
        pc_range=[0, -25.6, -3, 51.2, 25.6, 2],
        fp16_enabled=True,
        accelerate=True),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        voxel_size=[0.16, 0.16, 5],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=7,
        feat_channels=[64],
        voxel_size=[0.16, 0.16, 5],
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        legacy=False),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[320, 320]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    fusion_cfg=dict(
        type='ConcatFusion',
        num_convs=2,
        embed_dims=384),
    pts_bbox_head=dict(
        type='CenterHeadV2',
        in_channels=384,
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist'])
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[0, -25.6, -3, 51.2, 25.6, 2],
            post_center_range=[-5, -30.6, -6, 56.2, 30.6, 5],
            max_num=350,
            score_threshold=0.1,
            out_size_factor=2,
            voxel_size=[0.16, 0.16],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
            grid_size=[320, 320, 1],
            voxel_size=[0.16, 0.16, 5],
            out_size_factor=2,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-5, -30.6, -6, 56.2, 30.6, 5],
            min_radius=[4, 0.3, 0.85],
            score_threshold=0.1,
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            grid_size=[320, 320, 1],
            out_size_factor=2)))