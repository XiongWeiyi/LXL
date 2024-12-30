import numpy as np
import torch
from mmcv.cnn import Conv2d
from mmcv.runner import auto_fp16, force_fp32
from mmdet.models import DETECTORS

from mmdet3d.models.builder import MODELS, FUSION_LAYERS, HEADS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class LXLDetector(MVXTwoStageDetector):
    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 depth_head=None,
                 view_cfg=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 fusion_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 pretrained_img=None,
                 pretrained_pts=None,
                 load_img=None,
                 load_pts=None,
                 bev_augmentation_rate=0.0):
        super(LXLDetector,
              self).__init__(pts_voxel_layer=pts_voxel_layer, pts_voxel_encoder=pts_voxel_encoder, pts_middle_encoder=pts_middle_encoder,
                             pts_fusion_layer=None, img_backbone=img_backbone, pts_backbone=pts_backbone, img_neck=img_neck,
                             pts_neck=pts_neck, pts_bbox_head=pts_bbox_head, img_roi_head=None, img_rpn_head=None,
                             train_cfg=train_cfg, test_cfg=test_cfg, pretrained=pretrained)
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)

            self.depth_net = HEADS.build(depth_head)
            self.view_trans = MODELS.build(view_cfg)

        if fusion_cfg is not None:
            self.fusion_layer = FUSION_LAYERS.build(fusion_cfg)

        self.pretrained_img = pretrained_img
        self.pretrained_pts = pretrained_pts
        self.load_img = load_img
        self.load_pts = load_pts

        self.bev_augmetation_rate = bev_augmentation_rate

    def init_weights(self):
        branches_model_path = [self.pretrained_pts, self.pretrained_img]
        branches_load_keys = [self.load_pts, self.load_img]
        for branch_id in range(2):
            branch_model_path = branches_model_path[branch_id]
            branch_load_keys = branches_load_keys[branch_id]
            if branch_model_path is not None:
                ckpt_load = torch.load(branch_model_path, map_location="cuda:{}".format(torch.cuda.current_device()))["state_dict"]
                print("Loaded pretrained model from: {}".format(branch_model_path))
                for load_key in branch_load_keys:
                    if load_key not in ['view_trans']:
                        dict_load = {_key.replace(load_key + '.', '', 1): ckpt_load[_key] for _key in ckpt_load if load_key in _key}
                        if branch_id == 1 and 'img' not in load_key:
                            load_key = 'img_' + load_key
                        getattr(self, load_key).load_state_dict(dict_load)
                        if branch_id == 1:
                            for name, parameter in getattr(self, load_key).named_parameters():
                                parameter.requires_grad = False
                    else:
                        dict_load = {_key.replace('pts_bbox_head.' + load_key + '.', ''): ckpt_load[_key]
                                     for _key in ckpt_load if 'pts_bbox_head.' + load_key in _key}
                        getattr(self.pts_bbox_head, load_key).load_state_dict(dict_load)
                        for name, parameter in getattr(self.pts_bbox_head, load_key).named_parameters():
                            parameter.requires_grad = False
                    print("Loaded pretrained {}".format(load_key))
                    assert len(dict_load) > 0


    @force_fp32()
    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)[0]
        return x


    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is None:
            return None
        input_shape = img.shape[-2:]

        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            img_feats_reshaped.append(img_feat)

        if self.training:
            img_feats_ = []
            for lvl in range(len(img_feats_reshaped)):
                img_feats_.append([])

            for i in range(len(img_metas)):
                if img_metas[i].get('flip', False):
                    for lvl in range(len(img_feats_reshaped)):
                        img_feats_[lvl].append(torch.flip(img_feats_reshaped[lvl][i], dims=[2]))
                else:
                    for lvl in range(len(img_feats_reshaped)):
                        img_feats_[lvl].append(img_feats_reshaped[lvl][i])

            for lvl in range(len(img_feats_reshaped)):
                img_feats_[lvl] = torch.stack(img_feats_[lvl], dim=0)
            return img_feats_

        return img_feats_reshaped


    @auto_fp16(apply_to=('img'))
    def extract_feat(self, lidar_points, radar_points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        if img_feats is not None and getattr(self, 'depth_net', None) is not None:
            img_depth = self.depth_net(img_feats)
        else:
            img_depth = None

        radar_pts_feats = self.extract_pts_feat(radar_points)

        if img_feats is not None and getattr(self, 'view_trans', None) is not None:
            if self.training and (radar_pts_feats is not None):
                radar_pts_feats_ = []
                for frame_id in range(len(img_metas)):
                    if img_metas[frame_id].get('pcd_horizontal_flip', False):
                        radar_pts_feats_.append(torch.flip(radar_pts_feats[frame_id], dims=[1]))
                    else:
                        radar_pts_feats_.append(radar_pts_feats[frame_id])
                radar_pts_feats_ = torch.stack(radar_pts_feats_, dim=0)
                img_feats = self.view_trans(img_feats, img_depth=img_depth, img_metas=img_metas, radar_pts_feats=radar_pts_feats_)
            else:
                img_feats = self.view_trans(img_feats, img_depth=img_depth, img_metas=img_metas, radar_pts_feats=radar_pts_feats)

        if self.training:
            if img_feats is not None:
                img_feats_ = []
            else:
                img_feats_ = None
            if radar_pts_feats is not None:
                pts_feats_ = []
            else:
                pts_feats_ = None

            for frame_id in range(len(img_metas)):
                flip_bev = True if np.random.rand() < self.bev_augmetation_rate else False
                img_metas[frame_id]['flip_bev'] = flip_bev

                if img_feats is not None:
                    if (img_metas[frame_id].get('pcd_horizontal_flip', False) and (not flip_bev)) or \
                            (not img_metas[frame_id].get('pcd_horizontal_flip', False) and flip_bev):
                        img_feats_.append(torch.flip(img_feats[frame_id], dims=[1]))
                    else:
                        img_feats_.append(img_feats[frame_id])
                if radar_pts_feats is not None:
                    if flip_bev:
                        pts_feats_.append(torch.flip(radar_pts_feats[frame_id], dims=[1]))
                    else:
                        pts_feats_.append(radar_pts_feats[frame_id])

            if img_feats is not None:
                img_feats_ = torch.stack(img_feats_, dim=0)
            if radar_pts_feats is not None:
                pts_feats_ = torch.stack(pts_feats_, dim=0)

            bev_feats = self.fuse_feats(img_feats_, pts_feats_)
        else:
            bev_feats = self.fuse_feats(img_feats, radar_pts_feats)

        return [bev_feats]


    def fuse_feats(self, img_feats, pts_feats):
        if img_feats is None:
            return pts_feats

        if pts_feats is None:
            return img_feats

        fused_feats = self.fusion_layer(img_feats, pts_feats)
        return fused_feats


    def forward_pts_train(self,
                          bev_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas):
        """Forward function for point cloud branch.
        Args:
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sample
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(bev_feats, img_metas=img_metas)
        losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)

        return losses

    @force_fp32(apply_to=('img', 'lidar_points', 'radar_points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether return_loss=True.
        Note this setting will change the expected inputs. When `return_loss=True`, img and img_metas are single-nested
        (i.e. torch.Tensor and list[dict]), and when `return_loss=False`, img and img_metas should be double nested
        (i.e.  list[torch.Tensor], list[list[dict]]), with the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


    def forward_train(self,
                      lidar_points=None,
                      radar_points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        bev_feats = self.extract_feat(lidar_points, radar_points, img, img_metas)

        losses = dict()

        for i in range(len(img_metas)):
            if img_metas[i].get('flip_bev', False):
                gt_bboxes_3d[i].flip('horizontal')

        losses_pts = self.forward_pts_train(bev_feats, gt_bboxes_3d, gt_labels_3d, img_metas)
        losses.update(losses_pts)

        return losses


    def forward_test(self, img_metas, lidar_points=None, radar_points=None, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        results = self.simple_test(img_metas, lidar_points, radar_points, img)

        return results


    def simple_test_pts(self, bev_feats, img_metas):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(bev_feats, img_metas=img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]   # to('cpu')
        return bbox_results


    def simple_test(self, img_metas, lidar_points=None, radar_points=None, img=None):
        """Test function without augmentation."""
        bev_feats = self.extract_feat(lidar_points, radar_points, img, img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(bev_feats, img_metas)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list