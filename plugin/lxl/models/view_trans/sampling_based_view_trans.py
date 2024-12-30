import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.runner import auto_fp16
from torch import nn

from mmdet3d.models.builder import MODELS

@MODELS.register_module()
class SamplingBasedViewTrans(nn.Module):
    def __init__(self, num_convs=3, embed_dims=128, kernel_size=(3, 3), accelerate=False,  # 'radar2img' in VoD is the same so that acceleration is possible
                 pc_range=None, max_depth=None, voxel_shape=None, **kwargs):
        super(SamplingBasedViewTrans, self).__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        device = kwargs.get('device', 'cuda')

        self.max_depth = max_depth
        self.voxel_shape = voxel_shape

        _width = (torch.arange(0, self.voxel_shape[0], dtype=torch.float, device=device) + 0.5) / self.voxel_shape[0]
        _hight = (torch.arange(0, self.voxel_shape[1], dtype=torch.float, device=device) + 0.5) / self.voxel_shape[1]
        _depth = (torch.arange(0, self.voxel_shape[2], dtype=torch.float, device=device) + 0.5) / self.voxel_shape[2]
        reference_voxel = torch.stack(torch.meshgrid([_width, _hight, _depth]), dim=-1)

        reference_voxel[..., 0:1] = reference_voxel[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_voxel[..., 1:2] = reference_voxel[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_voxel[..., 2:3] = reference_voxel[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        self.reference_voxel = reference_voxel.permute(2, 1, 0, 3)

        self.accelerate = accelerate
        self.initial_flag = True

        padding = tuple([(_k - 1) // 2 for _k in kernel_size])
        self.conv_layer = []
        for k in range(num_convs):
            if k == 0:
                input_dims = embed_dims * voxel_shape[2] * 2
            else:
                input_dims = embed_dims
            conv = nn.Sequential(
                nn.Conv2d(input_dims, embed_dims, kernel_size=kernel_size, stride=1, padding=padding, bias=True, device=device),
                nn.BatchNorm2d(embed_dims, device=kwargs.get('device', 'cuda')),
                nn.ReLU(inplace=True))
            self.conv_layer.append(conv)

        self.radar_occupancy_net = nn.Conv2d(embed_dims, voxel_shape[2], kernel_size=1, padding=0, device='cuda')


    def init_weights(self):
        for layer in self.conv_layer:
            xavier_init(layer, distribution='uniform', bias=0.)


    @auto_fp16(apply_to=("mlvl_feats"))
    def forward(self, mlvl_feats, img_depth, img_metas, radar_pts_feats=None):
        batch_size = len(mlvl_feats[0])

        img_voxels, img_voxels_with_depth, mask = self.feature_sampling(mlvl_feats, img_metas, img_depth, batch_size)

        radar_occupancy = self.radar_occupancy_net(radar_pts_feats).sigmoid()
        radar_occupancy = radar_occupancy.flatten(1, 3).unsqueeze(1)
        img_voxels = torch.cat([img_voxels_with_depth, img_voxels * radar_occupancy], dim=1)
        img_voxels = img_voxels * mask
        img_voxels = img_voxels.reshape(img_voxels.shape[0], -1, *self.voxel_shape[::-1])

        B, C, D, H, W = img_voxels.shape
        img_bev_feats = img_voxels.reshape((B, -1, H, W))
        img_bev_feats = self.feat_encoding(img_bev_feats)

        return img_bev_feats


    def feat_encoding(self, img_bev_feats):
        for _idx, layer in enumerate(self.conv_layer):
            img_bev_feats = layer(img_bev_feats)
        return img_bev_feats


    def prepare_sampling(self, radar2img, input_shape, batch_size):
        if not self.accelerate:
            assert len(radar2img.shape) == 4
            reference_voxel = self.reference_voxel.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        else:
            assert len(radar2img.shape) == 3
            reference_voxel = self.reference_voxel

        reference_voxel = torch.cat((reference_voxel, torch.ones_like(reference_voxel[..., :1])), -1)
        reference_voxel = reference_voxel.flatten(-4, -2)
        reference_voxel = reference_voxel.unsqueeze(-1)

        if self.fp16_enabled:
            radar2img = radar2img.half()
            reference_voxel = reference_voxel.half()

        reference_voxel_cam = torch.matmul(radar2img, reference_voxel).squeeze(-1)

        eps = 1e-5
        reference_depth = reference_voxel_cam[..., 2:3].clone()
        mask = (reference_depth > eps)

        reference_voxel_cam = reference_voxel_cam[..., 0:2] / torch.maximum(reference_voxel_cam[..., 2:3], torch.ones_like(reference_voxel_cam[..., 2:3]) * eps)

        reference_voxel_cam[..., 0] /= input_shape[1]
        reference_voxel_cam[..., 1] /= input_shape[0]
        reference_voxel_cam = (reference_voxel_cam - 0.5) * 2

        reference_depth /= self.max_depth
        reference_depth = (reference_depth - 0.5) * 2

        reference_voxel_cam = torch.cat([reference_voxel_cam, reference_depth], dim=-1)

        mask = (mask & (reference_voxel_cam[..., 0:1] > -1.0) & (reference_voxel_cam[..., 0:1] < 1.0)
                     & (reference_voxel_cam[..., 1:2] > -1.0) & (reference_voxel_cam[..., 1:2] < 1.0)
                     & (reference_depth > -1.0) & (reference_depth < 1.0))

        if not self.accelerate:
            mask = mask.view(batch_size, 1, -1)
        else:
            mask = mask.reshape((1, 1, -1)).repeat(batch_size, 1, 1)
            reference_voxel_cam = reference_voxel_cam.reshape((1, -1, 3)).repeat(batch_size, 1, 1)

        return reference_voxel_cam, mask


    def feature_sampling(self, mlvl_feats, img_metas, img_depth=None, batch_size=1):
        if self.fp16_enabled:
            img_depth = [_depth.half() for _depth in img_depth]

        if self.accelerate:
            if self.initial_flag:
                radar2img = img_metas[0]['radar2img']
                radar2img = self.reference_voxel.new_tensor(radar2img)
                radar2img = radar2img.unsqueeze(0)

                self.reference_voxel_cam, self.mask = self.prepare_sampling(radar2img, input_shape=img_metas[0]['input_shape'], batch_size=batch_size)
                self.img2radar = radar2img.inverse()[:, :3, :]

            reference_voxel_cam, mask = self.reference_voxel_cam[:batch_size],  self.mask[:batch_size]
        else:
            radar2img = []
            for img_meta in img_metas:
                radar2img.append(img_meta['radar2img'])
            radar2img = np.asarray(radar2img)
            radar2img = self.reference_voxel.new_tensor(radar2img)

            reference_voxel_cam, mask = self.prepare_sampling(radar2img.unsqueeze(1), input_shape=img_metas[0]['input_shape'], batch_size=batch_size)

        sampled_feats = 0
        sampled_feats_with_depth = 0
        for lvl_id in range(len(mlvl_feats)):
            sampled_feat = F.grid_sample(mlvl_feats[lvl_id], reference_voxel_cam[..., :2].unsqueeze(2), mode='bilinear')
            sampled_feat = sampled_feat.squeeze(-1)

            sampled_depth = F.grid_sample(img_depth[lvl_id].unsqueeze(1), reference_voxel_cam[:, None, :, None, :], mode='bilinear')
            sampled_depth = sampled_depth.squeeze(2).squeeze(-1)

            sampled_feats = sampled_feats + sampled_feat
            sampled_feats_with_depth = sampled_feats_with_depth + sampled_feat * sampled_depth

        if self.initial_flag:
            self.initial_flag = False

        return sampled_feats, sampled_feats_with_depth, mask