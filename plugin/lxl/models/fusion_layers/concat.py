import torch
from torch import nn as nn
from mmdet3d.models.builder import FUSION_LAYERS

@FUSION_LAYERS.register_module()
class ConcatFusion(nn.Module):
    def __init__(self, num_convs=2, embed_dims=128):
        super(ConcatFusion, self).__init__()
        self.conv_after_fusion = nn.ModuleList()
        for k in range(num_convs):
            if k == 0:
                conv = nn.Sequential(
                    nn.Conv2d(2 * embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True, device='cuda'),
                    nn.BatchNorm2d(embed_dims, device='cuda'),
                    nn.ReLU(inplace=True))
            else:       # k >= 1
                conv = nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=True, device='cuda'),
                    nn.BatchNorm2d(embed_dims, device='cuda'),
                    nn.ReLU(inplace=True))
            self.conv_after_fusion.append(conv)

    def forward(self, img_feats, pts_feats):
        bev_feats = torch.cat((img_feats, pts_feats), dim=1)

        for layer in self.conv_after_fusion:
            bev_feats = layer(bev_feats)
        return bev_feats