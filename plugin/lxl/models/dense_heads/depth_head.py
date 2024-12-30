from mmcv.cnn import Conv2d
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.models.builder import HEADS


@HEADS.register_module()
class DepthHead(BaseModule):
    def __init__(self, in_channels, depth_dim=60, max_depth=60, num_lvl=3, init_cfg=None):
        super(DepthHead, self).__init__(init_cfg=init_cfg)
        self.num_lvl = num_lvl
        self.depth_dim = depth_dim
        self.max_depth = max_depth

        self.depth_net = nn.ModuleList()
        for lvl in range(num_lvl):
            self.depth_net.append(Conv2d(in_channels, depth_dim, kernel_size=1))


    def forward(self, img_feats):
        pred_depth = []
        for lvl_id in range(self.num_lvl):
            _depth = self.depth_net[lvl_id](img_feats[lvl_id])
            _depth = _depth.softmax(dim=1)
            pred_depth.append(_depth)
        return pred_depth