import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class TPVAggregator(BaseModule):
    def __init__(
            self,
            tpv_h,  # 100
            tpv_w,  # 100
            tpv_z,  # 8
            in_dims=64,  # 80
            hidden_dims=128,  # 160
            out_dims=None,  # 80
            use_checkpoint=True
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.use_checkpoint = use_checkpoint

    def forward(self, tpv_list):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.tpv_h, -1)

        fused = tpv_hw + tpv_zh + tpv_wz
        fused = fused.permute(0, 2, 3, 4, 1)
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
        else:
            fused = self.decoder(fused)
        fused = fused.permute(0, 4, 1, 2, 3)
        return fused
