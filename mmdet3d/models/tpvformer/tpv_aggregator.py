import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import HEADS


@HEADS.register_module()
class TPVAggregator(BaseModule):
    def __init__(
            self,
            tpv_h,
            tpv_w,
            tpv_z,
            # nbr_classes=20,
            in_dims=64,
            hidden_dims=128,
            out_dims=None,
            scale_h=2,
            scale_w=2,
            scale_z=2,
            use_checkpoint=True
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        # self.classifier = nn.Linear(out_dims, nbr_classes)
        # self.classes = nbr_classes
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

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw,
                size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh,
                size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz,
                size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
                mode='bilinear'
            )

        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z * self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w * self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h * self.tpv_h, -1)

        fused = tpv_hw + tpv_zh + tpv_wz
        fused = fused.permute(0, 2, 3, 4, 1)
        if self.use_checkpoint:
            fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
            # logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        else:
            fused = self.decoder(fused)
            # logits = self.classifier(fused)
        # logits = logits.permute(0, 4, 1, 2, 3)
        fused = fused.permute(0, 4, 1, 2, 3)

        return fused
