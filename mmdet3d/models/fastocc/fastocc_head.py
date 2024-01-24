# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from torch import nn

from mmdet3d.datasets.occ_metrics import Metric_mIoU


@HEADS.register_module()
class FastOCCHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 embed_dims=256,
                 out_dim=32,
                 pillar_h=16,
                 num_classes=18,
                 norm_cfg=dict(type='BN', ),
                 norm_cfg_3d=dict(type='BN3d', ),
                 use_3d=False,
                 use_conv=False,
                 loss_occ=None,
                 use_mask=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 **kwargs):
        super(FastOCCHead, self).__init__()

        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.out_dim = out_dim  # 256
        self.num_classes = num_classes
        self.use_mask = use_mask

        self.loss_occ = build_loss(loss_occ)

        self.pillar_h = pillar_h
        self.use_3d = use_3d
        self.use_conv = use_conv

        # if not use_3d:
        #     if use_conv:  # False
        #         use_bias = norm_cfg is None
        #         self.decoder = nn.Sequential(
        #             ConvModule(
        #                 self.embed_dims,
        #                 self.embed_dims,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 bias=use_bias,
        #                 norm_cfg=norm_cfg,
        #                 act_cfg=act_cfg),
        #             ConvModule(
        #                 self.embed_dims,
        #                 self.embed_dims * 2,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 bias=use_bias,
        #                 norm_cfg=norm_cfg,
        #                 act_cfg=act_cfg), )
        #
        #     else:
        #         self.decoder = nn.Sequential(
        #             nn.Linear(self.embed_dims, self.embed_dims * 2),
        #             nn.Softplus(),
        #             nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
        #         )
        # else:
        #     use_bias_3d = norm_cfg_3d is None
        #
        #     self.middle_dims = self.embed_dims // pillar_h
        #     self.decoder = nn.Sequential(
        #         ConvModule(
        #             self.middle_dims,
        #             self.out_dim,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #             bias=use_bias_3d,
        #             conv_cfg=dict(type='Conv3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),
        #         ConvModule(
        #             self.out_dim,
        #             self.out_dim,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1,
        #             bias=use_bias_3d,
        #             conv_cfg=dict(type='Conv3d'),
        #             norm_cfg=norm_cfg_3d,
        #             act_cfg=act_cfg),
        #     )

        self.final_conv = ConvModule(
            self.embed_dims,  # 256
            self.out_dim,  # 256
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )

        self.predicter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, num_classes * pillar_h),
        )

    def forward(self, bev_embed):
        """
        Args:
            bev_embed: (1,256,200,200) - (bs,c,dy,dx)
        Returns:
            occ_pred: (1,200,200,16,18) - (bs,dx,dy,dz,num_classes)
        """

        # (bs,c,dy,dx) -> (bs,dx,dy,c')
        occ_pred = self.final_conv(bev_embed).permute(0, 3, 2, 1)
        bs, dx, dy = occ_pred.shape[:3]
        # (bs,dx,dy,c') -> (bs,dx,dy,dz*num_classes)->(bs,dx,dy,dz,num_classes)
        occ_pred = self.predicter(occ_pred)
        occ_pred = occ_pred.view(bs, dx, dy, self.pillar_h, self.num_classes)

        return occ_pred

        # bs, _, X, Y = bev_embed.size()  # (1,256,200,200) bs,c,y,x
        # if self.use_3d:  # True
        #     outputs = self.decoder(bev_embed.view(bs, -1, self.pillar_h, X, Y))  # (1,16,16,200,200)->(1,32,16,200,200)
        #     outputs = outputs.permute(0, 4, 3, 2, 1)  # (1,200,200,16,32)
        #
        # elif self.use_conv:
        #
        #     outputs = self.decoder(bev_embed)
        #     outputs = outputs.view(bs, -1, self.pillar_h, X, Y).permute(0, 3, 4, 2, 1)
        # else:
        #     outputs = self.decoder(bev_embed.permute(0, 2, 3, 1))
        #     outputs = outputs.view(bs, X, Y, self.pillar_h, self.out_dim)
        # outputs = self.predicter(outputs)  # (1,200,200,16,32)->(1,200,200,16,18)
        #
        # return outputs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             ):

        loss_dict = dict()
        occ = preds_dicts
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        losses = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'] = losses

        # self.temp_metric(voxel_semantics,
        #                  mask_camera,
        #                  preds_dicts, )

        return loss_dict

    def temp_metric(self, voxel_semantics,
                    mask_camera,
                    preds_dicts):

        occ = self.get_occ(preds_dicts)

        tmp_preds_dicts = occ.detach().cpu().numpy()
        tmp_voxel_semantics = voxel_semantics.detach().cpu().numpy()
        tmp_mask_camera = mask_camera.detach().cpu().numpy().astype(bool)

        occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        occ_eval_metrics.add_batch(tmp_preds_dicts, tmp_voxel_semantics, None, tmp_mask_camera)

        occ_eval_metrics.count_miou()

    def loss_single(self, voxel_semantics, mask_camera, preds):
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)  # (1,200,200,16,18)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )
        return loss_occ

    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        occ_out = preds_dicts
        occ_score = occ_out.softmax(-1)
        occ_score = occ_score.argmax(-1)

        return occ_score
