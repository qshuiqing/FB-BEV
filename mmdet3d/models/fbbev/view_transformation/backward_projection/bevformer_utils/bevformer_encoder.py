# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import warnings

import torch
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import ext_loader

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class bevformer_encoder(TransformerLayerSequence):
    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 tpv_h,
                 tpv_w,
                 tpv_z,
                 pc_range=None,
                 grid_config=None,
                 data_config=None,
                 num_points_in_pillar=[4, 32, 32],
                 return_intermediate=False,
                 dataset_type='nuscenes',
                 fix_bug=False,
                 **kwargs):

        super(bevformer_encoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fix_bug = fix_bug
        self.x_bound = grid_config['x']
        self.y_bound = grid_config['y']
        self.z_bound = grid_config['z']
        self.final_dim = data_config['input_size']
        self.pc_range = pc_range
        self.fp16_enabled = False

        ref_3d_hw = self.get_reference_points(tpv_h, tpv_w, pc_range[5] - pc_range[2], num_points_in_pillar[0], '3d',
                                              device='cpu')  # (HW, num_points_in_pillar, 3)

        ref_3d_zh = self.get_reference_points(tpv_z, tpv_h, pc_range[3] - pc_range[0], num_points_in_pillar[1], '3d',
                                              device='cpu')
        ref_3d_zh = ref_3d_zh.permute(2, 0, 1)[[2, 0, 1]]  # (3, 800, 32)
        ref_3d_zh = ref_3d_zh.permute(1, 2, 0)  # (800, 32, 3)

        ref_3d_wz = self.get_reference_points(tpv_w, tpv_z, pc_range[4] - pc_range[1], num_points_in_pillar[2], '3d',
                                              device='cpu')
        ref_3d_wz = ref_3d_wz.permute(2, 0, 1)[[1, 2, 0]]
        ref_3d_wz = ref_3d_wz.permute(1, 2, 0)
        self.register_buffer('ref_3d_hw', ref_3d_hw)
        self.register_buffer('ref_3d_zh', ref_3d_zh)
        self.register_buffer('ref_3d_wz', ref_3d_wz)

        ref_2d_hw = self.get_reference_points(tpv_h, tpv_w, dim='2d', bs=1, device='cpu')
        self.register_buffer('ref_2d_hw', ref_2d_hw)

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in spatial cross-attn and self-attn.
        Args:
            H, W: spatial shape of tpv plane.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, -1).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, -1, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)  # (4, 100, 100, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(2, 0, 1)  # (HW, num_points_in_pillar, 3)
            # ref_3d = ref_3d.permute(1, 2, 0, 3)  # (100, 100, 4, 3)
            # ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # def get_reference_points(self, H, W, Z=8, dim='3d', bs=1, device='cuda', dtype=torch.float):
    #     """Get the reference points used in SCA and TSA.
    #     Args:
    #         H, W: spatial shape of bev.
    #         Z: hight of pillar.
    #         D: sample D points uniformly from each pillar.
    #         device (obj:`device`): The device where
    #             reference_points should be.
    #     Returns:
    #         Tensor: reference points used in decoder, has \
    #             shape (bs, num_keys, num_levels, 2).
    #     """
    #
    #     # reference points in 3D space, used in spatial cross-attention (SCA)
    #     if dim == '3d':
    #
    #         X = torch.arange(*self.x_bound, dtype=torch.float) + self.x_bound[-1] / 2
    #         Y = torch.arange(*self.y_bound, dtype=torch.float) + self.y_bound[-1] / 2
    #         Z = torch.arange(*self.z_bound, dtype=torch.float) + self.z_bound[-1] / 2
    #         Y, X, Z = torch.meshgrid([Y, X, Z])
    #         coords = torch.stack([X, Y, Z], dim=-1)
    #         coords = coords.to(dtype).to(device)
    #         # frustum = torch.cat([coords, torch.ones_like(coords[...,0:1])], dim=-1) #(x, y, z, 4)
    #         return coords
    #
    #     # reference points on 2D bev plane, used in temporal self-attention (TSA).
    #     elif dim == '2d':
    #         ref_y, ref_x = torch.meshgrid(
    #             torch.linspace(
    #                 0.5, H - 0.5, H, dtype=dtype, device=device),
    #             torch.linspace(
    #                 0.5, W - 0.5, W, dtype=dtype, device=device)
    #         )
    #         ref_y = ref_y.reshape(-1)[None] / H
    #         ref_x = ref_x.reshape(-1)[None] / W
    #         ref_2d = torch.stack((ref_x, ref_y), -1)
    #         ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    #         return ref_2d

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, pc_range, img_metas, cam_params=None, gt_bboxes_3d=None):

        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.final_dim

        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                     (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                     (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                     (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1)  # (1, 6, 10000, 4, 3)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 3,
                                                   3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= trans.view(B, N, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps),
                                          reference_points_cam[..., 2:3]], 4
                                         )
        reference_points_cam = post_rots.view(B, N, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(
            -1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 3)
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps)
                & (reference_points_cam[..., 0:1] < (1.0 - eps))
                & (reference_points_cam[..., 1:2] > eps)
                & (reference_points_cam[..., 1:2] < (1.0 - eps)))
        B, N, HW, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4).reshape(N, B, HW, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4).reshape(N, B, HW, D, 1).squeeze(-1)

        return reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]

    @auto_fp16()
    def forward(self,
                bev_query,  # list
                key,
                value,
                *args,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                cam_params=None,
                gt_bboxes_3d=None,
                pred_img_depth=None,
                bev_mask=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        reference_points_cams, tpv_masks, bev_query_depths = [], [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        for ref_3d in ref_3ds:
            reference_points_cam, per_cam_mask_list, bev_query_depth = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'], cam_params=cam_params, gt_bboxes_3d=gt_bboxes_3d)
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(per_cam_mask_list)
            bev_query_depths.append(bev_query_depth)

        for lid, layer in enumerate(self.layers):

            output = layer(  # BEVFormerEncoderLayer
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=self.ref_2d_hw,
                ref_3d=ref_3ds,  # unsued
                tpv_h=tpv_h,
                tpv_w=tpv_w,
                tpv_z=tpv_z,
                prev_bev=prev_bev,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cams,
                per_cam_mask_list=tpv_masks,
                bev_mask=bev_mask,
                bev_query_depth=bev_query_depths[0],
                pred_img_depth=pred_img_depth,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels=512,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) in {2, 4, 6}
        # assert set(operation_order) in set(['self_attn', 'norm', 'cross_attn', 'ffn'])

    @force_fp32()
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                # bev_h=None,
                # bev_w=None,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                bev_mask=None,
                bev_query_depth=None,
                per_cam_mask_list=None,
                lidar_bev=None,
                pred_img_depth=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query_0 = self.attentions[attn_index](
                    query[0],
                    None,
                    None,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos[0],
                    key_pos=bev_pos[0],
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=bev_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[tpv_h, tpv_w]], device=query[0].device),
                    level_start_index=torch.tensor([0], device=query[0].device),
                    **kwargs)
                attn_index += 1
                query = torch.cat([query_0, query[1], query[2]], dim=1)
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](  # DA_SpatialCrossAttention
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    bev_query_depth=bev_query_depth,
                    pred_img_depth=pred_img_depth,
                    bev_mask=bev_mask,
                    per_cam_mask_list=per_cam_mask_list,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        query = torch.split(query, [tpv_h * tpv_w, tpv_z * tpv_h, tpv_w * tpv_z], dim=1)
        return query
