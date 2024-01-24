# -*- coding: utf-8 -*-
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS, build_neck
from mmseg.ops import resize

from mmdet3d.core import bbox3d2result
from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class FastOCC(MVXTwoStageDetector):
    def __init__(
            self,
            neck_fuse,
            neck_3d,
            n_voxels,
            voxel_size,
            extrinsic_noise=0,
            seq_detach=False,
            multi_scale_id=None,
            multi_scale_3d_scaler=None,
            with_cp=False,
            backproject='inplace',
            style='v4',
            # bev encoder
            img_bev_encoder_backbone=None,
            img_bev_encoder_neck=None,
            # Deal with history
            do_history=True,
            interpolation_mode='bilinear',
            history_cat_num=16,
            history_cat_conv_out_channels=None,
            single_bev_num_channels=80,

            **kwargs
    ):
        super(FastOCC, self).__init__(**kwargs)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}',
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)

        # style v4: fastbev + img/bev ms
        self.style = style
        assert self.style in ['v1', 'v2', 'v3', 'v4'], self.style
        self.multi_scale_id = multi_scale_id
        self.multi_scale_3d_scaler = multi_scale_3d_scaler

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size

        # test time extrinsic noise
        self.extrinsic_noise = extrinsic_noise
        if self.extrinsic_noise > 0:
            for i in range(5):
                print("### extrnsic noise: {} ###".format(self.extrinsic_noise))

        self.backproject = backproject

        # checkpoint
        self.with_cp = with_cp

        # bev encoder
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) \
            if img_bev_encoder_backbone is not None else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) \
            if img_bev_encoder_neck is not None else None

        # # Deal with history
        # self.single_bev_num_channels = single_bev_num_channels
        # self.do_history = do_history
        # self.interpolation_mode = interpolation_mode
        # self.history_cat_num = history_cat_num
        # self.history_cam_sweep_freq = 0.5  # seconds between each frame
        # history_cat_conv_out_channels = (history_cat_conv_out_channels
        #                                  if history_cat_conv_out_channels is not None
        #                                  else self.single_bev_num_channels)
        # ## Embed each sample with its relative temporal offset with current timestep
        # conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        # self.history_keyframe_time_conv = nn.Sequential(
        #     conv(self.single_bev_num_channels + 1,
        #          self.single_bev_num_channels,
        #          kernel_size=1,
        #          padding=0,
        #          stride=1),
        #     nn.SyncBatchNorm(self.single_bev_num_channels),
        #     nn.ReLU(inplace=True))
        # ## Then concatenate and send them through an MLP.
        # self.history_keyframe_cat_conv = nn.Sequential(
        #     conv(self.single_bev_num_channels * (self.history_cat_num + 1),
        #          history_cat_conv_out_channels,
        #          kernel_size=1,
        #          padding=0,
        #          stride=1),
        #     nn.SyncBatchNorm(history_cat_conv_out_channels),
        #     nn.ReLU(inplace=True))
        # self.history_sweep_time = None
        # self.history_bev = None
        # self.history_bev_before_encoder = None
        # self.history_seq_ids = None
        # self.history_forward_augs = None
        # self.count = 0

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        ego2lidar = img_meta['ego2lidar']
        bda_mat = img_meta['bda_mat']
        for i in range(6):  # num_cam
            lidar2img = img_meta['lidar2img'][i]
            ego2img = lidar2img @ ego2lidar @ np.linalg.inv(bda_mat)
            ego2img = torch.tensor(ego2img, dtype=torch.float32)
            intrinsic = torch.eye(3, dtype=torch.float32)
            intrinsic[:2] /= stride
            if noise > 0:
                projection.append(intrinsic @ ego2img[:3] + noise)
            else:
                projection.append(intrinsic @ ego2img[:3])
        return torch.stack(projection)

    @force_fp32()
    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)

        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)

        if type(x) in [list, tuple]:
            x = x[0]

        return x

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda):  # align features with 3d shift

        voxel_feat = True if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3)  # n, c, z, h, w

        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx']
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence']
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)

        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1  # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum() > 0:
            if voxel_feat:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1,
                                                                                         1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)

            self.history_sweep_time[start_of_sequence] = 0  # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)

        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)

        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0

        tmp_bev = self.history_bev
        if voxel_feat:
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True,
                                            mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1)  # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev],
                              dim=1)  # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels,
            *feats_cat.shape[2:])  # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                    1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
                 ], dim=2)  # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                 ], dim=2)  # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
            feats_to_return.shape[0], feats_to_return.shape[1], -1,
            *feats_to_return.shape[3:])  # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:]))  # B x C x H x W or B x C x Z x H x W

        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()

    def extract_feat(self, points, img, img_metas):
        B = img.shape[0]  # B:1
        img = img.reshape([-1] + list(img.shape)[2:])  # [1,1,6,3,928,1600] -> [6,3,928,1600]

        x = self.img_backbone(img)  # [6,256,232,400]; [6,512,116,200]; [6,1024,58,100]; [6,2048,29,50]

        # fuse features
        def _inner_forward(x):
            out = self.img_neck(x)
            return out  # [6,64,232,400]; [6,64,116,200]; [6,64,58,100]; [6,64,29,50])

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)
        mlvl_feats = list(mlvl_feats)

        if self.multi_scale_id is not None:
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized_feat)

                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([B, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(  # (6,3,4)
                        img_meta, stride_i, noise=self.extrinsic_noise).to(feat_i.device)
                    if self.style in ['v1', 'v2']:
                        # wo/ bev ms
                        n_voxels, voxel_size = self.n_voxels[0], self.voxel_size[0]
                    else:
                        # v3/v4 bev ms
                        n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3,200,200,6]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["origin"]),
                    ).to(feat_i.device)

                    if self.backproject == 'inplace':
                        volume = backproject_inplace(
                            feat_i[:, :, :height, :width], points, projection)  # [c, vx, vy, vz]
                    else:
                        volume, valid = backproject_vanilla(
                            feat_i[:, :, :height, :width], points, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])

            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])

        if self.style in ['v1', 'v2']:
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, lvl*seq*c, vx, vy, vz]
        else:
            # bev ms: multi-scale bev map (different x/y/z)
            for i in range(len(mlvl_volumes)):
                mlvl_volume = mlvl_volumes[i]
                bs, c, x, y, z = mlvl_volume.shape
                # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
                mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z * c).permute(0, 3, 1, 2)

                # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
                if self.multi_scale_3d_scaler == 'pool' and i != (len(mlvl_volumes) - 1):
                    # pooling to bottom level
                    mlvl_volume = F.adaptive_avg_pool2d(mlvl_volume, mlvl_volumes[-1].size()[2:4])
                elif self.multi_scale_3d_scaler == 'upsample' and i != 0:
                    # upsampling to top level 
                    mlvl_volume = resize(
                        mlvl_volume,
                        mlvl_volumes[0].size()[2:4],
                        mode='bilinear',
                        align_corners=False)
                else:
                    # same x/y
                    pass

                # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
                mlvl_volume = mlvl_volume.unsqueeze(-1)
                mlvl_volumes[i] = mlvl_volume
            mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes

        def _inner_forward(x):
            # v1/v2: [bs, lvl*seq*c, vx, vy, vz] -> [bs, c', vx, vy]
            # v3/v4: [bs, z1*c1+z2*c2+..., vx, vy, 1] -> [bs, c', vx, vy]
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        x = self.bev_encoder(x)

        return x  # 1,256,200,200

    def forward_pts_train(self,
                          feature_bev,
                          voxel_semantics,
                          mask_camera,
                          img_metas,
                          ):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        # FastOccHead
        outs = self.pts_bbox_head(feature_bev)
        loss_inputs = [voxel_semantics, mask_camera, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_train(self,
                      img_metas=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      img=None,
                      **kwargs
                      ):
        feature_bev = self.extract_feat(None, img, img_metas)
        """
        feature_bev: [(1, 256, 100, 100)]
        """

        losses = dict()
        losses_pts = self.forward_pts_train(feature_bev, voxel_semantics, mask_camera, img_metas)
        losses.update(losses_pts)

        return losses

    def forward_test(self, img, img_metas, **kwargs):

        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas):

        feature_bev = self.extract_feat(None, img, img_metas)

        outs = self.pts_bbox_head(feature_bev)
        occ = self.pts_bbox_head.get_occ(outs)

        return occ

    def aug_test(self, imgs, img_metas):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):
            img_metas[0]['img_shape'] = img_shape_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24 * tta_id:24 * (tta_id + 1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    ys, xs, zs = torch.meshgrid(
        [
            torch.arange(n_voxels[1]),
            torch.arange(n_voxels[0]),
            torch.arange(n_voxels[2]),
        ]
    )
    points = torch.stack((xs, ys, zs))
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


def backproject_inplace(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [64, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]

    # method2：特征填充，只填充有效特征，重复特征直接覆盖
    volume = torch.zeros(
        (n_channels, points.shape[-1]), device=features.device
    ).type_as(features)
    for i in range(n_images):
        volume[:, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]

    volume = volume.view(n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume
