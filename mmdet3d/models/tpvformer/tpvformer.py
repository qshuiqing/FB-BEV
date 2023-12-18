import warnings

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS

from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .grid_mask import GridMask


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class TPVFormer(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 backward_projection=None,
                 pretrained=None,

                 # depth_net
                 depth_net=None,
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 occupancy_head=None,
                 **kwargs,
                 ):

        super(TPVFormer, self).__init__(**kwargs)

        if backward_projection:
            self.backward_projection = builder.build_head(backward_projection)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)

        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        # self.grid_mask = GridMask(
        #     True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        if forward_projection:
            self.forward_projection = builder.build_neck(forward_projection)
        if depth_net:
            self.depth_net = builder.build_head(depth_net)
        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        if occupancy_head:
            self.occupancy_head = builder.build_head(occupancy_head)

        self.history_sweep_time = None
        self.history_bev = None
        # self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        # self.count = 0

    def image_encoder(self, img):
        imgs = img  # [1, 6, 3, 256, 704]
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)  # [6, 1024, 16, 44], [6, 2048, 8, 22]

        if self.with_img_neck:
            x = self.img_neck(x)  # [6, 128, 16, 44]
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x

    @force_fp32()
    def bev_encoder(self, x):

        x = self.img_bev_encoder_backbone(x)

        x = self.img_bev_encoder_neck(x)

        if type(x) not in [list, tuple]:
            x = [x]

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

    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}

        context = self.image_encoder(img[0])  # [1, 6, 128, 16, 44]
        cam_params = img[1:7]
        mlp_input = self.depth_net.get_mlp_input(*cam_params)  # [1, 6, 27]
        context, depth = self.depth_net(context, mlp_input)  # [1, 6, 80, 16, 44]
        return_map['depth'] = depth
        return_map['context'] = context

        bev_feat = self.forward_projection(cam_params, context, depth, **kwargs)  # [1, 80, 100, 100, 8]
        return_map['cam_params'] = cam_params

        bev_feat_refined = self.backward_projection([context],  # TPVFormerHead
                                                    img_metas,
                                                    lss_bev=bev_feat.mean(-1),
                                                    cam_params=cam_params,
                                                    )

        bev_feat = bev_feat_refined + bev_feat  # [1, 80, 100, 100, 8]

        # Fuse History
        # bev_feat = self.fuse_history(bev_feat, img_metas, img[6])

        bev_feat = self.bev_encoder(bev_feat)  # [1, 256, 100, 100, 8]; [1, 256, 50, 50, 4]; [1, 256, 25, 25, 2]
        return_map['img_bev_feat'] = bev_feat

        return return_map

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        return self.extract_img_bev_feat(img, img_metas, **kwargs)

    def forward_train(self,
                      points=None,  # None
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,  # None
                      gt_bboxes=None,  # None
                      img_inputs=None,
                      proposals=None,  # None
                      gt_bboxes_ignore=None,  # None
                      gt_occupancy_flow=None,  # None
                      **kwargs):
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
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'],
                                                             results=results,
                                                             gt_occupancy=kwargs['gt_occupancy'],
                                                             gt_occupancy_flow=gt_occupancy_flow)
        losses.update(losses_occupancy)

        loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
        losses.update(loss_depth)

        return losses
