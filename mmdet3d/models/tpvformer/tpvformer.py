import torch
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS

from mmdet3d.models import builder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector


def generate_forward_transformation_matrix(bda):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class TPVFormer(MVXTwoStageDetector):

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 backward_projection=None,
                 # depth_net
                 depth_net=None,
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 occupancy_head=None,
                 fix_void=False,
                 **kwargs,
                 ):

        super(TPVFormer, self).__init__(**kwargs)

        if backward_projection:
            self.backward_projection = builder.build_head(backward_projection)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)
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

        self.fix_void = fix_void

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

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                              (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs == 1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:  # True
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                        **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)

        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]

        if self.with_pts_bbox:  # None
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]

        # if self.with_specific_component('occupancy_head'):
        pred_occupancy = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)['output_voxels'][0]

        pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
        if self.fix_void:
            pred_occupancy = pred_occupancy[..., 1:]
        pred_occupancy = pred_occupancy.softmax(-1)

        # convert to CVPR2023 Format
        pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
        pred_occupancy = torch.flip(pred_occupancy, [2])
        pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
        pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)

        if return_raw_occ:
            pred_occupancy_category = pred_occupancy
        else:
            pred_occupancy_category = pred_occupancy.argmax(-1)

        pred_occupancy_category = pred_occupancy_category.cpu().numpy()

        assert len(img_metas) == 1
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = None
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
        return bbox_list
