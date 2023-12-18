# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# we follow the online training settings  from solofusion
samples_per_gpu = 1
num_iters_per_epoch = 28130
num_epochs = 24
# checkpoint_epoch_interval = 1
# use_custom_eval_hook = True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

use_checkpoint = False

grid_config = {
    'x': [-40, 40, 0.8],  # 100
    'y': [-40, 40, 0.8],  # 100
    'z': [-1, 5.4, 0.8],  # 8
    'depth': [2.0, 42.0, 0.5],  # 80
}
depth_categories = 80
numC_Trans = 80
_ffn_dim_ = numC_Trans * 4
tpv_h_ = 100
tpv_w_ = 100
tpv_z_ = 8

_dim_ = 256
num_heads = 8

_num_levels_ = 4
_num_cams_ = 6

tpv_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0

empty_idx = 18  # noise 0-->255
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19

occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channel = 256
voxel_channels = [64, 64 * 2, 64 * 4]

self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=numC_Trans,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
        dict(
            type='TPVImageCrossAttention',
            pc_range=point_cloud_range,
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=numC_Trans,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=tpv_h_,
                tpv_w=tpv_w_,
                tpv_z=tpv_z_,
            ),
            embed_dims=numC_Trans,
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.0,
    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=numC_Trans,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.0,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)

model = dict(
    type='TPVFormer',
    img_backbone=dict(
        pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=use_checkpoint,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,  # 256
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',  # camera-aware depth net
        in_channels=_dim_,  # 256
        context_channels=numC_Trans,  # 80
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,  # 80
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    backward_projection=dict(
        type='TPVFormerHead',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        pc_range=point_cloud_range,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=numC_Trans,  # 80
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            num_layers=tpv_encoder_layers,
            pc_range=point_cloud_range,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer,
                self_cross_layer,
                self_cross_layer,
                self_layer,
                self_layer,
            ]),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=[30, 30, 20],
            h=tpv_h_,
            w=tpv_w_,
            z=tpv_z_
        ),
        tpv_aggregator=dict(
            type='TPVAggregator',
            tpv_h=tpv_h_,
            tpv_w=tpv_w_,
            tpv_z=tpv_z_,
            in_dims=numC_Trans,
            hidden_dims=2 * numC_Trans,
            out_dims=numC_Trans,
        ),
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        with_cp=use_checkpoint,
        block_strides=[1, 2, 2],
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    img_bev_encoder_neck=dict(
        type='FPN3D',
        with_cp=use_checkpoint,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occupancy_head=dict(
        type='OccHead',
        with_cp=use_checkpoint,
        use_focal_loss=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=occ_size,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy', 'gt_depth'
                                ])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(type='LoadOccupancy', occupancy_path=occupancy_path),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_occupancy', 'visible_mask'])
        ]
    )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    occupancy_path=occupancy_path,
    use_sequence_group_flag=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=6,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        img_info_prototype='bevdet',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=True,
        filter_empty_gt=filter_empty_gt,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 2e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch * num_epochs, ])
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch)
evaluation = dict(interval=20 * num_iters_per_epoch, pipeline=test_pipeline)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=10560,
#         priority='NORMAL',
#         interval=2 * num_iters_per_epoch,
#     ),
#     dict(
#         type='SequentialControlHook',
#         temporal_start_iter=num_iters_per_epoch * 2,
#     ),
# ]
load_from = 'ckpts/r50_256x705_depth_pretrain.pth'
fp16 = dict(loss_scale='dynamic')
