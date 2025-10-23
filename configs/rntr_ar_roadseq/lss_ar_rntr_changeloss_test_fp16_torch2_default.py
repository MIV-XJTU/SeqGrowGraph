_base_ = [
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/default_runtime.py',
    '../../../../configs/_base_/schedules/cyclic-20e.py'
]

custom_imports = dict(imports=['projects.SeqGrowGraph.seq_grow_graph'])
randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)

load_from ='ckpts/lss_roadseg_48x32_b4x8_resnet_adam_24e_default.pth'
vis_dir="default"
resume = False
transformer_dims = 256
transformer_layers = 6
head_dims = 32
num_center_classes = 576
label_class_weight = [1.0] * num_center_classes
connect_class_weight = [1.0] * num_center_classes
label_class_weight[201] = 0.2
connect_class_weight[250] = 0.2

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
grid_conf = dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0],)
bz_grid_conf = dict(
        xbound=[-55.0, 55.0, 0.5],
        ybound=[-55.0, 55.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0],)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
input_modality = dict(use_camera=True, use_lidar=False)
ida_aug_conf = {
    "resize_lim": (0.193, 0.225),
    "final_dim": (128, 352),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
}

model = dict(
    vis_dir=vis_dir,
    type='AR_RNTR',
    use_grid_mask=False,
    freeze_pretrain=True,
    img_backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[512, 1024, 2048],
        out_channels=transformer_dims,
        num_outs=3),
    lss_cfg=dict(downsample=8, 
                d_in=transformer_dims, 
                d_out=transformer_dims),
    grid_conf=grid_conf,
    bz_grid_conf=bz_grid_conf,
    # vis_cfg=dict(path='val_lss_prycon_l6_256_150'),
    data_aug_conf=ida_aug_conf,
    pts_bbox_head=dict(
        type='ARRNTRHead',
        num_classes=10,
        in_channels=transformer_dims,
        max_center_len=601,
        num_center_classes=num_center_classes,
        embed_dims=transformer_dims,
        num_query=900,
        LID=True,
        with_position=False,
        with_multiview=False,
        with_fpe=False,
        with_time=False,
        with_multi=False,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='LssSeqLineTransformer',
            decoder=dict(
                type='PETRTransformerLineDecoder',
                return_intermediate=True,
                num_layers=transformer_layers,
                transformerlayers=dict(
                    type='PETRLineTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='RNTR2MultiheadAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1),
                        dict(
                            type='RNTR2MultiheadAttention',
                            embed_dims=transformer_dims,
                            num_heads=transformer_dims // head_dims,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=transformer_dims,
                        feedforward_channels=transformer_dims * 4,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    with_cp=False,  ###use checkpoint to save memory
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=transformer_dims//2, normalize=True),
        bev_positional_encoding=dict(
                     type='PositionEmbeddingSineBEV',
                     num_feats=transformer_dims//2,
                     normalize=True),
        loss_coords=dict(
            type='mmdet.CrossEntropyLoss'),
        loss_labels=dict(
            type='mmdet.CrossEntropyLoss', class_weight=label_class_weight),
        loss_connects=dict(
            type='mmdet.CrossEntropyLoss', class_weight=connect_class_weight), 
        loss_coeffs=dict(
            type='mmdet.CrossEntropyLoss')
            ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4)))

dataset_type = 'CenterlineNuScenesDataset'
data_root = './data/nuscenes/'

backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_centerline_infos_train.pkl',
    rate=1.0)

train_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='CenterlineFlip', prob=0.5),
    dict(type='CenterlineRotateScale', prob=0.5, max_rotate_degree=22.5, scaling_ratio_range=(0.95, 1.05)),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='Pack3DDetInputs', 
         keys=['img'], 
         meta_keys=('filename', 'token', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control')),
]

test_pipeline = [
    dict(type='OrgLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='LoadNusOrderedBzCenterline', grid_conf=grid_conf, bz_grid_conf=bz_grid_conf),
    dict(type='TransformOrderedBzLane2Graph', n_control=3, orderedDFS=True),
    dict(type='Pack3DDetInputs', keys=['img'], 
         meta_keys=('filename', 'token', 'ori_shape', 'img_shape', 'lidar2img', 'intrinsics', 'extrinsics',
                'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                'img_norm_cfg', 'sample_idx', 'timestamp', 'centerline_coord', 'centerline_label', 
                'centerline_connect', 'centerline_coeff', 'centerline_sequence', 'lidar2ego', 'n_control'))
]

train_dataloader = dict(
    batch_size=24,
    num_workers=10,  # 4
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_centerline_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        test_mode=False,
        use_valid_flag=True,
        metainfo=metainfo,
        box_type_3d='LiDAR', 
        backend_args=backend_args,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf,
        ))

test_dataloader = dict(
    # batch_size=60,
    dataset=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        ann_file='nuscenes_centerline_infos_val.pkl', 
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        modality=input_modality, 
        metainfo=metainfo,
        test_mode=True,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf))

val_dataloader = dict(
    # batch_size=30,
    dataset=dict(
        type=dataset_type, 
        pipeline=test_pipeline, 
        ann_file='nuscenes_centerline_infos_val.pkl', 
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT'),
        modality=input_modality, 
        metainfo=metainfo,
        test_mode=True,
        grid_conf=grid_conf,
        bz_grid_conf=bz_grid_conf))

optim_wrapper = dict(
    # TODO Add Amp
    # type='AmpOptimWrapper',
    # loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

num_epochs = 300
# num_epochs = 1

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        # TODO Figure out what T_max
        T_max=num_epochs,
        by_epoch=True,
    )
]

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

val_evaluator = dict(
    type='NuScenesReachMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_centerline_infos_val.pkl',
    metric='ar_reach',
    backend_args=backend_args, 
    grid_conf=grid_conf,
    bz_grid_conf=bz_grid_conf,
    jsonfile_prefix='save_result',
    landmark_thresholds=[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    reach_thresholds=[1, 2, 3, 4, 5],
    )
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=num_epochs, val_interval=num_epochs+1)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=1, max_keep_ckpts=10)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
