# model settings
model = dict(
    type='CascadeRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
       # frozen_stages=1,
       # norm_cfg=dict(type='BN', requires_grad=True),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        norm_eval=True,
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        style='pytorch'),
    neck=[dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        add_extra_convs='on_input',
        num_outs=6,
      #  norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        upsample_cfg=dict(mode='bilinear')
        ),
        # dict(
            # type='DilaRMoudle',
            # in_channel=256,
            # num_levels=6,
            # norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            # share=True
        # )
        dict(
            type='Inception',
            in_channel=256,
            num_levels=6,
        #    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            share=True
        )
    ],
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=2**(4 / 3),
            scales_per_octave=3,
            ratios=[1.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 0.2, 0.2]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.3,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.45,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.35),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=100),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=100),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    gpu_assign_thr=100),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.45),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.45),
            max_per_img=100)))

### dataset
dataset_type = 'CocoDataset'
data_root = '/data/Dataset/DarkFace/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(0.1, 0.1), crop_type='relative_range'),
    #dict(type='RandomSquareCrop', crop_choice=[0.1, 0.3, 0.45, 0.6, 0.8, 1.0]),
   # dict(type='Resize', img_scale=[(10000, 1280), (10000, 640)], keep_ratio=True),
    dict(type='Resize', img_scale=[(2600, 1080), (2600, 720)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1080, 1620)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'DarkFace_Train_COCO.json',
        img_prefix=data_root + 'Images/darkface/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'DarkFace_Val_COCO.json',
        img_prefix=data_root + 'Images/darkface/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'DarkFace_Val_COCO.json',
        img_prefix=data_root + 'Images/darkface/',
        pipeline=test_pipeline))
        
evaluation = dict(interval=5, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[60, 100, 120])
runner = dict(type='EpochBasedRunner', max_epochs=160)

# 
checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level   = 'INFO'
load_from   = None
resume_from = None
workflow    = [('train', 1)]

# fp16 settings
fp16 = dict(loss_scale=512.)

### CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh configs/darkface/cascade_rcnn_r50_fpn_dconv_c3-c5_darface.py 4


