model = dict(
    type='GFL',
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        depth=101,
        num_stages=4,
        scales=4,
        base_width=26,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFocalHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.3],
            octave_base_scale=2.5198420997897464,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.250),
        reg_max=16,
        reg_topk=4,
        reg_channels=64,
        add_mean=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
        
    # training and testing settings
    train_cfg = dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

### dataset
dataset_type = 'CocoDataset'
data_root    = '/data/Dataset/darkface_ann/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(20000, 2600), (20000, 4000)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomCrop', crop_size = (2000, 2000)),
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
        img_scale=(10000, 3300),
        #img_scale=[(20000, 2600), (20000, 3300), (20000, 4000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
    ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Train.json',
        img_prefix=data_root + 'dark_face_annotated_20210422195402/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Val.json',
        img_prefix=data_root + 'dark_face_annotated_20210422195402/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Val.json',
        img_prefix=data_root + 'dark_face_annotated_20210422195402/',
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
    step=[24, 33])
total_epochs = 36

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(interval=50,hooks=[dict(type='TextLoggerHook'),])

# yapf:enable
dist_params = dict(backend='nccl')
log_level   = 'INFO'
load_from   = '/data/zzg/PreTrained/gfocal_r2n101_dcn_fpn_ms2x.pth'
resume_from = None
workflow = [('train', 1)]
