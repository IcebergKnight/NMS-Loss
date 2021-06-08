# model settings
model = dict(
    type='RPN_RGBRCNN',
    pretrained='modelzoo://resnet50',
    rcnn_pretrained='modelzoo://resnet50',
    fix_rpn = False,
    fix_rcnn = False,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=[3],
        frozen_stages=1,
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        style='pytorch'),
    neck=dict(
        type='MaskAttentionNeck',
        in_channels=[1024],
        mask_feat_channels=256,
        loss_mask_weight=1,
        output_indices = [0]),
    rpn_head=dict(
        type='RPN_NMS_Head',
        in_channels=1024+256,
        feat_channels=512,
        anchor_scales=[1.5, 2.1, 2.9, 3.9, 5.3, 7.2, 10.1, 14.5, 22.5],
        anchor_ratios=[1/0.41],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        nms_loss=dict(type='FinalNMSLoss', 
            pull_weight = 0.1, push_weight = 0.001, nms_thr = 0.5, 
            add_gt = True, push_select=True, use_score = True,
            pull_relax = True, push_relax = False, fix_push_score = True,
            fix_push_reg = False, fix_pull_score=True, fix_pull_reg=False)),
    bbox_roi_extractor=dict(
        type='Img_ROI_Extractor',
        roi_layer=dict(type='RoIAlign', out_size=(272, 112), sample_num=2),
        expand_ratio=0.25,
        ),
    rcnn_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=[2,3],
        frozen_stages=1,
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        style='pytorch'),
    bbox_head=dict(
        type='WekSegBBoxHead',
        max_feat_size=(17, 7),
        in_channels=[1024, 1024],
        out_channel = 256,
        mask_feat_channel = 256,
        loss_mask_weight=1,
        dropout_rate = 0.3,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=False, 
            loss_weight=1.0),
        ))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=0.5),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=1.0,
        debug=False,
        adaptive_pos_weight = False),
    rpn_proposal=dict(
        nms_across_levels = False,
        nms_type = 'nms',
        nms_pre=2500,
        nms_post=2500,
        max_num=2500,
        min_score = 0.005,
        nms_thr=-1,
        min_bbox_size=50),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=0.5),
        sampler=dict(
            type='RandomSampler',
            num=100,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels = True,
        nms_type = 'nms',
        nms_pre=10000,
        nms_post=10000,
        max_num=1000,
        nms_thr=0.5,
        min_score = 0.005,
        min_bbox_size=50),
    rcnn=dict(
        score_thr = 1e-9,
        nms = None,
        max_per_img = 25,
        merge_mode = 'merge',
    )
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'CityPersonsDataset'
data_root = '/data/research/citypersons/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
             type=dataset_type,
             ann_file=data_root + 'annotations/anno_train.mat',
             img_prefix=data_root + 'images/train/',
             img_scale=(1024, 512),
             img_norm_cfg=img_norm_cfg,
             size_divisor=32,
             flip_ratio=0.5,
             filter_empty=True,
             with_mask=False,
             with_crowd=True,
             with_label=True,
             extra_aug=dict(random_crop_d=dict(crop_size=0.5)),
             resize_keep_ratio=False)
             ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/anno_val.mat',
        img_prefix=data_root + 'images/val/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/anno_val.mat',
        img_prefix=data_root + 'images/val/',
        img_scale=(2048, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cityperons'
load_from = None
resume_from = None
workflow = [('train', 1)]
