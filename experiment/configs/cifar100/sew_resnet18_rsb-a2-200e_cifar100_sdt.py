_base_ = [
    '../_base_/models/sew_resnet_cifar.py',
    '../_base_/datasets/cifar100_bs256_rsb_a12.py',
    '../_base_/schedules/imagenet_bs2048_rsb.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

# model settings
model = dict(
    head=dict(
        type='SpikeLinearClsHead',
        num_classes=100,
        in_channels=512,
        cal_acc=False,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]))

# dataset settings
train_dataloader = dict(sampler=dict(type='RepeatAugSampler', shuffle=True))

# schedule settings
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(lr=0.008),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=195,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=200)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
