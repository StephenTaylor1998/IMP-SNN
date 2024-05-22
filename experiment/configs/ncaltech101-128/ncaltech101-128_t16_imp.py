_base_ = [
    '../_base_/models/vgg11_dvs.py',
    '../_base_/datasets/ncaltech101.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        neuron_cfg=dict(
            type='LazyStateLIFNode',
        ),
        in_channels=2,
    ),
    head=dict(
        type='TETLinearClsHead',
        num_classes=101,
        in_channels=512,
        out_time_step=None,
        cal_acc=False,
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.5),
        # dict(type='CutMix', alpha=1.0)
    ])
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.06,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_grad=dict(max_norm=1.0),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-8 / 2e-3,
        by_epoch=True,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=30)
]

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
auto_scale_lr = dict(base_batch_size=16)
