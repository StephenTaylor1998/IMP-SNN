_base_ = [
    '../_base_/models/vgg11_dvs_48x48.py',
    '../_base_/datasets/dvs_cifar10_48x48.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='StateVGG11R48x48Legacy',
        tau=0.25,
        time=10
    ),
    head=dict(
        type='TETClsHead',
        out_time_step=None,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.01,
            num_classes=10,
            reduction='mean',
            loss_weight=1.0),
        cal_acc=True,
    ),
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.te': dict(decay_mult=0.0),
            '.mem': dict(decay_mult=0.0),
        }
    ),
)
# learning policy
param_scheduler = [
    dict(type='CosineAnnealingLR', eta_min=0., by_epoch=True)
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
auto_scale_lr = dict(base_batch_size=32)
