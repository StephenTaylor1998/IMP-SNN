_base_ = [
    '../_base_/models/sew_resnet18.py',
    '../_base_/datasets/imagenet100_bs128_rsb_a3.py',
    '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='StaticPreprocessor',
    time_step=4,
)

model = dict(
    backbone=dict(neuron_cfg=dict(type='IFNode')),
    head=dict(num_classes=100, out_time_step=[3, ])
)

# optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
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
auto_scale_lr = dict(base_batch_size=64)
