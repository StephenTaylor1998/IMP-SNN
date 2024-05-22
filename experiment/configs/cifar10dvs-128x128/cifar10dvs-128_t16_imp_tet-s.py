_base_ = [
    '../_base_/models/vgg11_dvs.py',
    '../_base_/datasets/dvs_cifar10_spikformer.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(
        in_channels=2,
        layers=[1, 1, 2, 4, ],
        neuron_cfg=dict(
            detach_reset=True,
            surrogate_function=dict(type='Sigmoid'),
            type='LazyStateLIFNodeBeta',
            v_reset=None),
        type='VGG11',
        width=[64, 128, 256, 512, ]),
    head=dict(
        cal_acc=False,
        in_channels=512,
        loss=dict(
            label_smooth_val=0.01,
            loss_weight=1.0,
            num_classes=10,
            reduction='mean',
            type='LabelSmoothLoss'),
        num_classes=10,
        out_time_step=None,
        time_step_embed=None,
        type='TETLinearClsHead'),
    neck=dict(type='SpikeGlobalAveragePooling'),
    train_cfg=dict(augments=[
        dict(alpha=0.5, type='Mixup'),
        dict(alpha=1.0, type='CutMix'),
    ]),
    type='ImageClassifier')

optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(0.9, 0.999, ),
        eps=1e-08,
        lr=0.001,
        type='AdamW',
        weight_decay=0.06),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.time_embed': dict(decay_mult=0.0),
            '.init_state': dict(decay_mult=0.0),
        }
    ),
    type='AmpOptimWrapper')

# learning policy
param_scheduler = [
    dict(
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        start_factor=5e-06,
        type='LinearLR'),
    dict(begin=30, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR'),
]

train_cfg = dict(by_epoch=True, max_epochs=200, val_interval=1)
# train, val, test setting
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
auto_scale_lr = dict(base_batch_size=16)
