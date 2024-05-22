# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEWResNetCifar',
        block_type='SEWBasicBlock',
        layers=[2, 2, 2, 2],
        width=[64, 128, 256, 512],
        stride=[1, 2, 2, 2],
        in_channels=3,
        zero_init_residual=True,
        groups=1, width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        cnf_list=['add'],
        neuron_cfg=dict(
            type='IFNode',
            v_reset=None,
            surrogate_function=dict(
                type='ATan'
            )
        ),
    ),
    neck=dict(
        type='SpikeGlobalAveragePooling',
    ),
    head=dict(
        type='SpikeLinearClsHead',
        num_classes=10,
        in_channels=512,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)
