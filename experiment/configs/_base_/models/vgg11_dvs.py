# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG11',
        layers=[1, 1, 2, 4],
        width=[64, 128, 256, 512],
        in_channels=3,
        neuron_cfg=dict(
            type='LIFNode',
            v_reset=None,  # Todo: check here {default: v_reset=0.}
            detach_reset=True,  # Todo: check here {default: detach_reset=False}
            surrogate_function=dict(
                type='Sigmoid'
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
        time_step_embed=None,
        out_time_step=None,
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
