# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VGG11R48x48Legacy',
        tau=0.25
    ),
    head=dict(
        type='TETClsHead',
        cal_acc=True,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    )
)
