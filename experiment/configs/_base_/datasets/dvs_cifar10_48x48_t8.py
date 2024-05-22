# dataset settings

dataset_type = 'CIFAR10DVSLegacy'
time_step = 8

augmentation_space = {
    "Identity": ['torch.tensor(0.0)', False],
    "ShearX": ['torch.linspace(-0.3, 0.3, 31)', True],
    "ShearY": ['torch.linspace(-0.3, 0.3, 31)', True],
    "TranslateX": ['torch.linspace(-5.0, 5.0, 31)', True],
    "TranslateY": ['torch.linspace(-5.0, 5.0, 31)', True],
    "Rotate": ['torch.linspace(-30.0, 30.0, 31)', True],
    "Cutout": ['torch.linspace(1.0, 30.0, 31)', True],
}

data_preprocessor = dict(
    type='DVSPreprocessor',
    time_step=time_step,
    num_classes=10,
    to_rgb=False
)

train_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='ResizeDVS', scale=(48, 48), keys=['img']),
    dict(type='RandomCropDVS', crop_size=(48, 48), padding=4),
    dict(type='RandomHorizontalFlipDVS', prob=0.5, keys=['img']),
    # dict(type='SpikFormerDVS', keys=['img'], augmentation_space=augmentation_space),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='ResizeDVS', scale=(48, 48), keys=['img']),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        train=True,
        data_prefix='./data/dvs-cifar10',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    pin_memory=True,
    pin_memory_device="cuda"
)

val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        train=False,
        data_prefix='./data/dvs-cifar10',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    pin_memory=True,
    pin_memory_device="cuda"
)
val_evaluator = dict(type='Accuracy', topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
