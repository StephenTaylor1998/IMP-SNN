# dataset settings

dataset_type = 'NCaltech101'
time_step = 16
num_bins = 31

augmentation_space = {
    "Identity": ['torch.tensor(0.0)', False],
    "ShearX": ['torch.linspace(-0.3, 0.3, 31)', True],
    "ShearY": ['torch.linspace(-0.3, 0.3, 31)', True],
    "TranslateX": ['torch.linspace(-0.5, 5.0, 31)', True],
    "TranslateY": ['torch.linspace(-0.5, 5.0, 31)', True],
    "Rotate": ['torch.linspace(-30.0, 30.0, 31)', True],
    "Cutout": ['torch.linspace(1.0, 30.0, 31)', True],
}

data_preprocessor = dict(
    type='DVSPreprocessor',
    time_step=time_step,
    num_classes=101,
    to_rgb=False
)

train_pipeline = [
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ToTensor', keys=['img']),
    # dict(type='ResizeDVS', keys=['img'], scale=(128, 128)),
    dict(type='ToFloatTensor', keys=['img']),
    dict(type='RandomHorizontalFlipDVS', prob=0.5, keys=['img']),
    # SpikFormerDVS
    dict(type='SpikFormerDVS', keys=['img'], augmentation_space=augmentation_space),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='ToFloatTensor', keys=['img']),
    # dict(type='ResizeDVS', keys=['img'], scale=(128, 128)),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    # batch_size=32,
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=False,
        data_prefix='./data/ncaltech101',
        use_ckpt=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        time_step=time_step,
        data_type='frame',
        split_by='number',
        test_mode=True,
        data_prefix='./data/ncaltech101',
        use_ckpt=False,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1,))

test_dataloader = val_dataloader
test_evaluator = val_evaluator
