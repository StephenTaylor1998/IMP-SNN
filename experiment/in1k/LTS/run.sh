#!/usr/bin/env sh

# TET 62.9167
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet18 \
  --output-dir ./logs --tb --print-freq 96 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet18-f37072fd.pth \
  --tet -b 192

# SDT 63.2067
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet18 \
  --output-dir ./logs --tb --print-freq 96 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet18-f37072fd.pth \
  -b 192

# LST 64.3287
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train_olts.py --cos_lr_T 320 --model sew_resnet18 \
  --output-dir ./logs --tb --print-freq 96 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet18-f37072fd.pth \
  --olts -b 192

# TET 67.98
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet34 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet34-b627a593.pth \
  --tet -b 128

# SDT 68.10
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet34 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet34-b627a593.pth \
  -b 128

# LST 68.10
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train_olts.py --cos_lr_T 320 --model sew_resnet34 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet34-b627a593.pth \
  --olts -b 128

# TET 69.87
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet50 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet50-0676ba61.pth \
  --tet -b 48

# SDT 70.33
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train.py --cos_lr_T 320 --model sew_resnet50 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet50-0676ba61.pth \
  -b 48

# LST 71.24
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
  --use_env train_olts.py --cos_lr_T 320 --model sew_resnet50 \
  --output-dir ./logs --tb --print-freq 256 --amp \
  --cache-dataset --T 4 --lr 0.05 --epoch 320 \
  --data-path imagenet \
  --load resnet50-0676ba61.pth \
  --olts -b 48
