#!/usr/bin/env sh

# ImageNet100
CUDA_VISIBLE_DEVICES=0 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_lst.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_sdt.py
CUDA_VISIBLE_DEVICES=2 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_tet.py
CUDA_VISIBLE_DEVICES=0 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_lst_imp.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_sdt_imp.py
CUDA_VISIBLE_DEVICES=2 python main.py configs/imagenet100/sew_resnet18_bs64_ra_in100_tet_imp.py

# cifar10dvs-48
CUDA_VISIBLE_DEVICES=0 python main.py configs/cifar10dvs-48x48/cifar10dvs-48_t8_imp_tet-s.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/cifar10dvs-48x48/cifar10dvs-48_t10_imp.py
CUDA_VISIBLE_DEVICES=2 python main.py configs/cifar10dvs-48x48/cifar10dvs-48_t10_imp_tet-s.py

# cifar10dvs-128
CUDA_VISIBLE_DEVICES=0 python main.py configs/cifar10dvs-128x128/cifar10dvs-128_t16_imp.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/cifar10dvs-128x128/cifar10dvs-128_t16_imp_tet-s.py

# ncaltech101-48
CUDA_VISIBLE_DEVICES=0 python main.py configs/ncaltech101-48/ncaltech101-48_t10_imp.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/ncaltech101-48/ncaltech101-48_t10_imp_tet-s.py

# ncaltech101-128
CUDA_VISIBLE_DEVICES=0 python main.py configs/ncaltech101-128/ncaltech101-128_t16_imp.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/ncaltech101-128/ncaltech101-128_t16_imp_tet-s.py

# cifar100
CUDA_VISIBLE_DEVICES=0 python main.py configs/cifar100/sew_resnet18_rsb-a2-200e_cifar100_sdt.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/cifar100/sew_resnet18_rsb-a2-200e_cifar100_tet.py

# cifar10
CUDA_VISIBLE_DEVICES=0 python main.py configs/cifar100/sew_resnet18_rsb-a2-200e_cifar10_sdt.py
CUDA_VISIBLE_DEVICES=1 python main.py configs/cifar100/sew_resnet18_rsb-a2-200e_cifar10_tet.py
