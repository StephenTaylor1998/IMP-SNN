# Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks

This repository is the official implementation of [Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks]. 

The code used for the neuromorphic dataset experiments is available at [amazing-cls](https://github.com/AmazingLab/amazing-cls). 

**Warning:** 
This [code](https://github.com/AmazingLab/amazing-cls) may cause crashes (GPU not detected) on older CPUs (e.g., E5-2680v4) due to incompatibilities between **MMCV** and **SJ/CuPy**. To prevent potential issues, we strongly recommend not proceeding with further development using this version. We are in the process of migrating this project to **TIMM** framework, and a new version will be released soon. This upcoming release will incorporate our optimized CUDA kernels, delivering enhanced efficiency and an improved user experience. **Stay tuned for updates!**

# 警告
由于**MMCV**与**SJ/CuPy**存在兼容性问题，[代码](https://github.com/AmazingLab/amazing-cls)在较早的CPU（如**E5-2680v4**）上可能导致掉卡问题。我们建议停止基于此版本的开发，以避免风险外溢。目前，项目正在通过**TIMM**重新构建。新版本将集成优化后的**CUDA**内核，提供更高效率和更好的用户体验。

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
# recommend
sh train.sh 

# or
python main.py [path-to-config]

# example
python main.py configs/cifar10dvs-128x128/cifar10dvs-128_t16_imp.py
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
# recommend
python test.py [path-to-config] [path-to-checkpoint]

# example
python test.py configs/cifar10dvs-128x128/cifar10dvs-128_t16_imp.py work_dirs/cifar10dvs-128_t16_imp/last_checkpoint
```
