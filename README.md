# Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks

This repository is the official implementation of [Rethinking the Membrane Dynamics and Optimization Objectives of Spiking Neural Networks]. 


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
