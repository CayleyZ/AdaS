# CIFAR10-DVS QKFormer AdaS Experiment

This folder contains the code needed to run the CIFAR10-DVS QKFormer training experiment.

## Files

```text
train.py
model.py
utils.py
optim_factory.py
optimizer.py
autoaugment.py
requirements.txt
```

## Environment

Install the complete pinned Python dependency set:

```bash
pip install -r requirements.txt
```

The CuPy backend also needs CUDA toolkit headers. On the verified server, use:

```bash
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
```

## Run

```bash
cd cifar10-dvs
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-path /data/dataset/CIFAR10DVS/ \
  --device cuda \
  --epochs 96 \
  --batch-size 16 \
  --workers 4 \
  --output-dir ./logs
```
