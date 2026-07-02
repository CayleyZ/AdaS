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

The experiment was checked with:

```bash
/data/users/zhouzj/miniconda3/envs/spikelm
```

Install the complete pinned Python dependency set:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains the dependency closure needed by this experiment, not only the packages missing from another environment.

`spikingjelly==0.0.0.0.14` is required because the code uses the legacy `spikingjelly.clock_driven` API. `cupy-cuda12x` is required by the CuPy backend used in the model. With Python 3.12, `timm==0.9.12` is used instead of the original `timm==0.6.12`, which is not compatible with Python 3.12.

The CuPy backend also needs CUDA toolkit headers. On the verified server, use:

```bash
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
```

## Data

The default dataset path is:

```text
/data/dataset/CIFAR10DVS/
```

The script expects SpikingJelly's preprocessed frame folders, for example:

```text
/data/dataset/CIFAR10DVS/frames_number_16_split_by_number/
```

## Run

```bash
cd cifar10-dvs-gama-1.0
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

For a quick smoke test, use one epoch, a small batch size, and a temporary output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-path /data/dataset/CIFAR10DVS/ \
  --device cuda \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --print-freq 1 \
  --output-dir ./smoke_logs
```
