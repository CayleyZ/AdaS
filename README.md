# AdaS: Adaptive Gradient Descent for Spiking Transformers

Official implementation for **"AdaS: Adaptive Gradient Descent for Spiking Transformers"**.

AdaS is an optimizer designed for Spiking Transformers. The paper identifies an excessive parameter-update noise problem caused by the combination of surrogate-gradient learning and adaptive optimization. AdaS mitigates this issue by adaptively balancing the adaptive update direction with a momentum-based gradient update direction, keeping the update noise at a useful level rather than simply removing it.

This repository provides compact, runnable experiment bundles for the main released code paths used in the paper.

## Overview

Spiking Transformers combine Transformer-style representation learning with the event-driven efficiency of spiking neural networks. However, their non-differentiable spike functions require surrogate gradients during training, which can introduce additional update noise. When this noise is combined with the noise already present in adaptive optimizers such as AdamW, training can become less stable and the final performance can degrade.

AdaS addresses this by updating parameters with a weighted combination of two components:

```text
adaptive update component + momentum-based gradient update component
```

The balancing coefficient is computed adaptively from the update statistics and a target noise level. In practice, AdaS can be integrated into AdamW-style optimizers without adding extra optimizer-state memory, because it reuses the first-order momentum already maintained by the adaptive optimizer.

## Repository Structure

```text
AdaS/
|-- spikelm/          # SpikeLM GLUE fine-tuning with AdaS
|-- cifar10-dvs/      # QKFormer CIFAR10-DVS experiment with AdaS
`-- segmentation/     # SDT V3 ADE20K semantic segmentation with AdaS
```

Each subfolder is a self-contained training bundle with its own `README.md`, `requirements.txt`, and run instructions.

## Experiments

### SpikeLM on GLUE

The `spikelm/` folder contains the SpikeLM fine-tuning code used for NLP experiments on GLUE.

```bash
cd spikelm
bash download_weights.sh
```

Then follow [spikelm/README.md](spikelm/README.md) to create the environment and run GLUE fine-tuning.

In the paper, SpikeLM with AdaS improves the average GLUE score over AdamW:

```text
SpikeLM + AdamW: 76.5 average
SpikeLM + AdaS : 77.6 average
```

### QKFormer on CIFAR10-DVS

The `cifar10-dvs/` folder contains the CIFAR10-DVS QKFormer experiment.

```bash
cd cifar10-dvs
pip install -r requirements.txt
```

Then follow [cifar10-dvs/README.md](cifar10-dvs/README.md) for the training command.

Reported CIFAR10-DVS accuracy:

```text
QKFormer + AdamW: 84.0
QKFormer + AdaS : 85.1
```

### SDT V3 Segmentation on ADE20K

The `segmentation/` folder contains the SDT V3 semantic segmentation experiment on ADE20K. It includes both the baseline AdamW config and the AdaS config.

```bash
cd segmentation
mkdir -p pretrained
wget -O pretrained/V3_19.0M_1x4.pth \
  https://github.com/CayleyZ/AdaS/releases/download/segmentation-sdtv3-19m-pretrained/V3_19.0M_1x4.pth
```

Then follow [segmentation/README.md](segmentation/README.md) to install dependencies and launch training.

Reported ADE20K mIoU:

```text
E-SpikeFormer + AdamW: 38.2
E-SpikeFormer + AdaS : 40.2
```

### SDTrack on FE108 and VisEvent

The SDTrack tracking experiment is not included in this repository. To reproduce it, please refer to the official SDTrack repository:

[YmShan/SDTrack](https://github.com/YmShan/SDTrack)

Only the optimizer needs to be changed to AdaS to reproduce the AdaS tracking experiments.

Reported SDTrack-Tiny results:

```text
FE108:
  AdamW: AUC 59.0, PR 91.3
  AdaS : AUC 60.2, PR 92.5

VisEvent:
  AdamW: AUC 35.6, PR 49.2
  AdaS : AUC 36.3, PR 50.5
```

## Using AdaS in Your Own Project

AdaS follows the same usage pattern as standard PyTorch optimizers. Replace an AdamW-style optimizer with the AdaS implementation provided in the relevant experiment folder.

For example, the CIFAR10-DVS experiment provides:

```text
cifar10-dvs/optimizer.py
```

and the segmentation experiment provides:

```text
segmentation/mmseg/engine/optimizers/adas.py
```

The main hyperparameter introduced by AdaS is `gamma`, the target update-noise level. The released experiments keep the values used for the corresponding paper results inside their training scripts or config files.

## Checkpoints

Large pretrained weights are not tracked by git. They are provided through GitHub Releases:

- SpikeLM checkpoint: [spikelm-step-100000](https://github.com/CayleyZ/AdaS/releases/tag/spikelm-step-100000)
- SDT V3 segmentation checkpoint: [segmentation-sdtv3-19m-pretrained](https://github.com/CayleyZ/AdaS/releases/tag/segmentation-sdtv3-19m-pretrained)

## Citation

If this repository is useful for your research, please cite:

```text
AdaS: Adaptive Gradient Descent for Spiking Transformers
Zijian Zhou, Honglin Cao, Ammar Belatreche, Wenjie Wei, Yimeng Shan,
Yu Liang, Yu Yang, Shuai Wang, Yalan Ye, Malu Zhang, Yang Yang, Haizhou Li.
Proceedings of the 43rd International Conference on Machine Learning, 2026.
```

## Acknowledgements

This repository builds on several excellent open-source projects:

- [Xingrun-Xing/SpikeLM](https://github.com/Xingrun-Xing/SpikeLM)
- [zhouchenlin2096/qkformer](https://github.com/zhouchenlin2096/qkformer)
- [biclab/spike-driven-transformer-v3](https://github.com/biclab/spike-driven-transformer-v3)

We sincerely thank the authors and contributors of these projects for releasing their code.
