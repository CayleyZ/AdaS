# SDT V3 Segmentation Training Bundle

This folder contains the code needed to launch the SDT V3 semantic segmentation training entry point for the ADE20K configs.

## Included Configs

```text
configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py       # baseline with AdamW
configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k_adas.py  # AdaS optimizer
```

## Environment

Create a fresh environment and install all runtime dependencies from `requirements.txt`:

```bash
conda create -n adas-seg python=3.9 -y
conda activate adas-seg

pip install --upgrade pip setuptools wheel
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The local `mmseg/` package in this folder should be preferred at runtime, so run commands with `PYTHONPATH=.` from the `segmentation/` directory.

## Pretrained Checkpoint

The configs expect the SDT V3 19M pretrained checkpoint at:

```text
pretrained/V3_19.0M_1x4.pth
```

Download it from the GitHub release before training:

```bash
mkdir -p pretrained
wget -O pretrained/V3_19.0M_1x4.pth https://github.com/CayleyZ/AdaS/releases/download/segmentation-sdtv3-19m-pretrained/V3_19.0M_1x4.pth
sha256sum pretrained/V3_19.0M_1x4.pth
```

Expected SHA256:

```text
72adec1cabce8f9aacb4f9f16deaf37ae28c22e72d815d19e6c6fea0ec9b05a7
```

## Run

```bash
cd segmentation
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k
```

For the AdaS optimizer config:

```bash
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k_adas.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k_adas
```
