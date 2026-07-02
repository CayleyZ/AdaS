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

## Run

```bash
cd segmentation
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k
```

For the AdaS optimizer config:

```bash
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k_adas.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k_adas
```
