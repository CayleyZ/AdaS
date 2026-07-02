# SDT V3 Segmentation Minimal Training Bundle

This folder contains the minimal local code needed to launch the SDT V3 semantic segmentation training entry point for the two ADE20K configs kept here.

## Included Configs

```text
configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py
configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k_hybrid.py
```

Only the base config files required by those two configs are included.

## Environment

This bundle was validated with:

```bash
/data/users/zhouzj/miniconda3/envs/openmmlab
```

The local `mmseg/` package in this folder should be preferred at runtime, so run commands with `PYTHONPATH=.` from the `segmentation/` directory.

## Run

```bash
cd segmentation
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k
```

For the hybrid optimizer config:

```bash
PYTHONPATH=. python tools/train.py configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k_hybrid.py --work-dir ./work_dirs/fpn_sdtv3_512x512_19M_ade20k_hybrid
```

## Notes

- Training data is expected at `/data/dataset/ADE20K/ADEChallengeData2016`.
- The configs reference a pretrained checkpoint at `/data/users/zhangjy/projects/sdt-v3-main/SDT_V3/Detection/configs/pretrained/V3_19.0M_1x4.pth`. That checkpoint is not included in this repository.
- Runtime outputs, logs, work directories, checkpoints, caches, and pretrained weights are intentionally ignored.
