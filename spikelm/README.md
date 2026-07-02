# SpikeLM Fine-Tuning Bundle

This folder contains a minimal SpikeLM fine-tuning bundle derived from the original SpikeBERT experiment. It keeps only the code, tokenizer/config files, and environment metadata needed to run `finetune.py`.

The large pretrained checkpoint is not tracked in git. Download it from the GitHub Release asset before training.

## Directory Layout

```text
spikelm/
├── README.md
├── requirements.txt
├── download_weights.sh
├── spike_bert_core.py
├── urils_spike.py
├── bert-base-uncased/
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── base_spike/
│   └── step_100000/
│       └── .gitkeep
└── spike_ft-10w/
    ├── finetune.py
    ├── spike_bert.py
    └── adas.py
```

After downloading the release asset, the checkpoint should exist at:

```text
spikelm/base_spike/step_100000/pytorch_model.bin
```

## Requirements

This bundle was verified with the following environment:

```bash
/data/users/zhouzj/miniconda3/envs/spikelm
```

Main package versions from that environment:

- Python 3.12.8
- PyTorch 2.5.1
- Transformers 4.47.1
- Datasets 3.2.0
- Evaluate 0.4.3
- Accelerate 1.10.1

A CUDA GPU is required. The model code initializes some tensors with `.cuda()`, so CPU-only execution will fail.

## Install

Create and activate a conda environment, then install the exported Python packages:

```bash
conda create -n spikelm python=3.12 -y
conda activate spikelm
pip install -r spikelm/requirements.txt
```

For GPU training, make sure your PyTorch build matches the CUDA driver on the machine. If necessary, install PyTorch first using the command recommended by the official PyTorch selector, then install the remaining packages from `requirements.txt`.

## Download the Checkpoint

Download the pretrained checkpoint from the release asset:

```bash
cd spikelm
bash download_weights.sh
```

The script downloads:

```text
https://github.com/CayleyZ/AdaS/releases/download/spikelm-step-100000/pytorch_model.bin
```

and saves it as:

```text
base_spike/step_100000/pytorch_model.bin
```

You can also download it manually and place it at the same path.

## Run GLUE Training

Run training commands from `spikelm/spike_ft-10w`, because `finetune.py` uses relative paths for the local tokenizer/config and checkpoint.

Example CoLA run:

```bash
cd spikelm/spike_ft-10w

CUDA_VISIBLE_DEVICES=0 HF_ENDPOINT=https://hf-mirror.com \
python finetune.py \
  --model_name_or_path ../bert-base-uncased \
  --task_name cola \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 50 \
  --output_dir ./res/cola/binary/ \
  --seed 41 \
  --lr_scheduler_type constant
```

`HF_ENDPOINT=https://hf-mirror.com` is optional, but useful when GLUE datasets or metrics need to be downloaded through the Hugging Face mirror.

Keep the trailing slash in `--output_dir`. The script writes the best checkpoint to `args.output_dir + best/`, so `./res/cola/binary/` produces `./res/cola/binary/best/`.

## Run With Local CSV or JSON Data

You can train with local files instead of GLUE. The files must include a `label` column and one or two text columns.

```bash
cd spikelm/spike_ft-10w

CUDA_VISIBLE_DEVICES=0 \
python finetune.py \
  --model_name_or_path ../bert-base-uncased \
  --train_file /path/to/train.csv \
  --validation_file /path/to/validation.csv \
  --max_length 128 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --output_dir ./res/custom/run1/ \
  --seed 41 \
  --lr_scheduler_type constant
```

## Verified Smoke Test

Before packaging, this bundle was verified on GPU with the original `spikelm` conda environment. The smoke test successfully loaded the local tokenizer/config, loaded `base_spike/step_100000/pytorch_model.bin`, completed one training step, ran evaluation, and saved model outputs.
