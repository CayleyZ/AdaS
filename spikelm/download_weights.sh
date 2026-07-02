#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
OUT_DIR=$SCRIPT_DIR/base_spike/step_100000
OUT_FILE=$OUT_DIR/pytorch_model.bin
URL=https://github.com/CayleyZ/AdaS/releases/download/spikelm-step-100000/pytorch_model.bin

mkdir -p $OUT_DIR
if [ -f $OUT_FILE ]; then
  echo Weight already exists: $OUT_FILE
  exit 0
fi

if command -v wget >/dev/null 2>&1; then
  wget -O $OUT_FILE $URL
elif command -v curl >/dev/null 2>&1; then
  curl -L $URL -o $OUT_FILE
else
  echo Please install wget or curl, then rerun this script. >&2
  exit 1
fi

echo Downloaded: $OUT_FILE
