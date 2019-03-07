#!/usr/bin/env bash

set -Ceu

if [ "$#" -ne 2 ]; then
    echo "Usage: ./scripts/tools/create_coco214_json.sh COCO2017_DIR OUTPUT_DIR"
    exit 1
fi

coco2017_dir=$1
output_dir=$2
if [ -d ${output_dir} ]; then
    echo "Output directory already exists"
    exit 1
fi
mkdir -p ${output_dir}

wget -c https://pjreddie.com/media/files/coco/5k.part -O /tmp/5k.part

python -u scripts/tools/create_coco2014_json.py \
    --val_list /tmp/5k.part \
    --coco2017_dir ${coco2017_dir} \
    --outdir ${output_dir}
