#!/usr/bin/env bash

set -Ceu

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/tools/download_coco2017.sh OUTPUT_DIR"
    exit 1
fi

output_dir=$1
if [ -d ${output_dir} ]; then
    echo "Output directory already exists"
    exit 1
fi
mkdir -p ${output_dir}

wget http://images.cocodataset.org/zips/train2017.zip -P ${output_dir}
wget http://images.cocodataset.org/zips/val2017.zip -P ${output_dir}
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ${output_dir}

unzip ${output_dir}/train2017.zip -d ${output_dir}
unzip ${output_dir}/val2017.zip -d ${output_dir}
unzip ${output_dir}/annotations_trainval2017.zip -d ${output_dir}
