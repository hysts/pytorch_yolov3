#!/usr/bin/env bash

set -Ceu

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/tools/download_darknet_weights.sh OUTPUT_DIR"
    exit 1
fi

output_dir=$1
if [ -d ${output_dir} ]; then
    echo "Output directory already exists"
    exit 1
fi
mkdir -p ${output_dir}

wget https://pjreddie.com/media/files/yolov3.weights -P ${output_dir}
wget https://pjreddie.com/media/files/yolov3-tiny.weights -P ${output_dir}
wget https://pjreddie.com/media/files/darknet53.conv.74 -P ${output_dir}

python scripts/tools/convert_darknet_weight.py --weight_path ${output_dir}/yolov3.weights      --model_name yolov3      --outdir ${output_dir}
python scripts/tools/convert_darknet_weight.py --weight_path ${output_dir}/yolov3-tiny.weights --model_name yolov3-tiny --outdir ${output_dir}
python scripts/tools/convert_darknet_weight.py --weight_path ${output_dir}/darknet53.conv.74   --model_name darknet53   --outdir ${output_dir}
