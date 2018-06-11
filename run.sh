#!/bin/bash
set -e
gpu_ids=$1
config_file=$2
CUDA_VISIBLE_DEVICES="$gpu_ids" python demo.py --config "$config_file"
