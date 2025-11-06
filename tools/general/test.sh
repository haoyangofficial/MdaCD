#!/bin/bash

data_name=$1
config_file=$2
num_gpu=$3
work_dirs=$4
model_checkpoint=$(find "$work_dirs" -name 'best_mIoU_iter_*.pth' -type f -print -quit)
echo $model_checkpoint
if [ "$data_name" == "SYSU" ]; then
    label_dir="/home/haoyang/data/data/MdaCD/Dataset/SYSUCD/test/label"
    bash tools/general/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
elif [ "$data_name" == "LEVIR" ]; then
    label_dir="/home/haoyang/data/data/MdaCD/Dataset/LEVIRCD/test/label"
    bash tools/general/dist_test.sh "$config_file" "$model_checkpoint" $num_gpu --out "$work_dirs/test_result"
    python tools/general/metric.py --pppred "$work_dirs/test_result" --gggt "$label_dir"
fi