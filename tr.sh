#!/usr/bin/env bash

rm -rf work_dirs
rm -rf /home/haoyang/data/data/MdaCD/work_dirs

python tools/copy_clip_files.py LEVIRCD penalty_10_mask
bash tools/general/train.sh configs/MdaCD_LEVIRCD.py 1 --work-dir /home/haoyang/data/data/MdaCD/work_dirs/MdaCD_LEVIRCD

python tools/copy_clip_files.py SYSUCD penalty_100_mask
bash tools/general/train.sh configs/MdaCD_SYSUCD.py 1 --work-dir /home/haoyang/data/data/MdaCD/work_dirs/MdaCD_SYSUCD