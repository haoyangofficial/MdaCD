#!/usr/bin/env bash

rm -rf work_dirs
rm -rf /data/work_dirs/

python tools/copy_clip_files.py LEVIRCD 10_100_mask
bash tools/general/train.sh configs/MdaCD_LEVIRCD.py 2 --work-dir /data/work_dirs/MdaCD_LEVIRCD
bash tools/general/test.sh LEVIR configs/MdaCD_LEVIRCD.py 2 /data/work_dirs/MdaCD_LEVIRCD

python tools/copy_clip_files.py SYSUCD 10_1000_mask
bash tools/general/train.sh configs/MdaCD_SYSUCD.py 2 --work-dir /data/work_dirs/MdaCD_SYSUCD
bash tools/general/test.sh SYSU configs/MdaCD_SYSUCD.py 2 /data/work_dirs/MdaCD_SYSUCD