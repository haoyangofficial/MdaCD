#!/usr/bin/env bash

python tools/copy_clip_files.py SYSUCD penalty_100_mask
bash tools/general/test.sh SYSU configs/MdaCD_SYSUCD.py 1 /home/haoyang/data/data/MdaCD/work_dirs/MdaCD_SYSUCD