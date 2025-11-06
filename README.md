# Multimodal Difference Augmentation Learning for Remote Sensing Change Detection

## Implement Guide

### Data Preparation

This work uses two RSCD datasets: the LEVIR Lab's [LEVIR-CD](https://justchenhao.github.io/LEVIR/) and Q. Shi _et al._'s [SYSU-CD](https://github.com/liumency/SYSU-CD).

1. Download both datasets and reorganize them into this structure:

```
Dataset/LEVIRCD/train/time1/
Dataset/LEVIRCD/train/time2/
Dataset/LEVIRCD/train/label/
Dataset/LEVIRCD/val/...
Dataset/LEVIRCD/test/...
Dataset/SYSUCD/...
```

2. Run [`tools/write_path.py`](./tools/write_path.py) to generate three text files: `Dataset/LEVIRCD/train.txt`, `Dataset/LEVIRCD/val.txt`, and `Dataset/LEVIRCD/test.txt`. And modify the script to generate another three for the SYSU-CD dataset.

3. Download [official CLIP files for both datasets]() or generate your own CLIP files, then import them and reorganize them into this structure:

```
Dataset/clip_files/LEVIRCD/train/*.json
Dataset/clip_files/LEVIRCD/val/*.json
Dataset/clip_files/LEVIRCD/test/*.json
Dataset/clip_files/SYSUCD/...
```

You can also use custom datasets. Just follow the steps above. For unclear details, you can refer to [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP)'s data preparation guide.

### Environment Setup

### Training & Testing Commands

## License

This repository is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), 
which is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

All **original modifications, additions, and new code** contributed by Hao Yang 
are licensed under the **Hippocratic License 3.0 – NonCommercial (Customized)**.

### Summary of License Terms
- The original MMSegmentation components remain under **Apache License 2.0**.
- The new contributions by Hao Yang are under a **NonCommercial, No-Surveillance, No-Military** license.
- Any redistribution or derivative work must retain both licenses and comply with their respective terms.

Full text of the custom license is available in the file [`LICENSE.custom`](./LICENSE.custom).

## Acknowledgement

This work is built on OpenMMLab's [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and S. Dong _et al._'s [ChangeCLIP](https://github.com/dyzy41/ChangeCLIP). Thanks for their great work!

## Citation

```
@article{Yang2025,
title = {Multimodal Difference Augmentation Learning for Remote Sensing Change Detection},
journal = {IEE Transactions on Geoscience and Remote Sensing},
author = {Hao Yang and Zhiyu Jiang and Dandan Ma and Qi Wang}
}
```